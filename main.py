import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from torch.utils.data import Dataset, DataLoader, BatchSampler, random_split
from torchvision import transforms
from PIL import Image

from cnn_model import CNN
from clip_model import CLIPMultiLabelClassifier


# Create Dataset class for multilabel classification
class MultiClassImageDataset(Dataset):
    def __init__(self, ann_df, super_map_df, sub_map_df, img_dir, transform=None):
        self.ann_df = ann_df
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ann_df)

    def __getitem__(self, idx):
        img_name = self.ann_df['image'][idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        super_idx = self.ann_df['superclass_index'][idx]
        super_label = self.super_map_df['class'][super_idx]

        sub_idx = self.ann_df['subclass_index'][idx]
        sub_label = self.sub_map_df['class'][sub_idx]

        if self.transform:
            image = self.transform(image)

        return image, super_idx, super_label, sub_idx, sub_label


class MultiClassImageTestDataset(Dataset):
    def __init__(self, super_map_df, sub_map_df, img_dir, transform=None):
        self.super_map_df = super_map_df
        self.sub_map_df = sub_map_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):  # Count files in img_dir
        return len([fname for fname in os.listdir(self.img_dir)])

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name


train_ann_df = pd.read_csv('data/train_data.csv')
super_map_df = pd.read_csv('data/superclass_mapping.csv')
sub_map_df = pd.read_csv('data/subclass_mapping.csv')

train_img_dir = 'data/train_images'
test_img_dir = 'data/test_images'

# image_preprocessing = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0), std=(1)),
#     ]
# )
image_preprocessing = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        ),
    ]
)

# Create train and val split
train_dataset = MultiClassImageDataset(
    train_ann_df, super_map_df, sub_map_df, train_img_dir, transform=image_preprocessing
)
train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])

# Create test dataset
test_dataset = MultiClassImageTestDataset(
    super_map_df, sub_map_df, test_img_dir, transform=image_preprocessing
)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class Trainer:
    def __init__(
        self, model, criterion, optimizer, train_loader, val_loader, test_loader=None, device='cuda'
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_epoch(self):
        running_loss = 0.0
        for i, data in enumerate(self.train_loader):
            inputs, super_labels, sub_labels = (
                data[0].to(device),
                data[1].to(device),
                data[3].to(device),
            )

            self.optimizer.zero_grad()
            super_outputs, sub_outputs = self.model(inputs)
            loss = self.criterion(super_outputs, super_labels) + self.criterion(
                sub_outputs, sub_labels
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Training loss: {running_loss/i:.3f}')

    def validate_epoch(self):
        super_correct = 0
        sub_correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                inputs, super_labels, sub_labels = (
                    data[0].to(device),
                    data[1].to(device),
                    data[3].to(device),
                )

                super_outputs, sub_outputs = self.model(inputs)
                loss = self.criterion(super_outputs, super_labels) + self.criterion(
                    sub_outputs, sub_labels
                )
                _, super_predicted = torch.max(super_outputs.data, 1)
                _, sub_predicted = torch.max(sub_outputs.data, 1)

                total += super_labels.size(0)
                super_correct += (super_predicted == super_labels).sum().item()
                sub_correct += (sub_predicted == sub_labels).sum().item()
                running_loss += loss.item()

        print(f'Validation loss: {running_loss/i:.3f}')
        print(f'Validation superclass acc: {100 * super_correct / total:.2f} %')
        print(f'Validation subclass acc: {100 * sub_correct / total:.2f} %')

    def test(self, csv_name):
        if not self.test_loader:
            raise NotImplementedError('test_loader not specified')

        # Evaluate on test set, in this simple demo no special care is taken for novel/unseen classes
        test_predictions = {'image': [], 'superclass_index': [], 'subclass_index': []}
        total_super_unseen = 0
        total_sub_unseen = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, img_name = data[0].to(device), data[1]

                super_outputs, sub_outputs = self.model(inputs)

                # We convert with softmax to apply the threshold to probabilities, not logits.
                super_probs = F.softmax(super_outputs, dim=1)
                sub_probs = F.softmax(sub_outputs, dim=1)

                super_max, super_pred = torch.max(super_probs.data, 1)
                sub_max, sub_pred = torch.max(sub_probs.data, 1)

                super_pred_label = super_pred.item() if super_max.item() > 0.99 else 3
                sub_pred_label = sub_pred.item() if sub_max.item() > 0.95 else 87

                if not super_max.item() > 0.99:
                    total_super_unseen += 1
                if not sub_max.item() > 0.95:
                    total_sub_unseen += 1

                test_predictions['image'].append(img_name[0])
                test_predictions['superclass_index'].append(super_pred_label)
                test_predictions['subclass_index'].append(sub_pred_label)

        print(f'Total superclasses unseen: {total_super_unseen}')
        print(f'Total subclasses unseen: {total_sub_unseen}')

        test_predictions = pd.DataFrame(data=test_predictions)
        test_predictions.to_csv(csv_name, index=False)


# Init model and trainer
device = 'mps'
# device = 'cuda'
model = CLIPMultiLabelClassifier(device=device, num_subclasses=87).to(device)
# model = CNN(input_size=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, test_loader, device)

# Training loop
# for epoch in range(20):
#     print(f'Epoch {epoch+1}')
#     trainer.train_epoch()
#     trainer.validate_epoch()
#     print('')
# print('Finished training.')

# torch.save(model.state_dict(), 'clip_model.pth')
model.load_state_dict(torch.load('clip_model.pth', map_location=device))
model.to(device)

print('Saving...')
test_predictions = trainer.test('test_predictions_clip.csv')
print('Finished saving to CSV.')
