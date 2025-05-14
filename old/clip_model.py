import torch
import torch.nn as nn
import clip


class CLIPMultiLabelClassifier(nn.Module):
    def __init__(self, device, num_subclasses):
        super().__init__()
        self.clip_model, _ = clip.load('ViT-B/32', device=device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.superclass_head = nn.Linear(self.clip_model.visual.output_dim, 4)
        self.subclass_head = nn.Linear(self.clip_model.visual.output_dim, num_subclasses + 1)

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images).float()

        superclass_logits = self.superclass_head(features)
        subclass_logits = self.subclass_head(features)
        return superclass_logits, subclass_logits
