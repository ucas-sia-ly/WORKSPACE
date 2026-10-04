"""Frozen Torch Hub descriptors, with an image-gradient-capable SALAD backbone."""
from __future__ import annotations

import types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SALAD_REPO = 'serizba/salad'
BOQ_REPO = 'amaralibey/Bag-of-Queries'


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def enable_salad_image_gradients(model):
    # Upstream DINOv2.forward freezes early blocks with no_grad and then detaches.
    # Keep the exact token/block/norm/reshape computation, but allow input grads.
    backbone = model.backbone
    if not all(hasattr(backbone, name) for name in
               ('model', 'norm_layer', 'return_token', 'num_channels')):
        raise RuntimeError('Unsupported SALAD backbone; inspect upstream before using guidance')

    def forward(self, image):
        batch, _, height, width = image.shape
        tokens = self.model.prepare_tokens_with_masks(image)
        for block in self.model.blocks:
            tokens = block(tokens)
        if self.norm_layer:
            tokens = self.model.norm(tokens)
        token = tokens[:, 0]
        features = tokens[:, 1:].reshape(
            batch, height // 14, width // 14, self.num_channels
        ).permute(0, 3, 1, 2)
        return (features, token) if self.return_token else features

    backbone.forward = types.MethodType(forward, backbone)
    return freeze(model)


def pil_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1).float()[None] / 255


class Descriptor:
    def __init__(self, model, kind='salad', device='cuda'):
        self.model = freeze(model).to(device=device, dtype=torch.float32)
        self.kind = kind
        self.device = device

    def preprocess(self, image):
        # SALAD eval.py: 322x322 bilinear + ImageNet normalization.
        # BoQ README: tensor bicubic antialiased 322x322 + same normalization.
        image = F.interpolate(image.float(), size=(322, 322),
                              mode='bilinear' if self.kind == 'salad' else 'bicubic',
                              align_corners=False, antialias=True)
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (image - mean) / std

    def __call__(self, image):
        # Deliberately no no_grad here: this is also used in the guidance graph.
        output = self.model(self.preprocess(image.to(self.device)))
        if isinstance(output, (tuple, list)):
            output = output[0]  # BoQ returns descriptor and attention maps.
        return F.normalize(output.float(), dim=-1)

    def from_pil(self, image):
        if self.kind == 'salad':
            # Official SALAD evaluation resizes PIL before ToTensor.
            image = image.resize((322, 322), Image.Resampling.BILINEAR)
        return self(pil_tensor(image))


def load_salad(device='cuda', repo=SALAD_REPO):
    model = torch.hub.load(repo, 'dinov2_salad', pretrained=True, trust_repo=True,
                           source='local' if Path(repo).is_dir() else 'github')
    return Descriptor(enable_salad_image_gradients(model), 'salad', device)


def load_boq(device='cuda', repo=BOQ_REPO):
    # Only the retrieval evaluator imports/calls this function.
    model = torch.hub.load(repo, 'get_trained_boq', backbone_name='dinov2',
                           output_dim=12288, trust_repo=True,
                           source='local' if Path(repo).is_dir() else 'github')
    return Descriptor(model, 'boq', device)
