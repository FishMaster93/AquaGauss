"""Video backbone (S3D by default; X3D-M and MC3-18 also available, all Kinetics-400 pretrained) + intensity head
(CORAL-style ordinal head or 4-class softmax), used for the phase-0 adaptation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vcil.models.s3d import S3D, load_s3d_kinetics


def coral_loss(logits, targets, num_classes=4):
    ordinal = torch.stack([(targets > k).float() for k in range(num_classes - 1)], dim=1)
    return F.binary_cross_entropy_with_logits(logits, ordinal)


def coral_predict(logits):
    return torch.sigmoid(logits).sum(dim=-1).round().long().clamp(0, 3)


def make_loss_and_predict(head):
    """head='coral': ordinal head (three thresholds); head='ce': plain 4-class cross-entropy."""
    if head == 'coral':
        return coral_loss, coral_predict
    return F.cross_entropy, lambda logits: logits.argmax(dim=-1)


class TorchvisionVideoBackbone(nn.Module):
    """Wraps a torchvision / pytorchvideo Kinetics-400 video model with the same interface as S3D:
    input uint8 (B,T,H,W,3) or [0,1] float (B,T,3,H,W), output {'embedding': (B, dim)}."""

    def __init__(self, name):
        super().__init__()
        if name == 'x3d_m':
            from pytorchvideo.models.hub import x3d_m
            net = x3d_m(pretrained=True)
            net.blocks[-1].proj = nn.Identity()          # drop the 400-class layer, keep the 2048-d pooled feature
            net.blocks[-1].activation = None
            mean, std, self.dim = (0.45, 0.45, 0.45), (0.225, 0.225, 0.225), 2048
        elif name == 'mc3_18':
            from torchvision.models.video import MC3_18_Weights, mc3_18
            net = mc3_18(weights=MC3_18_Weights.KINETICS400_V1)
            net.fc = nn.Identity()
            mean, std, self.dim = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989), 512
        else:
            raise ValueError(name)
        self.net = net
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1, 1))

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.permute(0, 4, 1, 2, 3).float().div_(255.0)
        else:
            x = x.transpose(1, 2)
        emb = self.net((x - self.mean) / self.std).flatten(1)
        return {'embedding': emb}


def build_backbone(name):
    if name == 's3d':
        m = load_s3d_kinetics(S3D(400))
        m.dim = 1024
        return m
    return TorchvisionVideoBackbone(name)


class VideoModel(nn.Module):
    """mlp=False: dropout + linear head (used for the phase-0 adaptation).
    mlp=True : LayerNorm -> 512 -> ReLU -> Dropout -> output.
    """

    def __init__(self, head='coral', dropout=0.5, mlp=False, backbone='s3d'):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = build_backbone(backbone)
        d = self.backbone.dim
        out = 3 if head == 'coral' else 4                          # CORAL: 3 thresholds; CE: 4 classes
        if mlp:
            self.dropout = nn.Identity()
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(512, out))
        else:
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d, out)

    def forward(self, x):
        emb = self.backbone(x)['embedding']
        return self.head(self.dropout(emb)), emb
