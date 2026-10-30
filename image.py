import open_clip
import piq
import torch
from torch import nn

from unsplit import minmax, unsplit

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class Client(nn.Module):

    def __init__(self, s, pretrained='openai'):
        super().__init__()
        self.v = open_clip.create_model('ViT-B-32-quickgelu', pretrained=pretrained).visual.eval()
        self.s = s
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = self.v._embeds((x - self.mean) / self.std)
        for blk in self.v.transformer.resblocks[:self.s]:
            x = blk(x)
        return x


def attack(act, s, device='cuda', **iters):
    clone = Client(s).to(device)
    return unsplit(act.to(device), clone, (1, 3, 224, 224), tv_w=0.1 if s < 3 else 1.0, **iters)


def score(x_hat, gt):
    return float(piq.fsim(minmax(x_hat).clamp(0, 1), gt.clamp(0, 1), data_range=1.0))
