import os

import piq
import torch
import torchvision
from torch import nn

from unsplit import minmax, unsplit

CKPT = os.environ.get('WAV2CLIP_CKPT', os.path.expanduser('~/wav2clip/ckpt/Wav2CLIP.pt'))
SPEC_SHAPE = (1, 1, 257, 454)  # 16 kHz · 10 s, n_fft 512 → 257 bins, hop 353 → 454 frames


def log_spec(wav):
    z = torch.stft(wav, 512, hop_length=353, window=torch.hann_window(512, device=wav.device),
                   center=True, pad_mode='reflect', return_complex=True)
    x = torch.log(z.abs().pow(2).unsqueeze(1) + 1e-7)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.flatten(1).std(dim=1).view(-1, 1, 1, 1)
    return (x - m) / (s + 1e-9)


class Client(nn.Module):

    def __init__(self, s, ckpt=CKPT):
        super().__init__()
        enc = torchvision.models.resnet18()
        enc.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        enc.fc = nn.Identity()
        sd = torch.load(ckpt, map_location='cpu')
        enc.load_state_dict({k[len('audio_encoder.'):]: v for k, v in sd.items()
                             if k.startswith('audio_encoder.') and k != 'audio_encoder.spectrogram.window'})
        self.enc = enc
        self.s = s
        self.eval()

    def forward(self, x):
        e = self.enc
        x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
        for blk in (list(e.layer1) + list(e.layer2) + list(e.layer3) + list(e.layer4))[:self.s]:
            x = blk(x)
        return x


def attack(act, s, device='cuda', **iters):
    clone = Client(s).to(device)
    return unsplit(act.to(device), clone, SPEC_SHAPE, tv_w=0.0, **iters)


def score(x_hat, gt):
    a, b = minmax(x_hat).clamp(0, 1), minmax(gt).clamp(0, 1)
    return {'ssim': float(piq.ssim(a, b, data_range=1.0)),
            'fsim': float(piq.fsim(a, b, data_range=1.0, chromatic=False)),
            'mse': float(((a - b) ** 2).mean())}
