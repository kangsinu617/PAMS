import open_clip
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from torch import nn

SOT, EOT = 49406, 49407
tokenize = open_clip.get_tokenizer('ViT-B-32-quickgelu')  # 정답 id: tokenize([라벨])[0]
_rouge = rouge_scorer.RougeScorer(['rouge1'])


class Client(nn.Module):

    def __init__(self, s, pretrained='openai'):
        super().__init__()
        m = open_clip.create_model('ViT-B-32-quickgelu', pretrained=pretrained)
        del m.visual
        self.m = m.eval()
        self.s = s

    def forward(self, E):
        z = E + self.m.positional_embedding
        for blk in self.m.transformer.resblocks[:self.s + 1]:
            z = blk(z, attn_mask=self.m.attn_mask)
        return z


def decode(ids):
    hit = (ids == EOT).nonzero()
    if len(hit):
        ids = ids[:hit[0, 0] + 1]
    return open_clip.decode(ids.cpu()).replace('<start_of_text>', '').replace('<end_of_text>', '').strip()


def attack(act, s, device='cuda', target_index=0, main_iters=100, input_iters=100, model_iters=100,
           lr=1e-3, model_lr=5e-6):
    act = act.to(device)
    clone = Client(s).to(device)
    vocab = clone.m.token_embedding.weight.detach().clone()  # 공격자가 읽는 공개 토큰 표
    g = torch.Generator().manual_seed(target_index)
    E = torch.normal(float(vocab.mean()), float(vocab.std()), size=(1, 77, vocab.shape[-1]), generator=g)
    E = E.to(device=device, dtype=vocab.dtype).requires_grad_(True)
    opt = torch.optim.Adam([E], lr=lr)
    m_opt = torch.optim.AdamW(clone.parameters(), lr=model_lr)
    for _ in range(main_iters):
        for _ in range(input_iters):
            opt.zero_grad()
            F.mse_loss(clone(E), act).backward()
            opt.step()
        for _ in range(model_iters):
            m_opt.zero_grad()
            F.mse_loss(clone(E), act).backward()
            m_opt.step()
    return torch.cdist(E.detach()[0], vocab).argmin(dim=-1)


def score(ids, label):
    gt = tokenize([label])[0].tolist()
    n = gt.index(EOT) + 1
    ids = ids.cpu().tolist()
    tok_acc = sum(x == y for x, y in zip(gt[:n], ids)) / n
    ids[0] = SOT
    return {'tok_acc': tok_acc, 'rouge1_f1': _rouge.score(label, decode(torch.tensor(ids)))['rouge1'].fmeasure}
