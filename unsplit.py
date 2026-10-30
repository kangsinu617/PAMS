import torch
import torch.nn.functional as F


def tv(x):
    _, c, h, w = x.shape
    h_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).sum()
    w_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).sum()
    return (h_tv / (c * (h - 1) * w) + w_tv / (c * h * (w - 1))) / x.shape[0]


def minmax(x):
    lo, rng = x.amin(), x.amax() - x.amin()
    return (x - lo) / rng if float(rng) > 0 else torch.zeros_like(x)


def unsplit(act, clone, shape, *, tv_w, main_iters=100, input_iters=100, model_iters=100,
            lr=5e-6, input_lr=1e-3):
    x = torch.full(shape, 0.5, device=act.device, requires_grad=True)
    x_opt = torch.optim.Adam([x], lr=input_lr, amsgrad=True)
    m_opt = torch.optim.Adam(clone.parameters(), lr=lr, amsgrad=True)
    for _ in range(main_iters):
        for _ in range(input_iters):
            x_opt.zero_grad()
            loss = F.mse_loss(clone(x), act)
            if tv_w:
                loss = loss + tv_w * tv(x)
            loss.backward()
            x_opt.step()
        for _ in range(model_iters):
            m_opt.zero_grad()
            F.mse_loss(clone(x), act).backward()
            m_opt.step()
    return x.detach()
