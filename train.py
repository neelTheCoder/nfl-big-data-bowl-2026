"""
train.py — THE editable file of the autoresearch loop.

Baseline v2 = faithful scaled-down clone of the actual 1st-place training
code (reference/nfl2026-1st-place-train.py), verified against source:
  10 z-scored dynamic features x last 20 frames (+ mask channel),
  depthwise-conv SequenceStem, 12 static features, pre-norm SwiGLU
  transformer over players, conv decoder -> 48 x (dx,dy) normalized
  displacements + variance, separate aux heads for velocity/accel,
  loss = 0.2*GaussianNLL(traj) + 0.8*GaussianNLL(vel/acc),
  RAdam lr 5e-4 constant, EMA 0.9995 after 10% of budget,
  rotation/flip/frame-shift/xy-bias augmentation.
Scaled: input_dim_multiplier 64->32, dim1 256->192, ff 1024->768.
"""
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare
from prepare import P_MAX, T_IN, T_OUT

# ============================ <<AGENT EDITABLE>> ============================
T_USED = 20
MULT = 16              # channels per dynamic feature in the depthwise stem
DIM1 = 192             # player embedding / transformer width
DIM2 = 32              # decoder channels per output frame
N_LAYERS = 2
N_HEADS = 8
FF_DIM = 768
TFM_DROPOUT = 0.05
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 200.0
EMA_DECAY = 0.9995
EMA_AFTER_FRAC = 0.1   # start EMA after this fraction of the budget
AUG_ROT_P = 0.5
AUG_FLIP_P = 0.5
AUG_SHIFT_P = 0.675
AUG_SHIFT_MAX = 20
AUG_XBIAS_STD = 1.0
AUG_YBIAS_STD = 1.5
AUG_NOISE = 0.0        # sensor-noise scale: jitters raw x/y/s/a/dir/o inputs
DECODE = "cumsum"      # "cumsum" (2nd place, integrates deltas) | "direct"
                       # (true 1st-place head: per-frame displacement)
AUX_W = 0.0            # loss = (1-AUX_W)*traj + AUX_W*aux  (1st place value)
MIN_VAR = 1e-3
N_DYN = 11             # 10 features + mask channel
N_STATIC = 12
# =========================== <<END AGENT EDITABLE>> =========================

# Overnight driver overrides: HP_<NAME> env var sets the matching knob above,
# preserving its type. Lets overnight.py sweep configs without editing code.
for _k in ("T_USED", "MULT", "DIM1", "DIM2", "N_LAYERS", "N_HEADS", "FF_DIM",
           "TFM_DROPOUT", "BATCH", "LR", "WEIGHT_DECAY", "GRAD_CLIP", "EMA_DECAY",
           "EMA_AFTER_FRAC", "AUG_ROT_P", "AUG_FLIP_P", "AUG_SHIFT_P",
           "AUG_SHIFT_MAX", "AUG_XBIAS_STD", "AUG_YBIAS_STD", "AUG_NOISE",
           "DECODE", "AUX_W", "MIN_VAR"):
    _v = os.environ.get("HP_" + _k)
    if _v is not None:
        globals()[_k] = type(globals()[_k])(_v)

TIME_BUDGET = float(os.environ.get("TIME_BUDGET", 1200))
# STEP_BUDGET > 0 trains for a fixed number of optimizer steps instead of wall
# clock (hardware-independent; 1st place = ~37k steps/model)
STEP_BUDGET = int(os.environ.get("STEP_BUDGET", 0))
SEED = int(os.environ.get("SEED", 1337))
# FOLD >= 0 drops train samples where idx % 8 == FOLD (1st-place-style data
# folds for ensemble diversity; val weeks 17-18 are never touched)
FOLD = int(os.environ.get("HP_FOLD", -1))
DEVICE = os.environ.get(
    "DEVICE", "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu"))
EVAL_EVERY_SEC = float(os.environ.get("EVAL_EVERY_SEC", 240))

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# batch construction (numpy) — augmentation on raw arrays, then features
# ---------------------------------------------------------------------------
def _features(Xw, Mw, ball, role, n_eff, shift, stats):
    """Xw [B,P,T,6] raw (x y s a dir_rad o_rad) -> z-scored dyn/static feats."""
    x, y, s, a, dr, o = np.moveaxis(Xw, -1, 0)
    B, P, T = x.shape
    last_x, last_y = x[:, :, -1], y[:, :, -1]

    recv = role == 1
    rx = (x * recv[:, :, None]).sum(1, keepdims=True)
    ry = (y * recv[:, :, None]).sum(1, keepdims=True)
    pas = role == 0
    px = (last_x * pas).sum(1)          # [B]
    py = (last_y * pas).sum(1)

    dyn = np.stack([
        x, y,
        np.sin(o), np.cos(o),
        ball[:, 0, None, None] - x, ball[:, 1, None, None] - y,
        rx - x, ry - y,
        np.sin(dr) * s, np.cos(dr) * s,
    ], -1)
    dyn = (dyn - stats["dyn_mean"]) / stats["dyn_std"]
    dyn = np.concatenate([dyn, Mw[..., None].astype(np.float32)], -1)
    dyn = np.nan_to_num(dyn) * Mw[..., None]

    role_oh = np.eye(4, dtype=np.float32)[role]
    static = np.concatenate([
        role_oh,
        ((n_eff - stats["nout_mean"]) / stats["nout_std"])[..., None],
        np.broadcast_to(((shift - 3.0) / 3.0)[:, None, None], (B, P, 1)),
        np.broadcast_to(((px - stats["px_mean"]) / stats["px_std"])[:, None, None], (B, P, 1)),
        np.broadcast_to(((py - stats["py_mean"]) / stats["py_std"])[:, None, None], (B, P, 1)),
        np.broadcast_to(((ball[:, 0] - stats["bx_mean"]) / stats["bx_std"])[:, None, None], (B, P, 1)),
        np.broadcast_to(((ball[:, 1] - stats["by_mean"]) / stats["by_std"])[:, None, None], (B, P, 1)),
        ((last_x - stats["x_mean"]) / stats["x_std"])[..., None],
        ((last_y - stats["y_mean"]) / stats["y_std"])[..., None],
    ], -1).astype(np.float32)
    return np.nan_to_num(dyn.astype(np.float32)), np.nan_to_num(static)


def make_batch(d, idx, stats, rng=None, train=True, flip_all=False):
    X = d["X"][idx].copy()
    in_mask = d["in_mask"][idx]
    Y = d["Y"][idx].copy()
    out_mask = d["out_mask"][idx].copy()
    ball = d["ball_land"][idx].copy()
    role = d["role"][idx]
    n_out = d["n_out"][idx].astype(np.int32)
    B = len(idx)

    # --- frame shift (start prediction before the throw), 1st-place style ---
    shift = np.zeros(B, np.int32)
    if train and rng is not None and AUG_SHIFT_P > 0:
        nmax = n_out.max(1)
        cap = np.minimum(AUG_SHIFT_MAX, T_OUT - nmax)
        do = rng.random(B) < AUG_SHIFT_P
        raw = np.abs(rng.normal(0, np.maximum(cap, 1) / 3.0))
        shift = (np.clip(np.round(raw), 0, cap) * do).astype(np.int32)

    Xw = np.zeros((B, P_MAX, T_USED, 6), np.float32)
    Mw = np.zeros((B, P_MAX, T_USED), bool)
    Yw = np.zeros((B, P_MAX, T_OUT, 2), np.float32)
    OMw = np.zeros((B, P_MAX, T_OUT), bool)
    for b in range(B):
        k = int(shift[b])
        Xw[b] = X[b, :, T_IN - T_USED - k:T_IN - k]
        Mw[b] = in_mask[b, :, T_IN - T_USED - k:T_IN - k]
        if k > 0:
            Yw[b, :, :k] = X[b, :, T_IN - k:, 0:2]
            OMw[b, :, :k] = in_mask[b, :, T_IN - k:]
            Yw[b, :, k:] = Y[b, :, :T_OUT - k]
            OMw[b, :, k:] = out_mask[b, :, :T_OUT - k]
        else:
            Yw[b], OMw[b] = Y[b], out_mask[b]
    n_eff = np.minimum(n_out + shift[:, None], T_OUT).astype(np.float32)

    if flip_all:   # deterministic mirror for test-time augmentation
        Xw[:, :, :, 1] = prepare.FIELD_Y - Xw[:, :, :, 1]
        Yw[:, :, :, 1] = prepare.FIELD_Y - Yw[:, :, :, 1]
        ball[:, 1] = prepare.FIELD_Y - ball[:, 1]
        Xw[:, :, :, 4] = np.pi - Xw[:, :, :, 4]
        Xw[:, :, :, 5] = np.pi - Xw[:, :, :, 5]

    if train and rng is not None:
        # h-flip (mirror y)
        fl = rng.random(B) < AUG_FLIP_P
        Xw[fl, :, :, 1] = prepare.FIELD_Y - Xw[fl, :, :, 1]
        Yw[fl, :, :, 1] = prepare.FIELD_Y - Yw[fl, :, :, 1]
        ball[fl, 1] = prepare.FIELD_Y - ball[fl, 1]
        Xw[fl, :, :, 4] = np.pi - Xw[fl, :, :, 4]
        Xw[fl, :, :, 5] = np.pi - Xw[fl, :, :, 5]

        # uniform rotation about last-frame centroid
        rot = rng.random(B) < AUG_ROT_P
        ang = rng.uniform(0, 2 * np.pi, B).astype(np.float32) * rot
        ca, sa = np.cos(ang), np.sin(ang)
        cx = Xw[:, :, -1, 0].mean(1)[:, None, None]
        cy = Xw[:, :, -1, 1].mean(1)[:, None, None]

        def rotxy(px, py):
            dx, dy = px - cx, py - cy
            return (cx + ca[:, None, None] * dx - sa[:, None, None] * dy,
                    cy + sa[:, None, None] * dx + ca[:, None, None] * dy)

        Xw[:, :, :, 0], Xw[:, :, :, 1] = rotxy(Xw[:, :, :, 0], Xw[:, :, :, 1])
        Yw[:, :, :, 0], Yw[:, :, :, 1] = rotxy(Yw[:, :, :, 0], Yw[:, :, :, 1])
        bx, by = rotxy(ball[:, 0][:, None, None], ball[:, 1][:, None, None])
        ball = np.concatenate([bx[:, 0], by[:, 0]], 1)
        for c in (4, 5):
            ux, uy = np.sin(Xw[:, :, :, c]), np.cos(Xw[:, :, :, c])
            rx2 = ca[:, None, None] * ux - sa[:, None, None] * uy
            ry2 = sa[:, None, None] * ux + ca[:, None, None] * uy
            Xw[:, :, :, c] = np.arctan2(rx2, ry2)

        # global xy bias
        bxs = rng.normal(0, AUG_XBIAS_STD, B).astype(np.float32)[:, None, None]
        bys = rng.normal(0, AUG_YBIAS_STD, B).astype(np.float32)[:, None, None]
        Xw[:, :, :, 0] += bxs; Xw[:, :, :, 1] += bys
        Yw[:, :, :, 0] += bxs; Yw[:, :, :, 1] += bys
        ball[:, 0] += bxs[:, 0, 0]; ball[:, 1] += bys[:, 0, 0]

        # per-frame sensor noise on inputs only (targets untouched): simulates
        # season-to-season tracking recalibration drift; scaled by AUG_NOISE
        if AUG_NOISE > 0:
            shp = (B, P_MAX, T_USED)
            mw = Mw.astype(np.float32)
            for c, sd in ((0, 0.08), (1, 0.08), (2, 0.12), (3, 0.25),
                          (4, 0.05), (5, 0.05)):
                Xw[:, :, :, c] += (rng.normal(0, sd * AUG_NOISE, shp)
                                   .astype(np.float32) * mw)

    dyn, static = _features(Xw, Mw, ball, role, n_eff, shift.astype(np.float32), stats)

    last_xy = Xw[:, :, -1, 0:2]
    tgt = (Yw - last_xy[:, :, None, :] - stats["tgt_mean"]) / stats["tgt_std"]
    tgt = np.nan_to_num(tgt) * OMw[..., None]

    # aux targets: velocity/accel diffs incl. the last two input frames
    seq = np.concatenate([Xw[:, :, -2:, 0:2], Yw], 2)
    v = np.diff(seq, axis=2)
    acc = np.diff(v, axis=2)
    sa_t = np.concatenate([v[:, :, 1:], acc], -1)
    sa_t = (sa_t - stats["sa_mean"]) / stats["sa_std"]
    sa_t = np.nan_to_num(sa_t) * OMw[..., None]

    t = lambda a, dt: torch.as_tensor(a, dtype=dt)
    return dict(
        dyn=t(dyn, torch.float32), static=t(static, torch.float32),
        tgt=t(tgt, torch.float32), sa=t(sa_t, torch.float32),
        out_mask=t(OMw, torch.bool), valid=t(Mw.any(-1), torch.bool),
        last_xy=t(last_xy, torch.float32),
    )


def compute_stats(d):
    """z-score statistics from the training split (no aug, no shift)."""
    idx = d["train_idx"]
    X = d["X"][idx][:, :, T_IN - T_USED:]
    M = d["in_mask"][idx][:, :, T_IN - T_USED:]
    ball = d["ball_land"][idx]
    role = d["role"][idx]
    x, y, s, a, dr, o = np.moveaxis(X, -1, 0)
    recv = role == 1
    rx = (x * recv[:, :, None]).sum(1, keepdims=True)
    ry = (y * recv[:, :, None]).sum(1, keepdims=True)
    dyn = np.stack([
        x, y, np.sin(o), np.cos(o),
        ball[:, 0, None, None] - x, ball[:, 1, None, None] - y,
        rx - x, ry - y,
        np.sin(dr) * s, np.cos(dr) * s,
    ], -1)
    m = M[..., None] & np.ones_like(dyn, bool)
    mean = (dyn * m).sum((0, 1, 2)) / m.sum((0, 1, 2))
    std = np.sqrt(((dyn - mean) ** 2 * m).sum((0, 1, 2)) / m.sum((0, 1, 2)))

    last_x, last_y = x[:, :, -1], y[:, :, -1]
    pm = M.any(-1)
    Y = d["Y"][idx]
    OM = d["out_mask"][idx]
    dxy = Y - np.stack([last_x, last_y], -1)[:, :, None, :]
    om2 = OM[..., None]
    nvalid = om2.sum((0, 1, 2))
    tmean = (dxy * om2).sum((0, 1, 2)) / nvalid
    tstd = np.sqrt(((dxy - tmean) ** 2 * om2).sum((0, 1, 2)) / nvalid)

    seq = np.concatenate([X[:, :, -2:, 0:2], Y], 2)
    v = np.diff(seq, axis=2)
    acc = np.diff(v, axis=2)
    sa_all = np.concatenate([v[:, :, 1:], acc], -1)
    n1 = om2.sum()
    samean = (sa_all * om2).sum((0, 1, 2)) / n1
    sastd = np.sqrt(((sa_all - samean) ** 2 * om2).sum((0, 1, 2)) / n1)

    pas = role == 0
    px = (last_x * pas).sum(1); py = (last_y * pas).sum(1)
    n_out = d["n_out"][idx].astype(np.float32)
    return dict(
        dyn_mean=mean.astype(np.float32), dyn_std=std.astype(np.float32),
        tgt_mean=tmean.astype(np.float32), tgt_std=tstd.astype(np.float32),
        sa_mean=samean.astype(np.float32), sa_std=sastd.astype(np.float32),
        x_mean=last_x[pm].mean(), x_std=last_x[pm].std(),
        y_mean=last_y[pm].mean(), y_std=last_y[pm].std(),
        px_mean=px.mean(), px_std=px.std(), py_mean=py.mean(), py_std=py.std(),
        bx_mean=ball[:, 0].mean(), bx_std=ball[:, 0].std(),
        by_mean=ball[:, 1].mean(), by_std=ball[:, 1].std(),
        nout_mean=n_out[n_out > 0].mean(), nout_std=n_out[n_out > 0].std(),
    )


# ============================ <<AGENT EDITABLE>> ============================
class ConvBlock(nn.Module):
    def __init__(self, dim, k=3, groups=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Conv1d(dim, dim, k, groups=groups, padding=padding, bias=False),
            nn.BatchNorm1d(dim), nn.SiLU(),
            nn.Conv1d(dim, dim, k, groups=groups, padding=padding, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.block = ConvBlock(dim, k, padding=(k - 1) // 2)

    def forward(self, x):
        return x + self.block(x)


class SequenceStem(nn.Module):
    """Depthwise conv stem, no padding, crop-residuals; emphasizes last frame."""
    def __init__(self, in_dim, mult, k=3):
        super().__init__()
        dim = in_dim * mult
        self.stem = nn.Conv1d(in_dim, dim, k, groups=in_dim, bias=False)
        self.blocks = nn.ModuleList([
            ConvBlock(dim, k, groups=in_dim),
            ConvBlock(dim, k, groups=in_dim),
            ConvBlock(dim, k, groups=in_dim // 2 if in_dim % 2 == 0 else in_dim),
        ])
        self.crop = (k - 1) * 2

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = x[:, :, self.crop:] + blk(x)
        return x


class TfmLayer(nn.Module):
    def __init__(self, d, heads, ff, p):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.w1 = nn.Linear(d, ff)
        self.w2 = nn.Linear(d, ff)
        self.w3 = nn.Linear(ff, d)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, x, pad):
        h = self.n1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        x = x + self.drop(h)
        h = self.n2(x)
        return x + self.drop(self.w3(self.w1(h) * F.silu(self.w2(h))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        dim0 = N_DYN * MULT + MULT
        self.seq_stem = SequenceStem(N_DYN, MULT)
        self.static_stem = nn.Sequential(
            nn.Conv1d(N_STATIC, MULT, 1, bias=False), nn.BatchNorm1d(MULT), nn.SiLU())
        self.merge = nn.Sequential(
            nn.Conv1d(dim0, DIM1, 1, bias=False), nn.BatchNorm1d(DIM1),
            ResBlock(DIM1, 1), ResBlock(DIM1, 1), nn.BatchNorm1d(DIM1))
        self.layers = nn.ModuleList(
            TfmLayer(DIM1, N_HEADS, FF_DIM, TFM_DROPOUT) for _ in range(N_LAYERS))
        self.pred_seq = nn.Conv1d(DIM1, T_OUT * DIM2, 1)
        self.head_body = nn.Sequential(ResBlock(DIM2), ResBlock(DIM2))
        self.head_mu = nn.Conv1d(DIM2, 2, 1)
        self.head_var = nn.Conv1d(DIM2, 2, 1)
        self.head_aux_mu = nn.Conv1d(DIM2, 4, 1)
        self.head_aux_var = nn.Conv1d(DIM2, 4, 1)
        # normalization constants for cumsum decoding (filled from stats in
        # main; persisted in state_dict so inference reloads them)
        self.register_buffer("d_mean", torch.zeros(2))
        self.register_buffer("d_std", torch.ones(2))
        self.register_buffer("t_mean", torch.zeros(2))
        self.register_buffer("t_std", torch.ones(2))

    def forward(self, b):
        dyn, static = b["dyn"], b["static"]
        B, P, T, C = dyn.shape
        h = self.seq_stem(dyn.reshape(B * P, T, C).transpose(1, 2))[:, :, -1:]
        hs = self.static_stem(static.reshape(B * P, -1, 1))
        h = self.merge(torch.cat([h, hs], 1)).reshape(B, P, DIM1)
        pad = ~b["valid"]
        for lay in self.layers:
            h = lay(h, pad)
        z = self.pred_seq(h.reshape(B * P, DIM1, 1)).reshape(B * P, DIM2, T_OUT)
        z = self.head_body(z)
        rs = lambda t: t.transpose(1, 2).reshape(B, P, T_OUT, -1)
        if DECODE == "direct":
            # true 1st-place head: per-frame displacement from the last input
            # frame, already in tgt-normalized space (no integration)
            mean = rs(self.head_mu(z))
        else:
            # cumsum decoding (2nd place): head emits per-frame deltas;
            # integrate to displacement, then express in tgt-normalized space
            delta = rs(self.head_mu(z)) * self.d_std + self.d_mean
            mean = (torch.cumsum(delta, dim=2) - self.t_mean) / self.t_std
        return (mean, rs(self.head_var(z)),
                rs(self.head_aux_mu(z)), rs(self.head_aux_var(z)))


def gnll(mu, tgt, var):
    return 0.5 * (torch.log(var) + (tgt - mu) ** 2 / var)


def loss_fn(out, b):
    mu, var, amu, avar = out
    var = F.softplus(var) + MIN_VAR
    avar = F.softplus(avar) + MIN_VAR
    m = (b["out_mask"] & b["valid"][:, :, None]).float()[..., None]
    traj = (gnll(mu, b["tgt"], var) * m).sum() / (m.sum() * 2).clamp(1)
    aux = gnll(amu, b["sa"], avar) * (2.0 / 4)
    aux = (aux * m).sum() / (m.sum() * 2).clamp(1)
    return (1 - AUX_W) * traj + AUX_W * aux


def make_optimizer(model):
    return torch.optim.RAdam(model.parameters(), lr=LR,
                             weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True)


def lr_at(train_sec, step):
    return LR  # 1st place: constant LR, no warmup, no schedule
# =========================== <<END AGENT EDITABLE>> =========================


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


@torch.no_grad()
def predict_val(d, model, stats):
    model.eval()
    vi = d["val_idx"]
    preds = np.zeros((len(vi), P_MAX, T_OUT, 2), np.float32)
    tstd = torch.as_tensor(stats["tgt_std"], device=DEVICE)
    tmean = torch.as_tensor(stats["tgt_mean"], device=DEVICE)
    for s0 in range(0, len(vi), 256):
        idx = vi[s0:s0 + 256]
        b = {k: v.to(DEVICE) for k, v in make_batch(d, idx, stats, train=False).items()}
        mu = model(b)[0] * tstd + tmean
        p1 = (mu + b["last_xy"][:, :, None, :]).cpu().numpy()
        # h-flip TTA: mirror input, predict, mirror back, average
        b2 = {k: v.to(DEVICE) for k, v in
              make_batch(d, idx, stats, train=False, flip_all=True).items()}
        mu2 = model(b2)[0] * tstd + tmean
        p2 = (mu2 + b2["last_xy"][:, :, None, :]).cpu().numpy()
        p2[..., 1] = prepare.FIELD_Y - p2[..., 1]
        preds[s0:s0 + len(idx)] = 0.5 * (p1 + p2)
    model.train()
    return preds


def main():
    t_start = time.time()
    d = prepare.load()
    if FOLD >= 0:
        ti = d["train_idx"]
        d["train_idx"] = ti[np.arange(len(ti)) % 8 != FOLD]
        print(f"fold={FOLD}: train {len(ti)} -> {len(d['train_idx'])}", flush=True)
    stats = compute_stats(d)
    rng = np.random.default_rng(SEED)
    model = Model().to(DEVICE)
    with torch.no_grad():
        model.d_mean.copy_(torch.as_tensor(stats["sa_mean"][0:2]))
        model.d_std.copy_(torch.as_tensor(stats["sa_std"][0:2]))
        model.t_mean.copy_(torch.as_tensor(stats["tgt_mean"]))
        model.t_std.copy_(torch.as_tensor(stats["tgt_std"]))
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model)
    ema = None
    swa, swa_n = None, 0
    has_val = len(d["val_idx"]) > 0
    budget_str = f"{STEP_BUDGET} steps" if STEP_BUDGET else f"{TIME_BUDGET}s"
    print(f"device={DEVICE} params={n_params/1e6:.2f}M "
          f"train={len(d['train_idx'])} val={len(d['val_idx'])} budget={budget_str}", flush=True)

    train_sec, step, last_eval = 0.0, 0, 0.0
    order = rng.permutation(d["train_idx"])
    pos = 0
    progress = lambda: (step / STEP_BUDGET) if STEP_BUDGET else (train_sec / TIME_BUDGET)
    while progress() < 1.0:
        if pos + BATCH > len(order):
            order = rng.permutation(d["train_idx"])
            pos = 0
        idx = order[pos:pos + BATCH]; pos += BATCH
        t0 = time.time()
        b = {k: v.to(DEVICE) for k, v in make_batch(d, idx, stats, rng=rng, train=True).items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(train_sec, step)
        loss = loss_fn(model(b), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if progress() > 0.75 and step % 25 == 0:
            sd = model.state_dict()
            if swa is None:
                swa = {k: v.detach().clone().float() for k, v in sd.items()}
                swa_n = 1
            else:
                for k, v in sd.items():
                    if v.dtype.is_floating_point:
                        swa[k] += (v.detach().float() - swa[k]) / (swa_n + 1)
                    else:
                        swa[k] = v.detach().clone()
                swa_n += 1
        if ema is None and progress() > EMA_AFTER_FRAC:
            # scale decay to run length: half-life ~20% of remaining steps
            est_total = STEP_BUDGET or step / max(train_sec, 1e-6) * TIME_BUDGET
            decay = min(EMA_DECAY, 0.5 ** (1.0 / max(0.2 * est_total, 50)))
            print(f"ema starts: step={step} decay={decay:.5f}", flush=True)
            ema = EMA(model, decay)
        if ema is not None:
            ema.update(model)
        if DEVICE == "mps":
            torch.mps.synchronize()
        train_sec += time.time() - t0
        step += 1

        if step % 100 == 0:
            print(f"[t={train_sec:6.0f}s] step {step:5d} loss {loss.item():.4f}", flush=True)
        if (has_val and ema is not None and progress() < 0.97
                and train_sec - last_eval >= EVAL_EVERY_SEC):
            last_eval = train_sec
            eval_model = Model().to(DEVICE)
            ema.copy_to(eval_model)
            rmse = prepare.evaluate(d, predict_val(d, eval_model, stats))
            del eval_model
            print(f"[t={train_sec:6.0f}s] step {step:5d} interim val_rmse {rmse:.4f}", flush=True)

    # candidate weights: raw final, EMA, tail-average (SWA); keep the best on
    # val (same spirit as 1st place's best-epoch selection)
    cands = {"raw": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    if ema is not None:
        cands["ema"] = ema.shadow
    if swa is not None and swa_n >= 3:
        cands["swa"] = swa
    if has_val:
        scores = {}
        for name, sd in cands.items():
            model.load_state_dict(sd, strict=True)
            scores[name] = prepare.evaluate(d, predict_val(d, model, stats))
        best = min(scores, key=scores.get)
        if best != list(cands)[-1]:
            model.load_state_dict(cands[best], strict=True)
            prepare.evaluate(d, predict_val(d, model, stats))  # leave best preds on disk
        print("  ".join(f"{k}_rmse: {v:.6f}" for k, v in scores.items()) + f"  best: {best}", flush=True)
        val_rmse = scores[best]
    else:
        # no validation split: take EMA (1st-place default for final weights)
        best = "ema" if "ema" in cands else "raw"
        model.load_state_dict(cands[best], strict=True)
        print(f"no val split; keeping {best} weights", flush=True)
        val_rmse = -1.0
    os.makedirs("models", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "stats": {k: np.asarray(v) for k, v in stats.items()},
                "config": {k: globals()[k] for k in
                           ["T_USED", "MULT", "DIM1", "DIM2", "N_LAYERS", "N_HEADS",
                            "FF_DIM", "N_DYN", "N_STATIC", "SEED", "AUX_W", "LR",
                            "FOLD", "DECODE"]},
                "val_rmse": val_rmse}, "models/last.pt")

    # ------------- fixed results block (do not modify) -------------
    print(f"val_rmse: {val_rmse:.6f}")
    print(f"training_seconds: {train_sec:.1f}")
    print(f"total_seconds: {time.time() - t_start:.1f}")
    print(f"num_steps: {step}")
    print(f"num_params_M: {n_params/1e6:.3f}")
    print(f"device: {DEVICE}")


if __name__ == "__main__":
    main()
