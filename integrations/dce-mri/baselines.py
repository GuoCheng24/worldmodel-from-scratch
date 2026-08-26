"""What does spatial modelling actually buy on this task, and where?

Three predictors of the post-contrast slice, all evaluated in one pipeline on
held-out patients, globally and inside the lesion box:

  B0   predict no change            hand back the pre-contrast slice
  B2   conditional mean             E[post | pre intensity at this voxel], a
                                    lookup table fitted on the training
                                    patients. Sees one voxel, no context.
  UNet a plain U-Net                sees the whole slice and the phase index,
                                    trained to predict post - pre.

The U-Net predicts the CHANGE rather than the post-contrast image, which is
what B0 predicts as zero, so the three sit on one axis. Pre and post are put on
a COMMON intensity scale here - a single constant from the training patients -
rather than scaled by different constants, so that no part of the pre-to-post
mapping is handed to the model for free.
"""
import os, sys, glob, csv, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

CACHE = os.environ.get("DCE_CACHE") or sys.exit(
    "set DCE_CACHE to the directory prepare.py wrote")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NPH, NBINS, EPOCHS = 3, 256, int(os.environ.get("EPOCHS", 60))
LESION_W = float(os.environ.get("LESION_W", 1.0))   # loss weight inside the box
TAG = os.environ.get("TAG", "plain")
ORACLE = int(os.environ.get("ORACLE", 0))   # leak ONE true scalar per slice
torch.manual_seed(0); np.random.seed(0)

pats = sorted(f[:-4] for f in os.listdir(CACHE) if f.endswith(".npz"))
data = {p: np.load(os.path.join(CACHE, p + ".npz")) for p in pats}
rng = np.random.default_rng(0)
order = list(pats); rng.shuffle(order)
cut = int(len(order) * 0.6)
train_p, test_p = sorted(order[:cut]), sorted(order[cut:])
print("  %d patients: %d train, %d test" % (len(pats), len(train_p), len(test_p)))

SCALE = float(np.percentile(np.concatenate(
    [data[p]["vol"][:, 0].reshape(-1)[::11] for p in train_p]), 99))
print("  common intensity scale (99th pct of training pre-contrast): %.1f" % SCALE)


def pairs(ps):
    X, Y, PH, META, W, O = [], [], [], [], [], []
    for p in ps:
        v = data[p]["vol"] / SCALE
        box = data[p]["box"]; isl = data[p]["is_lesion"]
        r0, r1, c0, c1 = [int(round(b)) for b in box]
        for i in range(len(v)):
            for j in range(NPH):
                X.append(v[i, 0].astype(np.float16)); Y.append(v[i, j + 1].astype(np.float16))
                PH.append(j)
                META.append((p, i, j, isl[i], box))
                # the weight map is a box, so keep the box and build it in the
                # loop. Materialising it here was 3.3 GB of mostly ones and put
                # this process at the top of a shared node's memory table.
                W.append((r0, r1, c0, c1) if isl[i] else (-1, -1, -1, -1))
                if isl[i]:
                    d = v[i, j + 1][max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] \
                        - v[i, 0][max(r0, 0):r1 + 1, max(c0, 0):c1 + 1]
                else:
                    d = v[i, j + 1] - v[i, 0]
                O.append(np.float32(d.mean()) if d.size else np.float32(0))
    return (np.stack(X), np.stack(Y), np.array(PH), META, np.array(W, np.int32),
            np.array(O, np.float32))


Xtr, Ytr, Ptr, _, Wtr, Otr = pairs(train_p)
Xte, Yte, Pte, Mte, _, Ote = pairs(test_p)
print("  train pairs %d, test pairs %d" % (len(Xtr), len(Xte)))

# ---- B2: conditional mean of post given the pre intensity -------------------
edges = np.linspace(0, float(np.percentile(Xtr[::7].astype(np.float32), 99.9)), NBINS + 1)
LUT = np.zeros((NPH, NBINS), np.float32)
for j in range(NPH):
    m = Ptr == j
    # every 16th voxel: a 256-bin table over 17 million samples is already
    # far past the point where more data changes it.
    x = Xtr[m].reshape(-1)[::16].astype(np.float32)
    y = Ytr[m].reshape(-1)[::16].astype(np.float32)
    w = np.clip(np.digitize(x, edges) - 1, 0, NBINS - 1)
    s = np.bincount(w, weights=y, minlength=NBINS); c = np.bincount(w, minlength=NBINS)
    lut = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    occ = np.where(c > 0)[0]
    if len(occ):
        lut = lut[occ[np.clip(np.searchsorted(occ, np.arange(NBINS)), 0, len(occ) - 1)]]
    LUT[j] = lut


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    def __init__(s, ch=(32, 64, 128, 256)):
        super().__init__()
        s.inp = blk(1 + NPH + ORACLE, ch[0])
        s.down = nn.ModuleList([blk(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        s.up = nn.ModuleList([blk(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1))])
        s.out = nn.Conv2d(ch[0], 1, 1)

    def forward(s, x, ph):
        h = torch.cat([x, ph], 1)
        h = s.inp(h); skips = [h]
        for d in s.down:
            h = d(F.avg_pool2d(h, 2)); skips.append(h)
        h = skips.pop()
        for u in s.up:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = u(torch.cat([h, skips.pop()], 1))
        return s.out(h)


net = UNet().to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=2e-4)
# The whole training set is 3 GB in float16, which fits on the card. Leaving it
# on the host meant a random gather per step on a contended CPU, and the GPU sat
# at 17% waiting for it - five minutes an epoch instead of one.
Xt = torch.tensor(Xtr).unsqueeze(1).to(DEV)
Yt = torch.tensor(Ytr).unsqueeze(1).to(DEV)
Pt = F.one_hot(torch.tensor(Ptr), NPH).float()[:, :, None, None].to(DEV)
Wbox = torch.tensor(Wtr).to(DEV)
Ot = torch.tensor(Otr)[:, None, None, None].to(DEV)
n, bs = len(Xt), 16
print("  tag=%s  lesion loss weight=%.0f  oracle scalar=%d" % (TAG, LESION_W, ORACLE))
print("  U-Net params %.2fM, training %d epochs" % (sum(p.numel() for p in net.parameters()) / 1e6, EPOCHS))
t0 = time.time()
for ep in range(EPOCHS):
    idx = torch.randperm(n); tot = 0.0
    for k in range(0, n, bs):
        b = idx[k:k + bs]
        x, y = Xt[b].float(), Yt[b].float()
        ph = Pt[b].expand(-1, -1, x.shape[2], x.shape[3])
        if ORACLE:
            ph = torch.cat([ph, Ot[b].expand(-1, 1, x.shape[2], x.shape[3])], 1)
        w = torch.ones_like(x)
        if LESION_W != 1.0:
            for t_, (r0_, r1_, c0_, c1_) in enumerate(Wbox[b].tolist()):
                if r0_ >= 0:
                    w[t_, :, max(r0_, 0):r1_ + 1, max(c0_, 0):c1_ + 1] = LESION_W
        # A plain global L1 is what these pipelines use; the weighted variant
        # exists to test whether the lesion failure is caused by the loss
        # ignoring a few hundred pixels out of 65,536.
        loss = ((x + net(x, ph) - y).abs() * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(b)
    if ep % 10 == 0 or ep == EPOCHS - 1:
        print("    epoch %3d  L1 %.5f  (%.0f s)" % (ep, tot / n, time.time() - t0), flush=True)

# ---- evaluate ---------------------------------------------------------------
net.eval()
rows = []
with torch.no_grad():
    for k in range(len(Xte)):
        pre = Xte[k].astype(np.float32); post = Yte[k].astype(np.float32); j = Pte[k]
        p_, i_, _, isles, box = Mte[k]
        x = torch.tensor(pre)[None, None].to(DEV)
        ph = F.one_hot(torch.tensor([j]), NPH).float()[:, :, None, None].to(DEV).expand(-1, -1, 256, 256)
        if ORACLE:
            ph = torch.cat([ph, torch.full((1, 1, 256, 256), float(Ote[k]), device=DEV)], 1)
        unet = float_arr = (x + net(x, ph))[0, 0].cpu().numpy()
        preds = {"B0_identity": pre,
                 "B2_cond_mean": LUT[j][np.clip(np.digitize(pre, edges) - 1, 0, NBINS - 1)],
                 "UNet": unet}
        regions = [("whole_slice", (slice(None), slice(None)))]
        if isles:
            r0, r1, c0, c1 = [int(round(v)) for v in box]
            if r1 - r0 >= 8 and c1 - c0 >= 8:
                regions.append(("lesion_box", (slice(r0, r1 + 1), slice(c0, c1 + 1))))
        for reg, sl in regions:
            gt = post[sl]
            lo, hi = float(gt.min()), float(gt.max())
            if hi - lo < 1e-6 or min(gt.shape) < 7:
                continue
            g = (gt - lo) / (hi - lo) * 255.0
            for nm, pr in preds.items():
                q = np.clip((pr[sl] - lo) / (hi - lo), 0, 1) * 255.0
                rows.append((p_, int(j), reg, nm,
                             structural_similarity(g, q, data_range=255.0),
                             peak_signal_noise_ratio(g, q, data_range=255.0),
                             float(np.sqrt(np.mean((gt - pr[sl]) ** 2))) * SCALE))

os.makedirs("results", exist_ok=True)
out = "results/%s.csv" % TAG
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["patient", "phase", "region", "method", "ssim", "psnr", "rmse"])
    w.writerows(rows)

print("\n  held-out: %d patients, %d measurements" % (len(test_p), len(rows)))
print("  %-13s %-14s %-22s %-22s %s" % ("region", "method", "SSIM median [IQR]", "PSNR median [IQR]", "rMSE"))
print("  " + "-" * 92)
for reg in ("whole_slice", "lesion_box"):
    for nm in ("B0_identity", "B2_cond_mean", "UNet"):
        sel = [r for r in rows if r[2] == reg and r[3] == nm]
        if not sel:
            continue
        s = np.array([r[4] for r in sel]); q = np.array([r[5] for r in sel]); m = np.array([r[6] for r in sel])
        print("  %-13s %-14s %.3f [%.3f, %.3f]     %5.2f [%5.2f, %5.2f]    %6.1f"
              % (reg, nm, np.median(s), *np.percentile(s, [25, 75]),
                 np.median(q), *np.percentile(q, [25, 75]), np.median(m)))
print("\n  wrote %s" % out)
