"""Apply the paper's temporal metric to the ground truth itself.

cSSIM = mean SSIM between adjacent frames of a sequence. No ground truth enters
the formula, so it is a measure of how little a sequence changes. Three
sequences settle what it rewards:

  constant   the same frame at every time point - a model with no dynamics
  truth      the real DCE phases, which is what a perfect model would output
  paper      what the paper reports for its own method

A metric on which the ground truth does not score best is not measuring quality.
"""
import os, sys, glob
import numpy as np, pydicom
from skimage.transform import resize
from skimage.metrics import structural_similarity

if len(sys.argv) < 2:
    raise SystemExit(__doc__ + "\nusage: python adjacent_frame_metric.py <duke_dicom_dir>")
ROOT = sys.argv[1]
SEQ = ["ax_dyn_pre", "ax_dyn_1st_pass", "ax_dyn_2nd_pass", "ax_dyn_3rd_pass"]
SIZE, N_SLICE = 256, 12


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.dcm"))):
        try:
            ds = pydicom.dcmread(f)
            a = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0)) \
                + float(getattr(ds, "RescaleIntercept", 0.0))
            z = float(ds.ImagePositionPatient[2]) if "ImagePositionPatient" in ds else len(out)
            out.append((z, a))
        except Exception:
            continue
    if not out:
        return None
    out.sort(key=lambda x: x[0])
    return np.stack([a for _, a in out])


def cssim(frames, rng):
    """The paper's formula, verbatim: mean SSIM over adjacent pairs."""
    return float(np.mean([structural_similarity(frames[t], frames[t + 1], data_range=rng)
                          for t in range(len(frames) - 1)]))


truth, const = [], []
for p in sorted(os.listdir(ROOT)):
    if not p.startswith("Breast_MRI_"):
        continue
    vols = {}
    bad = False
    for ph in SEQ:
        v = load(os.path.join(ROOT, p, ph))
        if v is None:
            bad = True; break
        vols[ph] = v
    if bad:
        continue
    n = min(len(v) for v in vols.values())
    if n < N_SLICE + 20:
        continue
    for i in np.linspace(n * 0.30, n * 0.70, N_SLICE).astype(int):
        fr = [resize(vols[ph][i].astype(np.float32), (SIZE, SIZE), order=1,
                     preserve_range=True, anti_aliasing=True) for ph in SEQ]
        hi = max(f.max() for f in fr); lo = min(f.min() for f in fr)
        rng = float(hi - lo) if hi > lo else 1.0
        truth.append(cssim(fr, rng))
        const.append(cssim([fr[0]] * len(fr), rng))     # a model that predicts nothing
    print("  %s" % p, flush=True)

t, c = np.array(truth), np.array(const)
print("\n  %d slices from the real DCE sequences" % len(t))
print("  %-34s %8s %8s %8s" % ("sequence scored by cSSIM", "median", "25th", "75th"))
print("  %-34s %8.4f %8.4f %8.4f" % ("the ground truth itself",
                                     np.median(t), *np.percentile(t, [25, 75])))
print("  %-34s %8.4f %8.4f %8.4f" % ("a model that outputs one frame",
                                     np.median(c), *np.percentile(c, [25, 75])))
print("\n  Papers using this metric report higher as better. The constant")
print("  sequence scores %.4f and the ground truth %.4f, so a model that" % (np.median(c), np.median(t)))
print("  reproduced the real kinetics exactly would rank BELOW one that")
print("  predicted no change at all. The metric has no optimum at the truth.")
