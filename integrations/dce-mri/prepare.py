"""Cache Duke DCE slices as arrays, with the lesion box carried along.

Reading 160 DICOMs per series per epoch is the wrong way to train anything.
This pulls the slices once, at the resolution the papers use, and keeps the
lesion box in the same coordinates so evaluation can be restricted to it later.

Slice indexing follows InstanceNumber, because Duke's annotation table does:
sorting by z position puts the box on nothing for a third of patients.
"""
import os, sys, glob
import numpy as np, pydicom, pandas as pd
from skimage.transform import resize

import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--data", required=True, help="where fetch_duke.py put the DICOMs")
_ap.add_argument("--cache", required=True, help="where to write the .npz cache")
_ap.add_argument("--boxes", default="meta/Annotation_Boxes.xlsx")
_a = _ap.parse_args()
ROOT, OUT, BOXES = _a.data, _a.cache, _a.boxes
PRE, PHASES = "ax_dyn_pre", ["ax_dyn_1st_pass", "ax_dyn_2nd_pass", "ax_dyn_3rd_pass"]
SIZE = 256
os.makedirs(OUT, exist_ok=True)
boxes = pd.read_excel(BOXES).set_index("Patient ID")


def index_series(d):
    m = {}
    for f in glob.glob(os.path.join(d, "*.dcm")):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            m[int(ds.InstanceNumber)] = f
        except Exception:
            continue
    return m


def read(path):
    ds = pydicom.dcmread(path)
    return (ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1.0))
            + float(getattr(ds, "RescaleIntercept", 0.0)))


done = skipped = 0
for p in sorted(os.listdir(ROOT)):
    if not p.startswith("Breast_MRI_") or p not in boxes.index:
        continue
    dst = os.path.join(OUT, p + ".npz")
    if os.path.exists(dst):
        done += 1; continue
    idx = {ph: index_series(os.path.join(ROOT, p, ph)) for ph in [PRE] + PHASES}
    if any(len(v) < 20 for v in idx.values()):
        skipped += 1; continue
    common = sorted(set.intersection(*[set(v) for v in idx.values()]))
    if len(common) < 40:
        skipped += 1; continue
    b = boxes.loc[p]
    r0, r1 = int(b["Start Row"]), int(b["End Row"])
    c0, c1 = int(b["Start Column"]), int(b["End Column"])
    s0, s1 = int(b["Start Slice"]), int(b["End Slice"])
    # every lesion slice, plus an even spread of the rest, so the model does not
    # only ever see lesions and the evaluation is not only ever on them
    les = [k for k in common if s0 <= k <= s1]
    rest = [k for k in common if not (s0 <= k <= s1)]
    lo, hi = int(len(rest) * .25), int(len(rest) * .75)
    rest = [rest[i] for i in np.linspace(lo, max(hi - 1, lo), 16).astype(int)] if rest else []
    keep = sorted(set(les + rest))
    vols, ok = [], True
    native = None
    for k in keep:
        try:
            fr = [read(idx[ph][k]) for ph in [PRE] + PHASES]
        except Exception:
            ok = False; break
        native = fr[0].shape
        vols.append(np.stack([resize(f, (SIZE, SIZE), order=1, preserve_range=True,
                                     anti_aliasing=True) for f in fr]).astype(np.float32))
    if not ok or not vols:
        skipped += 1; continue
    sy, sx = SIZE / native[0], SIZE / native[1]
    np.savez_compressed(
        dst, vol=np.stack(vols).astype(np.float32), slices=np.array(keep),
        is_lesion=np.array([1 if s0 <= k <= s1 else 0 for k in keep]),
        box=np.array([r0 * sy, r1 * sy, c0 * sx, c1 * sx]), native=np.array(native))
    done += 1
    print("  %s: %d slices (%d lesion)" % (p, len(keep), sum(1 for k in keep if s0 <= k <= s1)), flush=True)

print("\n  cached %d patients, skipped %d -> %s" % (done, skipped, OUT))
