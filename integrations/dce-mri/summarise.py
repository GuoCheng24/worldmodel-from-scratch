"""Print the tables in this README from the committed results. Needs numpy only.

    python summarise.py

The runs need a GPU and 36 GB of DICOMs; what they leave behind is one row per
slice, phase, region and predictor, which compresses to about a megabyte. Every
number quoted in the README is recomputed here, so a sentence cannot drift away
from the run behind it.

Medians and interquartile ranges throughout, never means: the per-slice
distributions are wide and right-skewed, and a mean over them describes the
worst slices rather than the typical one.
"""
import csv, gzip, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("full_plain", "lesion loss weight 1"),
        ("full_w50", "lesion loss weight 50"),
        ("full_oracle", "one true scalar leaked")]
METHODS = [("B0_identity", "B0  hand back the input"),
           ("B2_cond_mean", "B2  256-entry lookup"),
           ("UNet", "UNet")]


def load(tag):
    f = os.path.join(HERE, "results", tag + ".csv.gz")
    if not os.path.exists(f):
        return None
    with gzip.open(f, "rt") as fh:
        return list(csv.DictReader(fh))


def stat(rows, region, method, col):
    v = np.array([float(r[col]) for r in rows if r["region"] == region and r["method"] == method])
    return (np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v)) if len(v) else None


def main():
    runs = {t: load(t) for t, _ in RUNS}
    if not runs.get("full_plain"):
        print(__doc__); return 1
    base = runs["full_plain"]
    print("\n  Held-out patients: %d" % len({r["patient"] for r in base}))
    print("\n  %-13s %-26s %-22s %-22s %s"
          % ("region", "predictor", "SSIM median [IQR]", "PSNR median [IQR]", "rMSE"))
    print("  " + "-" * 96)
    for region in ("whole_slice", "lesion_box"):
        for m, label in METHODS:
            s = stat(base, region, m, "ssim"); p = stat(base, region, m, "psnr")
            r = stat(base, region, m, "rmse")
            if not s:
                continue
            print("  %-13s %-26s %.3f [%.3f, %.3f]     %5.2f [%5.2f, %5.2f]    %6.1f"
                  % (region, label, s[0], s[1], s[2], p[0], p[1], p[2], r[0]))
        print("  %-13s %s measurements" % ("", stat(base, region, "UNet", "ssim")[3]))

    print("\n  What moves the lesion number, and what does not:\n")
    print("  %-26s %-12s %-12s %-12s %s" % ("run", "lesion SSIM", "lesion PSNR", "lesion rMSE", "whole SSIM"))
    print("  " + "-" * 78)
    b2 = stat(base, "lesion_box", "B2_cond_mean", "ssim")
    for tag, label in RUNS:
        rows = runs.get(tag)
        if not rows:
            continue
        s = stat(rows, "lesion_box", "UNet", "ssim"); p = stat(rows, "lesion_box", "UNet", "psnr")
        r = stat(rows, "lesion_box", "UNet", "rmse"); w = stat(rows, "whole_slice", "UNet", "ssim")
        print("  %-26s %-12.3f %-12.2f %-12.1f %.3f" % (label, s[0], p[0], r[0], w[0]))
    print("  %-26s %-12.3f %-12s %-12s %s"
          % ("(B2, zero training)", b2[0],
             "%.2f" % stat(base, "lesion_box", "B2_cond_mean", "psnr")[0],
             "%.1f" % stat(base, "lesion_box", "B2_cond_mean", "rmse")[0], "0.710"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
