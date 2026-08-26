"""Fetch a subset of Duke-Breast-Cancer-MRI from TCIA. No account, no token.

    python fetch_duke.py --out /somewhere/duke_dce --patients 150

Each patient contributes four series: the pre-contrast dynamic acquisition and
the first three post-contrast passes. That is about 100 MB per patient, so 150
patients is roughly 36 GB. The full collection is 387 GB and you do not need it.

The lesion boxes and the clinical tables are separate small files on the
collection's page; --meta downloads those too, into ./meta.
"""
import argparse, collections, json, os, subprocess, sys, urllib.request

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
WANT = ["ax dyn pre", "ax dyn 1st pass", "ax dyn 2nd pass", "ax dyn 3rd pass"]
META = ["Annotation_Boxes.xlsx", "Clinical_and_Other_Features.xlsx",
        "Imaging_Features.xlsx", "train_ids.csv", "test_ids.csv"]


def get(url, timeout=180):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients", type=int, default=150)
    ap.add_argument("--meta", action="store_true", help="also fetch boxes and clinical tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.meta:
        os.makedirs("meta", exist_ok=True)
        for f in META:
            dst = os.path.join("meta", f)
            if os.path.exists(dst):
                continue
            open(dst, "wb").write(get("https://www.cancerimagingarchive.net/wp-content/uploads/" + f))
            print("  meta/%s" % f)

    print("  listing series ...", flush=True)
    series = json.loads(get(API + "/getSeries?Collection=Duke-Breast-Cancer-MRI&format=json", 300))
    by = collections.defaultdict(dict)
    for s in series:
        by[s["PatientID"]][(s.get("SeriesDescription") or "").lower().strip()] = s["SeriesInstanceUID"]
    ok = sorted(p for p, v in by.items() if all(w in v for w in WANT))
    print("  %d of %d patients have all four series; taking %d"
          % (len(ok), len(by), min(a.patients, len(ok))), flush=True)

    for p in ok[:a.patients]:
        for w in WANT:
            dst = os.path.join(a.out, p, w.replace(" ", "_"))
            if os.path.isdir(dst) and any(f.endswith(".dcm") for f in os.listdir(dst)):
                continue
            os.makedirs(dst, exist_ok=True)
            zp = os.path.join(dst, "s.zip")
            for attempt in range(3):
                try:
                    open(zp, "wb").write(get(API + "/getImage?SeriesInstanceUID=" + by[p][w], 900))
                    subprocess.run(["unzip", "-qo", zp, "-d", dst], check=True)
                    os.remove(zp)
                    break
                except Exception as e:
                    if attempt == 2:
                        print("  FAILED %s %s: %s" % (p, w, e), file=sys.stderr)
        print("  %s" % p, flush=True)


if __name__ == "__main__":
    main()
