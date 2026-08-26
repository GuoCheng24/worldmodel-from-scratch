# What a zero-parameter predictor scores, and where it stops mattering

A world model does not have to be a robot simulator. A model that takes a
pre-contrast MRI slice and produces the post-contrast one is predicting how a
system evolves after an intervention, and the papers that do it increasingly
say so - the framing has moved from image translation to contrast *kinetics*.

The measurement in this repository transfers directly, and so does its first
question: **what does the most trivial predictor available already score?**

Everything below runs on Duke-Breast-Cancer-MRI, which is public on TCIA and
needs no account. Nothing here runs anyone else's model; every number is
produced by the scripts in this directory. Who to cite for that data, and what
its licence lets you do with the tables in `results/`, is at the bottom.

## Three predictors that should be in every table

| | what it is | parameters |
|---|---|---|
| **B0** | hand back the pre-contrast slice unchanged | 0 |
| **B1** | a global affine `a·pre + b`, one pair per dynamic phase, fitted on the training patients | 2 per phase |
| **B2** | `E[post \| pre intensity]`, a 256-entry lookup table fitted on the training patients. Sees one voxel and no context. | 256 per phase |

And, for scale, a plain 2M-parameter U-Net that sees the whole slice and the
phase index, trained to predict `post - pre`.

![Bars of SSIM for three predictors scored over the whole slice and inside the lesion box, and three lesion-box variants that barely differ](dce-baselines.png)

## What comes out

Held-out patients (60 of 150), native resolution, one pipeline. Medians over
7254 slice-phase pairs for the whole slice and 4305 for the lesion box, which is
smaller because only the annotated slices have one:

| region | predictor | SSIM | PSNR | rMSE |
|---|---|---|---|---|
| whole slice | B0 | 0.732 | 24.18 | 32.5 |
| whole slice | B2 | 0.710 | 25.32 | 28.0 |
| whole slice | **U-Net** | **0.804** | **28.29** | **19.4** |
| lesion box | B0 | 0.387 | 13.37 | 84.2 |
| lesion box | **B2** | **0.503** | 15.48 | **63.7** |
| lesion box | U-Net | 0.501 | **15.80** | 64.2 |

**Globally the U-Net wins by a wide margin - +0.07 SSIM, +4.1 dB, rMSE cut by
40%. Inside the lesion it draws level with a 256-entry lookup table** that has
no spatial context at all, and neither of them is far from handing back the
input.

A breast MRI slice is mostly fat, muscle, air and chest wall, none of which
enhance. The reason to inject gadolinium is the lesion, which is a few hundred
pixels out of 65,536. A global metric is therefore dominated by the part of the
image where doing nothing is the right answer, and it cannot separate a model
that predicts enhancement from one that preserves anatomy.

## What moves that number, and what does not

| what was changed | lesion SSIM | lesion PSNR | lesion rMSE | whole-slice SSIM |
|---|---|---|---|---|
| nothing (plain U-Net) | 0.501 | 15.80 | 64.2 | 0.804 |
| lesion loss weight x50 | 0.483 | 15.59 | 64.6 | 0.803 |
| one true scalar leaked in | **0.527** | **16.06** | **63.3** | 0.805 |
| *B2, which trains nothing* | *0.503* | *15.48* | *63.7* | *0.710* |

**Not enough data.** Going from 30 to 150 patients moved the U-Net's lesion
SSIM from 0.37 to 0.50 - from losing to the lookup table to matching it, and no
further. (The 30-patient run used a different, smaller patient set, so it is not
in the table.)

**The loss ignores the lesion.** Weighting the loss inside the box by fifty
changes nothing that matters, and the whole-slice numbers do not move either.

Under an L1 objective the optimal prediction is the conditional median given the
input. If the pre-contrast image says little about how a particular lesion will
enhance, the best any such model can do inside the lesion is the population
answer - which is what B2 already is, in 256 numbers.

**A control that did not settle it.** To test that reading, the true mean
enhancement inside the box was leaked to the model as one extra input channel.
If what is missing were a scalar's worth of magnitude, that should close the
gap. It bought **+0.026 SSIM**, and the final training loss was identical to the
plain run to four decimals - which is also what you would see if the model
largely ignored a constant broadcast channel. **Those two readings cannot be
separated here**, so this run is reported and not interpreted. Settling it needs
a graded oracle - one scalar, then a few components, then the full enhancement
map - which is not run here.

## A temporal metric with no optimum at the truth

Papers that frame this as kinetics need a temporal measure, and one that
appears is the mean SSIM between **adjacent frames of the generated sequence**:

    cSSIM = (1/(N-1)) Σ SSIM(I_t, I_{t+1})

No ground truth enters it. Run it on Duke's real DCE phases and on a sequence
that repeats one frame:

| sequence scored | cSSIM, median [IQR] |
|---|---|
| a model that outputs one frame forever | **1.0000** |
| the real DCE sequence | **0.8293** [0.793, 0.864] |

Higher is reported as better, so a model that reproduced the kinetics exactly
would rank below one that predicted no change at all. `adjacent_frame_metric.py`
is nine lines of measurement; the conclusion does not depend on any model.

## Reproducing it

```bash
pip install pydicom scikit-image pandas openpyxl torch
python fetch_duke.py --out /data/duke_dce --patients 150 --meta   # ~36 GB
python prepare.py --data /data/duke_dce --cache /data/duke_cache
python adjacent_frame_metric.py /data/duke_dce

DCE_CACHE=/data/duke_cache TAG=plain  LESION_W=1  python baselines.py
DCE_CACHE=/data/duke_cache TAG=w50    LESION_W=50 python baselines.py
```

**One thing will silently ruin this if you skip it.** Duke's annotation table
indexes slices by `InstanceNumber`, not by z position. Sort by z and the box
lands on nothing for about a third of patients. `prepare.py` checks it the way
it should be checked - inside the box the post-contrast signal must rise more
than outside - and the two orderings give median in/out enhancement ratios of
**4.45** and **1.38**.

## Caveats

- One collection, one modality, one anatomy. 150 patients, 60 held out.
- The annotation is a bounding box, not a segmentation, so it contains normal
  tissue and the lesion numbers are optimistic.
- Dynamic phases are not registered to each other here, so patient motion is
  inside every number, for every predictor equally.
- One U-Net at one training budget. This shows that a global metric does not
  reflect lesion-region performance. It does **not** show that no model can do
  better in the lesion, and a larger or better-designed model may.
- No published model was run. Nothing here is a measurement of anyone's method.

## What is already known

That inferring enhancement from a pre-contrast slice is underdetermined is not
a new observation, and it is worth reading the people who made it carefully.
[MIRAGE](https://arxiv.org/abs/2607.19137) states it in its first sentence -
"post-contrast appearance contains physiological information that is not
uniquely encoded in baseline anatomy" - and builds lesion-aware supervision on
top of that premise, evaluated on 301 multi-centre cases with eight metric
families including region-based ones. [Osuala et
al.](https://doi.org/10.1117/1.JMI.12.S2.S22014) evaluate synthesised DCE by
what a downstream detector can do with it, which is a better question than
pixel fidelity.

What none of them reports is the floor: what a predictor with no parameters
already achieves, globally and where it matters. That is the only thing this
directory adds.

## The data, and what you may do with it

Duke-Breast-Cancer-MRI, on TCIA, no account needed:

> Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E.,
> Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2021). *Dynamic contrast-enhanced
> magnetic resonance images of breast cancer patients with tumor locations*
> [Data set]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/TCIA.e3sv-re93>

TCIA asks that the archive be cited alongside the collection: Clark, K., Vendt, B.,
Smith, K., et al. (2013). The Cancer Imaging Archive (TCIA): maintaining and
operating a public information repository. *Journal of Digital Imaging* 26(6),
1045-1057. <https://doi.org/10.1007/s10278-013-9622-7>

The collection is released under **CC BY-NC 4.0**. The code in this repository is
MIT, but `results/*.csv.gz` are measurements computed from that collection, so they
carry its terms with them: attribute it, and keep them out of commercial use. No
images are redistributed here - each row is one SSIM, PSNR and RMSE for one patient,
phase, region and method, and the identifiers in them (`Breast_MRI_001`) are TCIA's
own, de-identified at the source.
