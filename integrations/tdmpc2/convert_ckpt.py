"""Remap TD-MPC2's released single-task checkpoints onto its current layout.

This is NOT a working fix, and it is here as evidence, not as a tool. The remap
is exact - every parameter name and shape matches the model TD-MPC2's current
code builds - and the result still does not behave like the released model:
converted `cartpole-swingup-1.pt` scores 233.7 where the repository's own
`results/tdmpc2/cartpole-swingup.csv` records 866.2 for that seed, and its
reward head correlates 0.68 with the true reward. Since the parameter layout is
recovered exactly, the remaining difference has to live in the parameter-free
operations, which no published version of the code contains. See README.md.

Fourteen of the 312 released single-task checkpoints are affected; the other
298 load with the current code unchanged. What those fourteen store is a FLAT
nn.Sequential per MLP:

    [Linear, LayerNorm, Mish, Linear, LayerNorm, Mish, Linear]

The current code builds the same computation out of NormedLinear modules, which
fold the LayerNorm into the Linear:

    [NormedLinear, NormedLinear, NormedLinear]

TD-MPC2 ships api_model_conversion() for exactly this kind of migration, but it
only renames the Q-ensemble parameters, so loading a released single-task
checkpoint with the current code raises a RuntimeError listing 28 missing keys.

The pairing here is not hard-coded: a 2-D weight is a Linear, a 1-D weight is a
LayerNorm, and a Linear immediately followed by a LayerNorm becomes one
NormedLinear. A trailing Linear with no LayerNorm after it (the reward and
policy output heads) stays a plain Linear. Every produced key is then checked
against the model the current code actually builds - name and shape - and every
source tensor must be consumed. Nothing is dropped silently. The Q-ensemble
keys are left in their original form so TD-MPC2's own api_model_conversion()
still handles them at load time.

    TDMPC2_DIR=/path/to/tdmpc2/tdmpc2 python convert_ckpt.py \
        task=cartpole-swingup checkpoint=in.pt +out=out.pt
"""
import sys, os, argparse
TD = os.environ.get("TDMPC2_DIR")
if TD is None or not os.path.isdir(TD):
    raise SystemExit("set TDMPC2_DIR to the tdmpc2/tdmpc2 directory of a clone of\n"
                     "https://github.com/nicklashansen/tdmpc2 (see README.md here)")
sys.path.insert(0, TD); os.chdir(TD)
import torch, hydra
from common.parser import parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2

BLOCKS = ("_encoder.state", "_dynamics", "_reward", "_pi")


def _entries(sd, prefix):
    """Ordered [(sortkey, weight, bias)] for one flat block."""
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix + "."):
            continue
        rest = k[len(prefix) + 1:].split(".")
        idx, leaf = tuple(int(p) for p in rest[:-1]), rest[-1]
        out.setdefault(idx, {})[leaf] = v
    return [(i, out[i]["weight"], out[i]["bias"]) for i in sorted(out)]


def convert_block(sd, prefix):
    """Fold [Linear, LayerNorm] pairs into NormedLinear-style keys."""
    ent, new, i, j = _entries(sd, prefix), {}, 0, 0
    while i < len(ent):
        _, w, b = ent[i]
        assert w.ndim == 2, "%s entry %d is not a Linear (shape %s)" % (prefix, i, tuple(w.shape))
        new["%s.%d.weight" % (prefix, j)], new["%s.%d.bias" % (prefix, j)] = w, b
        i += 1
        if i < len(ent) and ent[i][1].ndim == 1:          # a LayerNorm follows: fold it in
            _, lw, lb = ent[i]
            new["%s.%d.ln.weight" % (prefix, j)], new["%s.%d.ln.bias" % (prefix, j)] = lw, lb
            i += 1
        j += 1
    return new


@hydra.main(config_name="config", config_path=os.environ["TDMPC2_DIR"], version_base=None)
def main(cfg):
    src_path, dst_path = cfg.checkpoint, cfg.get("out")
    cfg = parse_cfg(cfg)
    make_env(cfg)
    target = TDMPC2(cfg).model.state_dict()

    blob = torch.load(src_path, map_location="cpu", weights_only=False)
    src = blob["model"] if "model" in blob else blob
    if "_encoder.state.0.ln.weight" in src:
        print("  already in the current layout; nothing to convert."); return

    converted, consumed = {}, set()
    for pre in BLOCKS:
        keys = [k for k in src if k.startswith(pre + ".")]
        if not keys:
            continue
        converted.update(convert_block(src, pre)); consumed.update(keys)

    # --- verification: names, shapes, and that nothing was dropped -----------
    for k, v in converted.items():
        assert k in target, "produced key %s does not exist in the current model" % k
        assert tuple(v.shape) == tuple(target[k].shape), \
            "%s: converted %s vs model %s" % (k, tuple(v.shape), tuple(target[k].shape))
    for k in target:
        if k.startswith(BLOCKS) and k not in converted:
            raise AssertionError("the current model wants %s and the conversion did not produce it" % k)
    leftover = [k for k in src if k not in consumed and not k.startswith(("_Qs", "_target_Qs"))]
    assert not leftover, "source tensors nobody consumed: %s" % leftover

    out = {k: v for k, v in src.items() if k.startswith(("_Qs", "_target_Qs"))}
    out.update(converted)
    torch.save({"model": out, "metadata": blob.get("metadata", {})}, dst_path)
    print("  converted %d tensors across %d blocks" % (len(converted), len(BLOCKS)))
    print("  every name and shape matches the model the current code builds")
    print("  Q-ensemble keys left untouched for TD-MPC2's own api_model_conversion()")
    print("  wrote %s" % dst_path)


main()
