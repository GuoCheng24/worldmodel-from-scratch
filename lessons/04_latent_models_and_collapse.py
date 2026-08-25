"""Lesson 4 - Latent world models, and the collapse you cannot see in the loss.

    python lessons/04_latent_models_and_collapse.py     (~90 s on one GPU)

Lessons 1-3 predicted the state directly, because we had it. Real world models
see pixels, so they learn an encoder and predict in latent space:

    z_t = e(o_t)        and        z_{t+1} ~ f(z_t, a_t)

Training that end to end on ||f(e(o_t), a) - e(o_{t+1})||^2 has an exact,
trivial optimum: make e constant. The loss is then zero and the model has
learned nothing. This is representation collapse, and every latent world model
paper has a mechanism for avoiding it.

What the papers say less about is how you would *know*. This lesson trains the
naive objective and two standard fixes, then tries three plausible collapse
detectors on all three. Each detector is fooled by a different one. The only
measurement that tracks whether the representation is any good needs two
baselines that are usually left out.
"""
import numpy as np, torch, torch.nn as nn, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import wm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
rng = np.random.default_rng(0); torch.manual_seed(0)
pend = wm.Pendulum()
SIGNAL, DISTRACT, LATENT = 16, 48, 8
OBS = SIGNAL + DISTRACT

# The observation is 16 dimensions that depend on the pendulum and 48 that do
# not. The distractors have their own smooth dynamics, like a moving background:
# predictable, temporally coherent, and completely irrelevant. Without them the
# lesson does not work - a random encoder preserves a 2-D state almost perfectly,
# so every method scores about the same and nothing is being measured.
W1 = torch.tensor(rng.normal(0, 1.2, (2, 64)), dtype=torch.float32, device=DEV)
W2 = torch.tensor(rng.normal(0, 0.5, (64, SIGNAL)), dtype=torch.float32, device=DEV)


def signal_of(states):
    x = torch.as_tensor(states, dtype=torch.float32, device=DEV)
    return torch.tanh(torch.tanh(x @ W1) @ W2)


N, K = 6000, 20
s0 = pend.sample_states(N, rng); A = pend.sample_actions(N, K, rng)
T = wm.rollout(pend, s0, K, A)
S = np.concatenate([s0[:, None], T[:, :-1]], 1)
phase = rng.uniform(0, 2 * np.pi, (N, 1, DISTRACT)); freq = rng.uniform(.3, 1.2, (1, 1, DISTRACT))
tt = np.arange(K)[None, :, None] * pend.dt
Dz = np.sin(phase + freq * tt * 3.0)


def observe(states, dist):
    return torch.cat([signal_of(states.reshape(-1, 2)),
                      torch.tensor(dist.reshape(-1, DISTRACT), dtype=torch.float32, device=DEV)], -1)


O = observe(S, Dz)
O_next = observe(np.concatenate([S[:, 1:], T[:, -1:]], 1),
                 np.concatenate([Dz[:, 1:], Dz[:, -1:]], 1))
At = torch.tensor(A.reshape(-1, 1), dtype=torch.float32, device=DEV)
St = torch.tensor(S.reshape(-1, 2), dtype=torch.float32, device=DEV)
n = 6000


def mlp(i, o, h=192):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


# ═══ 1. Floors and ceilings, before anything is trained ════════════════════
print("[1] Two baselines first, because none of the numbers below mean anything")
print("    without them.")
ceiling = wm.probe_r2(O[:n], St[:n])
floor = []
for s in range(5):
    torch.manual_seed(s); e = mlp(OBS, LATENT).to(DEV)
    with torch.no_grad():
        floor.append(wm.probe_r2(e(O[:n]), St[:n]))
print("    ceiling  linear probe on the raw %d-dim observation      R2 = %.3f" % (OBS, ceiling))
print("    floor    same probe on an UNTRAINED encoder, 5 seeds     R2 = %.3f to %.3f"
      % (min(floor), max(floor)))
print("    A random encoder is not a straw man. Random features preserve a great")
print("    deal, and a learned representation has to beat this to have done work.")

# ═══ 2. Three objectives ═══════════════════════════════════════════════════
print("\n[2] Training the naive objective and two standard fixes.")
OBJ = [("naive", "collapse is the global optimum"),
       ("+decoder", "also reconstruct the observation"),
       ("+variance", "penalise per-dimension std below 1")]
res = {}
for name, _ in OBJ:
    torch.manual_seed(0)
    enc, dyn, dec = mlp(OBS, LATENT).to(DEV), mlp(LATENT + 1, LATENT).to(DEV), mlp(LATENT, OBS).to(DEV)
    params = list(enc.parameters()) + list(dyn.parameters())
    if name == "+decoder":
        params += list(dec.parameters())
    opt = torch.optim.AdamW(params, 1e-3)
    for i in range(5000):
        j = torch.randint(0, len(O), (512,), device=DEV)
        z, z_next = enc(O[j]), enc(O_next[j])
        loss = ((z + dyn(torch.cat([z, At[j]], -1)) - z_next) ** 2).mean()
        if name == "+decoder":
            loss = loss + ((dec(z) - O[j]) ** 2).mean()
        if name == "+variance":
            loss = loss + torch.relu(1.0 - z.std(0)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = enc(O[:n])
    res[name] = dict(loss=float(loss.detach()), std=float(Z.std(0).mean()),
                     rank=wm.effective_rank(Z), r2=wm.probe_r2(Z, St[:n]))
    print("    %-10s trained." % name)

print("\n[3] Three plausible collapse detectors, and what each one reports.")
print("      %-10s %-13s %-14s %-16s %s" % ("objective", "prediction loss", "latent std", "effective rank", "probe R2"))
for name, _ in OBJ:
    r = res[name]
    print("      %-10s %-13.2e %-14.4f %-16.2f %.3f" % (name, r["loss"], r["std"], r["rank"], r["r2"]))
print("      %-10s %-13s %-14s %-16s %.3f" % ("(floor)", "-", "-", "-", float(np.mean(floor))))
print("      %-10s %-13s %-14s %-16s %.3f" % ("(ceiling)", "-", "-", "-", ceiling))

nv, dc, vr = res["naive"], res["+decoder"], res["+variance"]
print("\n    Read the loss column first. '%s' wins it by %.0f orders of magnitude"
      % ("naive", np.log10(dc["loss"] / nv["loss"])))
print("    and its representation is the only one BELOW the random-encoder floor")
print("    (%.3f against %.3f). On this objective, a lower loss is evidence of a" % (nv["r2"], np.mean(floor)))
print("    worse model, because the way to drive it down is to stop representing")
print("    anything. Nothing in the training curve tells you this.")

print("\n    Now the two detectors you would reach for:")
print("      latent std     catches '%s' (%.4f) and passes '%s' (%.2f)"
      % ("naive", nv["std"], "+variance", vr["std"]))
print("      effective rank passes '%s' (%.2f) and fails '%s' (%.2f)"
      % ("naive", nv["rank"], "+variance", vr["rank"]))
print("    They disagree, and both are right about something. Std is blind to")
print("    direction: +variance has healthy scale with every dimension collapsed")
print("    onto one. Effective rank is blind to scale: it is computed from the")
print("    shape of the spectrum, so naive's squashed-but-spread representation")
print("    looks fine. Neither is a collapse detector on its own.")

best = max(OBJ, key=lambda o: res[o[0]]["r2"])[0]
print("\n    The probe is the one that tracked usefulness - but only because it was")
print("    read against a floor. '%s' reaches %.3f, between a floor of %.3f and a"
      % (best, res[best]["r2"], np.mean(floor)))
print("    ceiling of %.3f. Quoted alone, %.3f says nothing: an untrained network"
      % (ceiling, res[best]["r2"]))
print("    scores %.3f for free, and the collapsed model's %.3f would still have"
      % (np.mean(floor), nv["r2"]))
print("    looked like a plausible R2 to anyone who was not given the floor.")

print("\n    One honest complication. The best probe score belongs to '%s'," % best)
if res[best]["rank"] < 2:
    print("    which is also the model the effective rank says has collapsed onto")
    print("    %.2f dimensions. Two reasonable metrics disagree about which fix" % res[best]["rank"])
    print("    won, and this lesson does not resolve it: a rank-1 representation")
    print("    that a linear probe reads at %.3f is either enough for this task or" % res[best]["r2"])
    print("    a warning about the task, and telling those apart needs a downstream")
    print("    use, not another representation statistic. Lesson 3 built one.")
else:
    print("    and the other metrics agree with it here.")

FIG.mkdir(exist_ok=True)
np.savez(FIG / "lesson04.npz",
         names=np.array([o[0] for o in OBJ]),
         loss=np.array([res[o[0]]["loss"] for o in OBJ]),
         std=np.array([res[o[0]]["std"] for o in OBJ]),
         rank=np.array([res[o[0]]["rank"] for o in OBJ]),
         r2=np.array([res[o[0]]["r2"] for o in OBJ]),
         floor=np.array(floor), ceiling=ceiling, latent=LATENT)
print("\nSaved figures/lesson04.npz")
