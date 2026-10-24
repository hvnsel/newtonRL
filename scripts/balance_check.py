"""Replay the mass-balance terms in scalar Python, same arithmetic as the env."""
SOIL = 1500.0; CAP = 40.0; LOAD_REF = 2 * CAP
W_REMOVAL = 100.0; W_UNHELD = 120.0
CUT_REF = 0.0432                      # m3, 64.8 kg
CAPTURE_MIN = 0.50; SHAPE_FRAC = 0.30

def run(name, steps):
    """steps: list of (above_volume_m3, payload_per_drum_kg)."""
    above_init = steps[0][0]
    rem_prev = unheld_prev = 0.0
    rem_sum = unheld_sum = 0.0
    fresh = True
    for above, per_drum in steps:
        payload = min(per_drum, CAP) * 2
        removal = max((above_init - above) * SOIL, 0.0)
        unheld = abs(removal - payload)
        rd = 0.0 if fresh else removal - rem_prev
        ud = 0.0 if fresh else unheld - unheld_prev
        rem_sum += W_REMOVAL * rd / LOAD_REF
        unheld_sum += -W_UNHELD * ud / LOAD_REF
        rem_prev, unheld_prev, fresh = removal, unheld, False
    shape_ok = (steps[-1][0] <= SHAPE_FRAC * CUT_REF
                and payload >= CAPTURE_MIN * removal)
    total = rem_sum + unheld_sum
    print(f"{name:26s} removed {removal:6.1f}  held {payload:6.1f}  "
          f"removal {rem_sum:+7.2f}  unheld {unheld_sum:+7.2f}  "
          f"net {total:+7.2f}  success {shape_ok}")
    assert abs(rem_sum - W_REMOVAL * removal / LOAD_REF) < 1e-9, "removal must telescope"
    assert abs(unheld_sum + W_UNHELD * unheld / LOAD_REF) < 1e-9, "unheld must telescope"
    return total

full = CUT_REF
def ramp(frac_pairs): return [(full * (1 - f), p) for f, p in frac_pairs]
fr = (0, .25, .5, .75, .9)
exc   = ramp([(f, 0.733 * full * f * SOIL / 2) for f in fr])   # measured capture
ideal = ramp([(f, full * f * SOIL / 2) for f in fr])           # keeps all it cuts
push  = ramp([(f, 0.0) for f in fr])                           # displaces it
pack  = ramp([(f, 86.0 * f) for f in fr])                      # buries the drums
off   = [(full, 20.0 * f) for f in fr]                         # mines elsewhere

i = run("ideal: keeps what it cuts", ideal)
a = run("measured capture 0.733", exc)
b = run("displacement (push)", push)
d = run("packing the drums", pack)
c = run("off-target mining", off)
print()
assert i > a > d, f"ideal {i:.1f} > honest {a:.1f} > packing {d:.1f}"
assert b < 0 and c < 0, "pushing and off-target mining must both lose points"
print(f"ordering holds: ideal {i:.1f} > honest {a:.1f} > packing {d:.1f} "
      f"> push {b:.1f}, off-target {c:.1f}")
