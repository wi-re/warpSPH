#!/bin/bash
# Overnight batch, proposed-default-scheme variant of run_overnight_batch.sh
# (2026-09-13): same three-stage structure (Marrone 3.1 -> Marrone 3.4 ->
# stop-if-diverged -> full gallery), plus a dedicated sloshingTank leg, all
# three scheme-relevant runs using the accumulated best-known WCSPH combo
# from BOUNDARY_DENSITY_PLAN.md's investigation:
#
#   shifting=True (case default, left alone)
#   --noPenShift finalize        (DualSPHysics-style once-per-step placement,
#                                  DELTASPH_VALIDATION_PLAN.md 5.13 -- never
#                                  promoted to the case default itself)
#   --integrationScheme symplecticEuler
#   --densityDiffusionTerm fourtakas2019
#   --mdbcDensityScheme english2025   (sign-fixed + eps=1e-30-regularized --
#                                       "the clamp" -- BOUNDARY_DENSITY_PLAN.md
#                                       Section 9)
#
# Deliberately NOT included: --pressureForceRenormalized (the completeness
# gate, Section 10 of that plan) -- tested 2026-09-17, confirmed a mixed
# result on Marrone 3.1 and literally zero effect on the ceiling-hover
# mechanism, so it stays off here; this run is testing the combo that is
# actually proposed, not every knob this investigation has touched.
#
# The gallery step (render_examples.py, --only weaklyCompressible) runs each
# example at its own shipped settings -- no forced overrides. It used to force
# densityDiffusionTerm/integrationScheme to fourtakas2019/symplecticEuler on
# the theory that "these examples concern free-surface behaviour, so those
# two knobs still apply" -- wrong for most of the 13: only a few (dambreak,
# impact, open-flow) have a free surface at all, and forcing symplecticEuler
# on the rest actively broke two that were previously fine, unrelated to
# either knob:
#   - lidDrivenCavity blew up (density -> ~2x rho0, velocities > 20x the lid
#     speed). Root cause had nothing to do with this combo: boundary/ghost
#     particles carrying a nonzero BC-prescribed velocity (the lid's Dirichlet
#     condition) drifted in *position* under symplecticEuler's second,
#     semi-implicit half-step, which reads raw current velocity instead of
#     the masked derivative every other update path uses -- 1.5 domain-widths
#     of drift by t=3, wrecking the lattice at the lid interface. Fixed in
#     `systems/weaklyCompressible.py`'s `finalize()` (boundary positions are
#     now explicitly restored every step, matching the existing density
#     restore right above it) -- symplecticEuler now matches rungeKutta2's
#     numbers on this case, so the override is no longer the risk it was, but
#     there's no reason to force it broadly now that the actual bug is fixed.
#   - randomFlow-periodic decayed much faster than its own 2026-08-14
#     baseline -- looked like a regression but wasn't: bisected to
#     `3e7b78e`'s two-sided viscosity fix, which also corrected TGV's
#     previously-wrong half-rate decay. Neither this combo's knobs nor the
#     integrator moved that number at all in testing; it's the physically
#     correct result for this field now, not a regression this batch's
#     scheme choice caused. Nothing to override here either.
# This step is a broad regression net ("did anything else break" against each
# example's own default), not a re-run of the three targeted legs above --
# letting each example run at its shipped default is what that actually
# requires.
#
# All output, raw run trees and the published gif/mp4/png alike, lands under
# one folder:
#   examples/weaklyCompressible/output/overnight_2026-09-17/
# render_examples.py's [4/4] step used to ALSO copy each gallery example's
# gif/mp4/png into its own (git-tracked) examples/weaklyCompressible/.../outputs/
# folder unconditionally -- every nightly pass silently refreshed the shipped
# docs media in place, so `git status` would show those files modified
# whether or not anything had actually changed, and an accidental `git add -A`
# could ship a regenerated video nobody reviewed. `--publishRoot
# "$BASE/gallery/published"` (see [4/4] below) redirects that copy under this
# same output folder instead, alongside the raw run trees -- nothing under
# `examples/` gets touched by this script anymore. Deliberately refreshing the
# shipped docs media is still `scripts/render_examples.py --only weaklyCompressible`
# with no `--publishRoot`, unchanged.
set -o pipefail
cd /home/lu26029/dev/warpSPH

BASE=examples/weaklyCompressible/output/overnight_2026-09-17
mkdir -p "$BASE"
LOG=$BASE/batch.log
exec > >(tee -a "$LOG") 2>&1

source /home/lu26029/miniconda3/etc/profile.d/conda.sh
conda activate warp

SCHEME_OVERRIDES="--noPenShift finalize --integrationScheme symplecticEuler --densityDiffusionTerm fourtakas2019 --mdbcDensityScheme english2025"

echo "=== overnight batch (proposed default scheme) started $(date) ==="
echo "scheme overrides applied to marrone31/marrone34/sloshingTank: $SCHEME_OVERRIDES"

echo
echo "--- [1/4] Marrone 3.1 (nx=67, scheme=deltaSPH, shifting=default i.e. PST on, tLimit=5.0s ~ t*=20.2) ---"
python scripts/probe_deltaSPHMarrone.py --scheme deltaSPH --nx 67 --tLimit 5.0 \
  $SCHEME_OVERRIDES \
  --video --out "$BASE/marrone31" 2>&1 | tee "$BASE/marrone31.log"

echo
echo "--- [2/4] Marrone 3.4 (nx=256, scheme=deltaSPH, shifting=off i.e. Marrone's own Sec.3 spec, tStar=20) ---"
python scripts/probe_deltaSPHMarrone34.py --nx 256 --tStar 20 \
  $SCHEME_OVERRIDES \
  --video --out "$BASE/marrone34" 2>&1 | tee "$BASE/marrone34.log"

echo
echo "--- [3/4] sloshingTank (nx=200 case default, full tLimit=7.0s SPHERIC TC10 reference duration) ---"
python examples/sloshingTank/run_sloshingTank.py --tLimit 7.0 \
  --mdbcDensityScheme english2025 --densityDiffusionTerm fourtakas2019 \
  --integrationScheme symplecticEuler --noPenShift finalize \
  --video --out "$BASE/sloshingTank" 2>&1 | tee "$BASE/sloshingTank.log"

echo
echo "--- checking all three scheme-relevant runs for divergence before continuing ---"
m31_diverged=$(grep -o "diverged=True" "$BASE/marrone31.log" | tail -1)
m34_diverged=$(grep -o "diverged=True" "$BASE/marrone34.log" | tail -1)
slosh_diverged=$(grep -o "diverged=True\|diverged = True" "$BASE/sloshingTank.log" | tail -1)

if [[ -n "$m31_diverged" || -n "$m34_diverged" || -n "$slosh_diverged" ]]; then
  echo "STOPPING: at least one scheme-relevant run diverged (m31: ${m31_diverged:-clean}, m34: ${m34_diverged:-clean}, sloshingTank: ${slosh_diverged:-clean})."
  echo "Not proceeding to the examples/weaklyCompressible gallery batch."
  echo "=== overnight batch ended (stopped early) $(date) ==="
  exit 1
fi

echo "All three scheme-relevant runs completed without diverging -- proceeding to the gallery."
echo
echo "--- [4/4] examples/weaklyCompressible gallery (render_examples.py, each example at its own shipped default settings -- see the header comment for why this no longer forces fourtakas2019/symplecticEuler) ---"
python scripts/render_examples.py --only weaklyCompressible --outRoot "$BASE/gallery" --publishRoot "$BASE/gallery/published" 2>&1 | tee "$BASE/gallery.log"

echo
echo "=== overnight batch finished $(date) ==="
