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
# example at its own shipped settings EXCEPT densityDiffusionTerm and
# integrationScheme, forced to fourtakas2019/symplecticEuler -- these
# examples concern free-surface behaviour, so those two knobs (the ones a
# non-mDBC WCSPH case can actually have an opinion on) still apply.
# mdbcDensityScheme/noPenShift are left alone: most gallery examples have no
# boundary/mDBC particles at all, so that knob wouldn't do anything there,
# and this step is still primarily a broad regression net ("did anything
# else break"), not a re-run of the three targeted legs above.
# `densityDiffusionTerm` isn't a generic CaseSpec field the way
# `integrationScheme` is (`caseMain` has no per-example CLI flag for it), so
# it's forced via `WARPSPH_DEFAULT_DDT` -- see
# configurations/moduleConfigurations/weaklyCompressibleDiffusionParams.py's
# DIAGNOSTIC ONLY block, a no-op for everything else that doesn't set it.
#
# All raw run output lands under one folder:
#   examples/weaklyCompressible/output/overnight_2026-09-17/
# NOTE: render_examples.py ALSO copies each gallery example's gif/mp4/png
# into that example's own (git-tracked) examples/weaklyCompressible/.../outputs/
# folder -- refreshes shipped documentation media in place, `git status` will
# show those files modified afterward. Left uncommitted deliberately for review.
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
echo "--- [4/4] examples/weaklyCompressible gallery (render_examples.py, shipped settings per example, EXCEPT densityDiffusionTerm=fourtakas2019 + integrationScheme=symplecticEuler forced -- these examples concern free-surface behaviour, so the two knobs this combo actually changes for a WCSPH case still apply. mdbcDensityScheme/noPenShift are left at each example's own default -- most gallery examples have no mDBC boundary particles at all, so that knob wouldn't do anything there.) ---"
WARPSPH_DEFAULT_DDT=fourtakas2019 python scripts/render_examples.py --only weaklyCompressible --outRoot "$BASE/gallery" -- --integrationScheme symplecticEuler 2>&1 | tee "$BASE/gallery.log"

echo
echo "=== overnight batch finished $(date) ==="
