#!/bin/bash
# Overnight batch: Marrone 3.1 + 3.4 at extended time horizons with the
# corrected PressureForceScheme.Antuono kernel (DELTASPH_VALIDATION_PLAN.md
# 5.14), video on; if both survive to their target time without diverging,
# continue into the full examples/weaklyCompressible gallery
# (scripts/render_examples.py, each example at its own shipped settings).
#
# All raw run output lands under one folder:
#   examples/weaklyCompressible/output/overnight_2026-09-13/
# NOTE: render_examples.py ALSO copies each gallery example's gif/mp4/png
# into that example's own (git-tracked) examples/weaklyCompressible/.../outputs/
# folder -- this refreshes the shipped documentation media in place, which is
# exactly what the tool is for, but it means `git status` will show those
# files as modified afterward. Left uncommitted deliberately for review.
set -o pipefail
cd /home/lu26029/dev/warpSPH

BASE=examples/weaklyCompressible/output/overnight_2026-09-13
LOG=$BASE/batch.log
exec > >(tee -a "$LOG") 2>&1

source /home/lu26029/miniconda3/etc/profile.d/conda.sh
conda activate warp

echo "=== overnight batch started $(date) ==="

echo
echo "--- [1/3] Marrone 3.1 (nx=67, scheme=deltaSPH, shifting=default i.e. PST on, tLimit=5.0s ~ t*=20.2) ---"
python scripts/probe_deltaSPHMarrone.py --scheme deltaSPH --nx 67 --tLimit 5.0 \
  --video --out "$BASE/marrone31" 2>&1 | tee "$BASE/marrone31.log"

echo
echo "--- [2/3] Marrone 3.4 (nx=256, scheme=deltaSPH, shifting=off i.e. Marrone's own Sec.3 spec, tStar=20) ---"
python scripts/probe_deltaSPHMarrone34.py --nx 256 --tStar 20 \
  --video --out "$BASE/marrone34" 2>&1 | tee "$BASE/marrone34.log"

echo
echo "--- checking both Marrone runs for divergence before continuing ---"
m31_diverged=$(grep -o "diverged=True" "$BASE/marrone31.log" | tail -1)
m34_diverged=$(grep -o "diverged=True" "$BASE/marrone34.log" | tail -1)

if [[ -n "$m31_diverged" || -n "$m34_diverged" ]]; then
  echo "STOPPING: at least one Marrone run diverged (m31: ${m31_diverged:-clean}, m34: ${m34_diverged:-clean})."
  echo "Not proceeding to the examples/weaklyCompressible gallery batch."
  echo "=== overnight batch ended (stopped early) $(date) ==="
  exit 1
fi

echo "Both Marrone runs completed without diverging -- proceeding to the gallery."
echo
echo "--- [3/3] examples/weaklyCompressible gallery (render_examples.py, shipped settings per example) ---"
python scripts/render_examples.py --only weaklyCompressible --outRoot "$BASE/gallery" 2>&1 | tee "$BASE/gallery.log"

echo
echo "=== overnight batch finished $(date) ==="
