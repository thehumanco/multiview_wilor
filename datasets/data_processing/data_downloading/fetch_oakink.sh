#!/usr/bin/env bash
# Extract the OakInk-v2 dataset tarballs IN PLACE.
# The dataset dir is owned by qasim and not writable by us, so this MUST
# run as root:   sudo bash ~/prometheus/fetch_oakink.sh
#
# Extracts:
#   - 6 top-level object_*.tar / program*.tar archives  -> $DST/extracted/
#   - all 627 data/*.tar sequence archives (~2 TB)       -> $DST/data/extracted/
# Tars are KEPT (non-destructive). Each tar unpacks to its own self-named dir,
# so a run is resumable: a tar whose output dir already exists is skipped.
#
# A background monitor logs progress every 30s: bytes done, %, rate, elapsed, ETA.
set -euo pipefail

DST="/lambda/nfs/hfm/qasim/hand_kp_dataset/oakink"
DATA_DIR="$DST/data"
DATA_OUT="$DATA_DIR/extracted"      # data/*.tar      -> here
TOP_OUT="$DST/extracted"            # top-level *.tar -> here
LOG="$DST/_fetch_oakink.log"

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: must run as root (the dataset dir is owned by qasim)."
  echo "Run:  sudo bash $0"
  exit 1
fi
if [ ! -d "$DST" ]; then
  echo "ERROR: $DST not found"; exit 1
fi

mkdir -p "$DATA_OUT" "$TOP_OUT"
echo "=== oakink extract started $(date -Is) ===" | tee -a "$LOG"

# --- Build the work list: "<outdir>|<tarpath>" ------------------------------
declare -a TARS=()
for f in "$DST"/object_affordance.tar "$DST"/object_preview.tar \
         "$DST"/object_raw.tar "$DST"/object_repair.tar \
         "$DST"/program.tar "$DST"/program_extension.tar; do
  [ -e "$f" ] && TARS+=("$TOP_OUT|$f")
done
for f in "$DATA_DIR"/*.tar; do
  [ -e "$f" ] && TARS+=("$DATA_OUT|$f")
done

# --- Total bytes = sum of every tar's size (drives ETA/%) -------------------
echo "Scanning ${#TARS[@]} archives to compute total size..." | tee -a "$LOG"
TOTAL_BYTES=0
for entry in "${TARS[@]}"; do
  z="${entry#*|}"
  b=$(stat -c%s "$z")
  TOTAL_BYTES=$((TOTAL_BYTES + b))
done
echo "Total to extract: $TOTAL_BYTES bytes ($(numfmt --to=iec "$TOTAL_BYTES")) across ${#TARS[@]} archives" | tee -a "$LOG"

# Expected output dir for a tar (named after the tar, sans .tar extension).
out_dir_for() { local out="${1%%|*}" z="${1#*|}"; echo "$out/$(basename "$z" .tar)"; }

# Baseline: bytes of tars already extracted (their output dir exists), so %
# reflects only newly-extracted data on a resumed run.
done_tar_bytes() {
  local total=0 entry
  for entry in "${TARS[@]}"; do
    if [ -d "$(out_dir_for "$entry")" ]; then
      total=$((total + $(stat -c%s "${entry#*|}")))
    fi
  done
  echo "$total"
}
BASE_BYTES=$(done_tar_bytes)

# --- Background progress monitor: logs every 30s ----------------------------
START_TS=$(date +%s)
monitor() {
  while true; do
    sleep 30
    now=$(date +%s)
    cur=$(done_tar_bytes)
    done_b=$((cur - BASE_BYTES)); [ "$done_b" -lt 0 ] && done_b=0
    total_done=$cur
    elapsed=$((now - START_TS)); [ "$elapsed" -lt 1 ] && elapsed=1
    pct=$(awk -v d="$total_done" -v t="$TOTAL_BYTES" 'BEGIN{if(t>0)printf "%.1f",100*d/t; else printf "0.0"}')
    rate=$((done_b / elapsed))                       # bytes/sec this session
    remain_b=$((TOTAL_BYTES - total_done)); [ "$remain_b" -lt 0 ] && remain_b=0
    if [ "$rate" -gt 0 ]; then
      remain=$(( remain_b / rate ))
      eta=$(printf '%dh%02dm%02ds' $((remain/3600)) $((remain%3600/60)) $((remain%60)))
    else
      eta="--"
    fi
    printf '[%s] %s / %s (%s%%)  remain=%s  rate=%s/s  elapsed=%s  ETA=%s\n' \
      "$(date +%T)" \
      "$(numfmt --to=iec "$total_done")" "$(numfmt --to=iec "$TOTAL_BYTES")" \
      "$pct" "$(numfmt --to=iec "$remain_b")" "$(numfmt --to=iec "$rate")" \
      "$(printf '%dh%02dm%02ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))" \
      "$eta" | tee -a "$LOG"
  done
}
monitor & MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

# --- Extract each tar into its output dir (skip if already extracted) -------
n=0
for entry in "${TARS[@]}"; do
  n=$((n + 1))
  out="${entry%%|*}"; z="${entry#*|}"
  odir="$(out_dir_for "$entry")"
  if [ -d "$odir" ]; then
    echo "[$n/${#TARS[@]}] $(date +%T) skip (exists): $(basename "$z")" | tee -a "$LOG"
    continue
  fi
  echo "[$n/${#TARS[@]}] $(date +%T) tar xf $(basename "$z") ($(numfmt --to=iec "$(stat -c%s "$z")"))" | tee -a "$LOG"
  # -m: don't restore mtimes (archives store epoch-0 times -> harmless
  #     "implausibly old time stamp" warnings otherwise)
  if tar -xmf "$z" -C "$out"; then
    echo "    OK: $(basename "$z")" | tee -a "$LOG"
  else
    echo "    ERROR extracting: $z" | tee -a "$LOG"
  fi
done

kill "$MON_PID" 2>/dev/null || true
echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG"
du -sh "$DATA_OUT" "$TOP_OUT" 2>/dev/null | tee -a "$LOG"
