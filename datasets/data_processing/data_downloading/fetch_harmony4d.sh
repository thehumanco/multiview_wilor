#!/usr/bin/env bash
# Extract the harmony4d dataset zips IN PLACE (train/ and test/ splits).
# The dataset dir is owned by qasim and not writable by us, so this MUST
# run as root:   sudo bash ~/prometheus/fetch_harmony4d.sh
#
# Zips are stored uncompressed (method=store), so extraction is I/O bound.
# A background monitor logs progress every 30s: bytes done, %, rate, ETA.
set -euo pipefail

DST="/lambda/nfs/hfm/qasim/hand_kp_dataset/harmony4d"
LOG="$DST/_fetch_harmony4d.log"
SPLITS=(train test)

if [ "$(id -u)" -ne 0 ]; then
  echo "ERROR: must run as root (the dataset dir is owned by qasim)."
  echo "Run:  sudo bash $0"
  exit 1
fi
if [ ! -d "$DST" ]; then
  echo "ERROR: $DST not found"; exit 1
fi

echo "=== harmony4d extract started $(date -Is) ===" | tee -a "$LOG"

# --- Compute the total uncompressed bytes across all zips (for ETA/%) -------
# unzip -l prints a trailing summary line "<bytes> <nfiles> files"; sum those.
echo "Scanning archives to compute total size (this takes a moment)..." | tee -a "$LOG"
TOTAL_BYTES=0
declare -a ZIPS=()
for split in "${SPLITS[@]}"; do
  for z in "$DST/$split"/*.zip; do
    [ -e "$z" ] || continue
    ZIPS+=("$split|$z")
    b=$(unzip -l "$z" | awk 'END{print $1}')
    TOTAL_BYTES=$((TOTAL_BYTES + b))
  done
done
echo "Total to extract: $TOTAL_BYTES bytes ($(numfmt --to=iec "$TOTAL_BYTES")) across ${#ZIPS[@]} archives" | tee -a "$LOG"

# Baseline: bytes already present in the split dirs that are NOT the zips,
# so % reflects only newly-extracted data on a resumed run.
dir_extracted_bytes() {
  local total=0 split b
  for split in "${SPLITS[@]}"; do
    # size of split dir minus the zip files themselves
    local dirb zipb
    dirb=$(du -sb "$DST/$split" 2>/dev/null | awk '{print $1}'); dirb=${dirb:-0}
    zipb=$(du -scb "$DST/$split"/*.zip 2>/dev/null | awk 'END{print $1}'); zipb=${zipb:-0}
    total=$((total + dirb - zipb))
  done
  echo "$total"
}
BASE_BYTES=$(dir_extracted_bytes)

# --- Background progress monitor: logs every 30s ----------------------------
START_TS=$(date +%s)
monitor() {
  while true; do
    sleep 30
    now=$(date +%s)
    cur=$(dir_extracted_bytes)
    done_b=$((cur - BASE_BYTES)); [ "$done_b" -lt 0 ] && done_b=0
    elapsed=$((now - START_TS)); [ "$elapsed" -lt 1 ] && elapsed=1
    pct=$(awk -v d="$done_b" -v t="$TOTAL_BYTES" 'BEGIN{if(t>0)printf "%.1f",100*d/t; else printf "0.0"}')
    rate=$((done_b / elapsed))                       # bytes/sec
    if [ "$rate" -gt 0 ]; then
      remain=$(( (TOTAL_BYTES - done_b) / rate ))
      eta=$(printf '%dh%02dm%02ds' $((remain/3600)) $((remain%3600/60)) $((remain%60)))
    else
      eta="--"
    fi
    printf '[%s] %s / %s (%s%%)  rate=%s/s  elapsed=%s  ETA=%s\n' \
      "$(date +%T)" \
      "$(numfmt --to=iec "$done_b")" "$(numfmt --to=iec "$TOTAL_BYTES")" \
      "$pct" "$(numfmt --to=iec "$rate")" \
      "$(printf '%dh%02dm%02ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))" \
      "$eta" | tee -a "$LOG"
  done
}
monitor & MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT

# --- Extract each archive into its split dir (-n: resumable, never clobber) -
for entry in "${ZIPS[@]}"; do
  split="${entry%%|*}"; z="${entry#*|}"
  echo "=== $(date +%T) unzip $z ($(numfmt --to=iec "$(stat -c%s "$z")")) ===" | tee -a "$LOG"
  unzip -n -q "$z" -d "$DST/$split" && echo "    OK: $z" | tee -a "$LOG"
done

kill "$MON_PID" 2>/dev/null || true
echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG"
ls -la "$DST/train" "$DST/test" | tee -a "$LOG"
