#!/usr/bin/env bash
# Download + extract HO-3D v3 from the OneDrive remote into qasim's ho3d dir.
# Idempotent: rclone skips already-downloaded files; unzip -n skips existing.
# During extraction, prints extracted / remaining / ETA every 30s.
# Run inside tmux:  bash ~/prometheus/fetch_ho3d.sh
set -euo pipefail

RCLONE="$HOME/bin/rclone"
SRC="onedrive:HO3D_v3"
DST="/lambda/nfs/hfm/qasim/hand_kp_dataset/ho3d"
LOG="$DST/_fetch_ho3d.log"

echo "=== HO3D fetch started $(date -Is) ===" | tee -a "$LOG"

if [ ! -w "$DST" ]; then
  echo "ERROR: $DST not writable. Run:  sudo chmod 777 $DST" | tee -a "$LOG"
  exit 1
fi

# --- Download (skips files already present at correct size) -----------------
"$RCLONE" copy "$SRC" "$DST" \
  --transfers 4 --checkers 8 \
  --progress \
  --stats 30s --stats-one-line \
  --log-file "$LOG" --log-level INFO

echo "=== download done $(date -Is), verifying ===" | tee -a "$LOG"
"$RCLONE" lsl "$SRC" | tee -a "$LOG"

# --- Background progress monitor for extraction -----------------------------
# Polls extracted bytes (everything except the .zip files) every 30s and
# compares against the known uncompressed total to print remaining + ETA.
monitor_extract() {
  local total_bytes="$1"           # expected uncompressed bytes for this archive
  local base_bytes="$2"            # bytes already on disk before this unzip
  local start_ts; start_ts=$(date +%s)
  while :; do
    sleep 30
    local now cur done_b pct rate eta_s
    now=$(date +%s)
    cur=$(du -sb --exclude='*.zip' "$DST" 2>/dev/null | awk '{print $1}')
    done_b=$(( cur - base_bytes ))
    (( done_b < 0 )) && done_b=0
    local elapsed=$(( now - start_ts )); (( elapsed < 1 )) && elapsed=1
    rate=$(( done_b / elapsed ))                       # bytes/sec
    pct=$(awk -v d="$done_b" -v t="$total_bytes" 'BEGIN{printf (t>0)?"%.1f":"?", (d/t)*100}')
    if (( rate > 0 )); then
      eta_s=$(( (total_bytes - done_b) / rate ))
      (( eta_s < 0 )) && eta_s=0
    else
      eta_s=0
    fi
    printf '[extract %s] %.2f / %.2f GiB (%s%%) | %.1f MiB/s | ETA %dm%02ds\n' \
      "$(date +%H:%M:%S)" \
      "$(awk -v b="$done_b" 'BEGIN{print b/1073741824}')" \
      "$(awk -v b="$total_bytes" 'BEGIN{print b/1073741824}')" \
      "$pct" \
      "$(awk -v r="$rate" 'BEGIN{print r/1048576}')" \
      "$(( eta_s / 60 ))" "$(( eta_s % 60 ))" | tee -a "$LOG"
  done
}

cd "$DST"
for z in HO3D_v3.zip HO3D_v3_segmentations_rendered.zip; do
  [ -f "$z" ] || continue
  echo "=== unzip $z $(date -Is) ===" | tee -a "$LOG"
  # Uncompressed total for this archive (sum of entry sizes) and current on-disk base.
  total_unc=$(unzip -l "$z" | tail -1 | awk '{print $1}')
  base_now=$(du -sb --exclude='*.zip' "$DST" 2>/dev/null | awk '{print $1}')
  monitor_extract "$total_unc" "$base_now" &
  MON_PID=$!
  unzip -n -q "$z" && echo "unzipped $z" | tee -a "$LOG"
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
done

echo "=== ALL DONE $(date -Is) ===" | tee -a "$LOG"
echo "Tip: tail -f $LOG"
