sudo bash -c '
DIR=/lambda/nfs/hfm/qasim/hand_kp_dataset/interhands2.6m/InterHand2.6M_5fps_batch1
LOG=/root/interhand_extract.log
TARGET=$((77 * 1024*1024*1024))
prev=0; prevt=0
while true; do
  b=$(du -sb "$DIR" 2>/dev/null | awk "{print \$1}")
  n=$(find "$DIR" -type f 2>/dev/null | wc -l)
  t=$(date +%s)
  gb=$(awk -v b="$b" "BEGIN{printf \"%.1f\", b/1073741824}")
  pct=$(awk -v b="$b" -v T="$TARGET" "BEGIN{printf \"%.1f\", 100*b/T}")
  line="$(date +%H:%M:%S)  ${gb}GB (${pct}%)  ${n} files"
  if [ "$prevt" -gt 0 ]; then
    dt=$((t-prevt))
    rate=$(awk -v a="$prev" -v c="$b" -v dt="$dt" "BEGIN{printf \"%.1f\", (c-a)/dt/1048576}")
    eta=$(awk -v r="$((TARGET-b))" -v a="$prev" -v c="$b" -v dt="$dt" "BEGIN{s=(c-a)/dt; if(s>0) printf \"%.0f min\", r/s/60; else print \"--\"}")
    line="$line  +${rate}MB/s  ETA ~${eta}"
  fi
  echo "$line"
  # stop when the tar process is gone
  pgrep -f "tar -xvf" >/dev/null || { echo "tar finished."; break; }
  prev=$b; prevt=$t; sleep 30
done
'