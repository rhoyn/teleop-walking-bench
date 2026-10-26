#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  echo "usage: $0 [--benchmark tour|rotate|direction]" >&2
  exit 2
}
MODE=tour
while [ $# -gt 0 ]; do
  case "$1" in
    --benchmark) [ $# -ge 2 ] || usage; MODE=$2; shift 2 ;;
    --benchmark=*) MODE=${1#*=}; shift ;;
    *) usage ;;
  esac
done
case "$MODE" in
  tour | rotate | direction) ;;
  *) usage ;;
esac

NGPUS=$(nvidia-smi -L 2>/dev/null | grep -c "  MIG " || true)
[ "$NGPUS" -ge 1 ] || NGPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)
[ "$NGPUS" -ge 1 ] || NGPUS=1
JOBS=${JOBS:-$((8 * NGPUS))}
RUNID=${RUNID:-0}
[ "$MODE" = tour ] || RUNID=0
NICE=${NICE:-10}
CPUS=1-$(($(nproc) - 1))
if [ "$MODE" = tour ]; then
  WORK=${WORK:-build/preview-clips}
  VIDEO=${VIDEO:-assets/preview.mp4}
  IMAGE=${IMAGE:-assets/preview.jpg}
  STILL=${STILL:-44.6}
  CRF=${CRF:-24}
else
  WORK=${WORK:-build/preview-clips-$MODE}
  VIDEO=${VIDEO:-assets/preview_$MODE.mp4}
  IMAGE=${IMAGE:-assets/preview_$MODE.jpg}
  STILL=${STILL:-9}
fi
FPS=${FPS:-60}
CRF=${CRF:-20}
MAX_BYTES=${MAX_BYTES:-10000000}
BIN=${BIN:-build/teleop-walking-benchmark}
DELAY=${DELAY:-20}
export MODE RUNID NICE CPUS WORK BIN DELAY SCALE NGPUS

SCALE=1
[ "$MODE" = tour ] || SCALE=${SCALE_MOVEMENTS:-2}
COLS=10 ROWS=9 BG=0x3D9356
TW=$((128 * SCALE)) TH=$((72 * SCALE)) OW=1280 OH=720
FONT=assets/JetBrainsMono.ttf
ENGINES=${ENGINES:-"physx mujoco"}

if [ "$MODE" = tour ]; then
  POLICIES=${POLICIES:-$(build/table results/punches_gainx1.0/*.csv | sed 's/`<br>`/_/g' |
    sed -n 's/^| ~*`\([a-z0-9_]*\)`.*/\1/p')}
else
  POLICIES=${POLICIES:-$(ls results/cmd_vs_real_movements/[0-9][0-9][0-9]_*.png |
    sed 's|.*/[0-9]*_\(.*\)\.png|\1|')}
fi
[ -n "$POLICIES" ] || { echo "no policies to record" >&2; exit 1; }
TOTAL=$(($(echo $POLICIES | wc -w) * $(echo $ENGINES | wc -w)))
export TOTAL
if [ "$MODE" != tour ]; then
  COLS=$((4 * $(echo $ENGINES | wc -w)))
  TW=$((OW / COLS)) TH=$(((OW / COLS * 9 / 16 + 1) / 2 * 2))
  ROWS=$(((TOTAL + COLS) / COLS))
  OH=$((ROWS * TH))
fi
[ "$TOTAL" -lt $((COLS * ROWS)) ] ||
  { echo "$TOTAL clips + label do not fit ${COLS}x${ROWS} tiles" >&2; exit 1; }

RUN='
  out=$WORK/$1/$0
  sleep $((RANDOM % (DELAY + 1)))
  CUDA_VISIBLE_DEVICES=$(($2 % NGPUS)) nice -n "$NICE" taskset -c "$CPUS" "$BIN" --mode "$MODE" --engine "$1" --policy "$0" \
    --runid "$RUNID" --record "$WORK/$1" --record-scale "$SCALE" --csv "$out.csv" \
    >"$out.log" 2>&1 &&
    echo "[$(ls "$WORK"/*/*.mp4 2>/dev/null | wc -l)/$TOTAL] $0 on $1" ||
    echo "FAILED $0 on $1" >&2'

nice -n "$NICE" taskset -c "$CPUS" make
nice -n "$NICE" taskset -c "$CPUS" make table
rm -rf "$WORK"
for e in $ENGINES; do mkdir -p "$WORK/$e"; done
echo "recording $TOTAL clips, $JOBS at a time"

i=0
for p in $POLICIES; do
  for e in $ENGINES; do
    echo "$p $e $i"
    i=$((i + 1))
  done
done | xargs -P "$JOBS" -n 3 bash -c "$RUN"

clips=() layout= scale= stack= n=0
for p in $POLICIES; do
  for e in $ENGINES; do
    clips+=(-i "$WORK/$e/$p.mp4")
    layout+="${layout:+|}$((n % COLS * TW))_$((n / COLS * TH))"
    scale+="[$n:v]scale=$TW:$TH:flags=lanczos[v$n];"
    stack+="[v$n]"
    n=$((n + 1))
  done
done
pad=$(((OH - ROWS * TH) / 2))
[ "$n" -gt 1 ] && stack+="xstack=inputs=$n:layout=$layout:fill=$BG,"

FILTER="$scale$stack
  pad=$OW:$OH:0:$pad:color=$BG,
  drawtext=fontfile=$FONT:text='r h o y n':fontcolor=white:fontsize=$((10 * TW / 128))
    :x=$((n % COLS * TW + 37 * TW / 128)):y=$((n / COLS * TH + pad + 30 * TH / 72)),
  format=yuv420p"

echo "combining $n clips"
nice -n "$NICE" taskset -c "$CPUS" ffmpeg -hide_banner -loglevel error -stats -y \
  "${clips[@]}" -filter_complex "$FILTER" \
  -c:v libx264 -preset slow -crf 18 -an "$WORK/montage.mp4"

echo "re-encoding into $VIDEO at $FPS fps"
nice -n "$NICE" taskset -c "$CPUS" ffmpeg -hide_banner -loglevel error -stats -y \
  -err_detect aggressive -fflags discardcorrupt -i "$WORK/montage.mp4" \
  -r "$FPS" -c:v libx264 -preset veryslow -crf "$CRF" -pix_fmt yuv420p \
  -movflags +faststart -an "$VIDEO"
[ "$(stat -c %s "$VIDEO")" -lt "$MAX_BYTES" ] ||
  { echo "$VIDEO is $(stat -c %s "$VIDEO") bytes, limit $MAX_BYTES" >&2; exit 1; }

ffmpeg -hide_banner -loglevel error -y -ss "$STILL" -i "$VIDEO" \
  -frames:v 1 -q:v 2 "$IMAGE"
[ -s "$IMAGE" ] || { echo "no still at ${STILL}s of $VIDEO" >&2; exit 1; }

echo "wrote $VIDEO and $IMAGE"
