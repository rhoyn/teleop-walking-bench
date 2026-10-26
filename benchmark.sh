#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

usage() {
  echo "usage: $0 [--benchmark tour|rotate|direction|all]" >&2
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
  tour | rotate | direction | all) ;;
  *) usage ;;
esac

if [ "$MODE" = all ]; then
  for m in tour rotate direction; do "$0" --benchmark "$m"; done
  exit
fi

if [ "$MODE" = tour ]; then
  RUNS=${RUNS:-512}
  ROUNDS=${ROUNDS:-5}
  BASE=punches
  PREFIX=p
else
  RUNS=${RUNS:-1024}
  ROUNDS=${ROUNDS:-4}
  BASE=$MODE
  PREFIX=${MODE:0:1}
fi
LEG_GAINS="1.0 1.5 2.0"
GAIN_RUNS=${GAIN_RUNS:-1024}
NGPUS=$(nvidia-smi -L 2>/dev/null | grep -c "  MIG " || true)
[ "$NGPUS" -ge 1 ] || NGPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)
[ "$NGPUS" -ge 1 ] || NGPUS=1
JOBS=${JOBS:-$((4 * NGPUS))}
NICE=${NICE:-10}
CPUS=1-$(($(nproc) - 1))
NCORES=$(($(nproc) - 1))
CORESEQ=$(mktemp -t benchmark-core.XXXXXX)
echo 0 >"$CORESEQ"
GPUDIR=$(mktemp -d -t benchmark-gpu.XXXXXX)
GPUSLOTS=$(((JOBS + NGPUS - 1) / NGPUS))
export MODE BASE PREFIX RUNS GAIN_RUNS CPUS NICE NCORES CORESEQ NGPUS GPUDIR GPUSLOTS

ONLY=${POLICIES:-}
POLICIES=$(for d in policies/*/; do [ -f "$d/policy.cpp" ] && basename "$d"; done |
  grep -vE '^(decoupled_wbc|gr00t_wbc|mimic_lite|morphmimic|supermimic|supermimic_breezewrf|supermimic_vibed)$')
POLICIES="$POLICIES clobot_with_arms handoff_with_arms"
TRACKERS="mimic_lite morphmimic supermimic_breezewrf"
for v in kangzejian kyrieyoung rayzhao dyn m5 panhainan omegai; do
  TRACKERS="$TRACKERS supermimic_$v"
done
for t in $TRACKERS; do
  POLICIES="$POLICIES ${t}_sonicplnr ${t}_openwbtplnr ${t}_gr00twbcplnr"
done
for h in h074_p000 h070_p000 h066_p000 h066_p012; do
  POLICIES="$POLICIES gr00t_wbc_$h decoupled_wbc_$h"
done

TIER_A='
gr00t_wbc_h066_p012
gr00t_wbc_h066_p000
gr00t_wbc_h070_p000
gr00t_wbc_h074_p000
decoupled_wbc_h066_p000
decoupled_wbc_h066_p012
decoupled_wbc_h070_p000
decoupled_wbc_h074_p000
homie
wbc_agile_velocity
amo
morphmimic_sonicplnr
morphmimic_gr00twbcplnr
'

RUN='
  ROUND=$0
  P=$1
  ENGINE=$2
  G=$3
  DIR=results/${BASE}_gainx$G
  R=$(printf "%s%02d" "$PREFIX" "$ROUND")
  n=$RUNS
  [ "$G" = 1.0 ] || n=$GAIN_RUNS
  f=$((ROUND * n))
  out=$R.$P.$ENGINE
  log=$out.gainx$G
  CORE=$(
    exec 9>"$CORESEQ.lock"
    flock 9
    i=$(cat "$CORESEQ")
    echo $(((i + 1) % NCORES)) >"$CORESEQ"
    echo $((1 + i))
  )
  GPU=
  while [ -z "$GPU" ]; do
    for s in $(seq 0 $((GPUSLOTS - 1))); do
      for g in $(seq 0 $((NGPUS - 1))); do
        exec 8>"$GPUDIR/$g.$s"
        if flock -n 8; then GPU=$g; break 2; fi
        exec 8>&-
      done
    done
    [ -z "$GPU" ] && sleep 1
  done
  if CUDA_VISIBLE_DEVICES="$GPU" chrt --idle 0 nice -n "$NICE" taskset -c "$CORE" build/teleop-walking-benchmark \
      --mode "$MODE" --engine "$ENGINE" --policy "$P" --runids "$f-$((f + n - 1))" \
      --leg-gain "$G" --csv "$DIR/$out.csv" >"progress/$log.log" 2>&1; then
    grep "^mujoco:" "progress/$log.log" | sed "s|^|$log |" >>benchmark.log
  else
    echo "FAILED $log" >&2
    { echo "FAILED $log"; cat "progress/$log.log"; } >>benchmark.log
  fi
  rm -f "progress/$log.log"'

chrt --idle 0 nice -n "$NICE" taskset -c "$CPUS" make
make capture MJWARP_NWORLD="$((RUNS > GAIN_RUNS ? RUNS : GAIN_RUNS))"
mkdir -p progress
for g in $LEG_GAINS; do mkdir -p "results/${BASE}_gainx$g"; done
rm -f progress/*.log benchmark.log

CPUPREV=$(mktemp -t benchmark-cpu.XXXXXX)
grep '^cpu[0-9]' /proc/stat >"$CPUPREV"

while sleep 1; do
  printf '\033[H\033[2J'
  for f in progress/*.log; do
    [ -e "$f" ] || continue
    printf '%-34s %8s robot-sim-s/s\n' "$(basename "$f" .log)" \
      "$(grep -o '[0-9]\+ robot-sim-s/s' "$f" | tail -1 | cut -d' ' -f1)"
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.max_limit \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' '{printf "gpu%-2s %3s%%  %6s/%-6s MiB  %5.0f/%-5.0f W\n", $1, $2, $3, $4, $5, $6}'
  grep '^cpu[0-9]' /proc/stat >"$CPUPREV.new"
  awk 'NR==FNR { for (i = 2; i <= NF; i++) p[$1, i] = $i; next }
       { t = 0; for (i = 2; i <= NF; i++) t += $i - p[$1, i]
         idle = ($5 - p[$1, 5]) + ($6 - p[$1, 6])
         printf "%4.0f", (t > 0) ? 100 * (t - idle) / t : 0
         if (++n % 16 == 0) printf "\n" }
       END { if (n % 16) printf "\n" }' "$CPUPREV" "$CPUPREV.new"
  mv "$CPUPREV.new" "$CPUPREV"
done &
MON=$!
trap 'kill $MON 2>/dev/null || true; rm -rf "$CORESEQ" "$CORESEQ.lock" "$GPUDIR" "$CPUPREV" "$CPUPREV.new"' EXIT

for r in $(seq 0 $((ROUNDS - 1))); do
  for p in ${ONLY:-$POLICIES}; do
    for g in $LEG_GAINS; do
      rounds=1
      [ "$MODE" = tour ] && [ "$g" = 1.0 ] && echo "$TIER_A" | grep -qxF "$p" &&
        rounds=$ROUNDS
      [ "$MODE" = tour ] || rounds=$ROUNDS
      [ "$r" -lt "$rounds" ] || continue
      dir=results/${BASE}_gainx$g
      name=$(printf "%s%02d" "$PREFIX" "$r").$p
      for e in mujoco physx; do
        [ -s "$dir/$name.$e.csv" ] && continue
        echo "$r $p $e $g"
      done
    done
  done
done | xargs -r -P "$JOBS" -n 4 bash -c "$RUN"
