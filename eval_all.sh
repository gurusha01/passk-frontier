#!/bin/bash
# pass@k eval for every finished arm. Round-robins arms across the GPUs given in $GPUS.
# Usage: GPUS="0 1 2 3 4 5 6 7" bash eval_all.sh [arm ...]
cd ${PF_ROOT}
PY=${PF_VLLM}/bin/python
export HF_HOME=${PF_HF_HOME}
mkdir -p evals logs
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
read -ra G <<< "$GPUS"

arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then
  arms=()
  for d in runs/*/final; do [ -d "$d" ] && arms+=("$(basename "$(dirname "$d")")"); done
fi

i=0
for arm in "${arms[@]}"; do
  [ -f "evals/$arm.jsonl" ] && { echo "skip $arm (already evaluated)"; continue; }
  [ -d "runs/$arm/final" ] || { echo "skip $arm (no checkpoint yet)"; continue; }
  gpu=${G[$((i % ${#G[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PY -u evalk.py \
    --model runs/"$arm"/final --out evals/"$arm".jsonl --n 64 --greedy \
    > logs/eval-"$arm".log 2>&1 < /dev/null &
  echo "eval $arm on gpu $gpu"
  i=$((i + 1))
  # one vLLM engine per GPU at a time: wait for a full round before starting the next
  [ $((i % ${#G[@]})) -eq 0 ] && wait
done
wait
echo "evals done"
