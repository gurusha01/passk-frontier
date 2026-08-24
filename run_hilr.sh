#!/bin/bash
# Wave 1c: the full wave-1 arm set again at lr 3e-6, everything else identical.
#
# Why: at lr 1e-6 GRPO moved pass@1 by +0.004 over base (0.415 -> 0.419) and pass@64 DOWN
# only slightly. The pass@1-up/pass@k-down effect this study is built on did not reproduce,
# so the run was simply undertrained -- 120 optimizer steps at 1e-6 raised train reward
# (0.470 -> 0.514) without transferring to MATH-500. Same budget, same steps, 3x lr, so
# wave1 vs wave1c is a clean single-variable lr ablation.
cd ${PF_ROOT}
PY=${PF_PY_ENV}/bin/python
export HF_HOME=${PF_HF_HOME}
mkdir -p runs logs
LR=3e-6

launch () {  # gpu arm objective extra...
  local gpu=$1 arm=$2 obj=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u train.py \
    --objective "$obj" --lr $LR --out_dir runs/"$arm" "$@" \
    > logs/"$arm".log 2>&1 &
  echo "launched $arm (obj=$obj lr=$LR) on gpu $gpu"
}

launch 0 hi_sft-offline raft  --flush_every 120 --opt_bs 128
launch 1 hi_sft-iter    raft  --flush_every 30  --opt_bs 128
launch 2 hi_sft-online  raft  --flush_every 1   --opt_bs 128
launch 3 hi_grpo        grpo
launch 4 hi_maxrl       maxrl
launch 5 hi_entropic    entropic
launch 6 hi_hientropy   hientropy

wait
echo "wave 1c done"
