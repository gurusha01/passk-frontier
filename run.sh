#!/bin/bash
# Wave 1: 7 training arms, one GPU each on host-a. `base` needs no training.
# Every arm sees exactly steps*P*G = 120*16*8 = 15360 rollouts.
cd ${PF_ROOT}
PY=${PF_PY_ENV}/bin/python
export HF_HOME=${PF_HF_HOME}
mkdir -p runs logs

launch () {  # gpu arm objective extra...
  local gpu=$1 arm=$2 obj=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py \
    --objective "$obj" --out_dir runs/"$arm" "$@" \
    > logs/"$arm".log 2>&1 &
  echo "launched $arm (obj=$obj) on gpu $gpu -> logs/$arm.log"
}

# opt_bs 128 == n_prompts*num_gen, so EVERY arm gets exactly 120 optimizer steps of batch
# 128 on 15360 rollouts. The arms are then matched on both axes and differ only in the
# objective and, for the SFT ladder, in when the rollout policy is refreshed.
#
# --- the SFT ladder: same objective, different flush schedule ---------------------
launch 0 sft-offline raft  --flush_every 120 --opt_bs 128   # policy frozen during collection
launch 1 sft-iter    raft  --flush_every 30  --opt_bs 128   # 4 ReST rounds
launch 2 sft-online  raft  --flush_every 1   --opt_bs 128   # fully on-policy
# --- RL arms (buffer is already 128, so opt_bs default 0 == 128) -------------------
launch 3 grpo        grpo
launch 4 maxrl       maxrl
launch 5 entropic    entropic
launch 6 hientropy   hientropy

wait
echo "wave 1 done"
