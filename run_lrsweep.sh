#!/bin/bash
# Wave 1b: learning-rate sweep, on host-c.
#
# Why: at the single shared lr=1e-6, grpo/entropic improve while every non-negative-advantage
# arm (raft, maxrl) degrades monotonically. GRPO's z-scored advantage has mean ~0 so its
# gradient is a self-anchoring difference between correct and incorrect samples; raft/maxrl
# push strictly upward and saturate the grad-norm clip every step. A single lr therefore
# risks measuring lr sensitivity instead of the objectives. This sweeps BOTH sides -- the
# losing arms AND grpo -- so each objective can be read at its own best lr.
cd ${PF_ROOT}
PY=${PF_PY_ENV}/bin/python
export HF_HOME=${PF_HF_HOME}
mkdir -p runs logs

launch () {  # gpu name objective lr extra...
  local gpu=$1 name=$2 obj=$3 lr=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u train.py \
    --objective "$obj" --lr "$lr" --out_dir runs/"$name" "$@" \
    > logs/"$name".log 2>&1 &
  echo "launched $name (obj=$obj lr=$lr) on gpu $gpu"
}

launch 2 lr_sft-online_1e-7 raft  1e-7 --flush_every 1 --opt_bs 128
launch 3 lr_sft-online_3e-7 raft  3e-7 --flush_every 1 --opt_bs 128
launch 4 lr_maxrl_1e-7      maxrl 1e-7
launch 5 lr_maxrl_3e-7      maxrl 3e-7
launch 6 lr_grpo_3e-7       grpo  3e-7
launch 7 lr_grpo_3e-6       grpo  3e-6

wait
echo "lr sweep done"
