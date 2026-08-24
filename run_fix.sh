#!/bin/bash
# Wave 2: corrected MaxRL, A_i = (r_i - mean)/(mean + eps).
#
# The earlier `maxrl` used the un-baselined estimator (1/K)*sum r_i S_i. Same expectation,
# but strictly non-negative, so it had no anchoring negative gradient -- which is why it
# tracked plain RAFT to within 0.004 at every lr and collapsed identically. The baselined
# form gives failures A_i = -1, sums to zero within the group, and up-weights hard prompts
# as 1/p rather than GRPO's 1/sqrt(p).
#
# Also launches the 300-step runs: hi_grpo's pass@1 was still climbing at step 119, so the
# pass@k collapse had not finished developing at 120 steps.
cd ${PF_ROOT}
PY=${PF_PY_ENV}/bin/python
export HF_HOME=${PF_HF_HOME}
mkdir -p runs logs

launch () {  # gpu name objective lr steps extra...
  local gpu=$1 name=$2 obj=$3 lr=$4 steps=$5; shift 5
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u train.py \
    --objective "$obj" --lr "$lr" --steps "$steps" --out_dir runs/"$name" "$@" \
    > logs/"$name".log 2>&1 &
  echo "launched $name (obj=$obj lr=$lr steps=$steps) on gpu $gpu"
}

# --- corrected MaxRL, lr bracket at the standard 120-step budget -------------------
launch 0 fix_maxrl_1e-6 maxrl 1e-6 120
launch 1 fix_maxrl_3e-6 maxrl 3e-6 120
launch 2 fix_maxrl_1e-5 maxrl 1e-5 120

# --- long runs at 3e-6: does pass@k keep falling once pass@1 saturates? ------------
launch 3 long_grpo      grpo     3e-6 300
launch 4 long_entropic  entropic 3e-6 300
launch 5 long_maxrl     maxrl    3e-6 300

# --- the other diagnosis: RAFT with a KL anchor to base ----------------------------
launch 6 fix_raft_kl    klanchor 3e-6 120 --kl_beta 0.01

wait
echo "wave 2 done"
