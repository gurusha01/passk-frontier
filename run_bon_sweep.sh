#!/bin/bash
# Independent lr sweep for BonBon / BOND / J-BOND. Every run gets the same 15,360
# generated-sequence budget as the earlier arms, so the frontier stays comparable.
# Usage: HOST=host-b bash run_bon_sweep.sh
cd ${PF_ROOT}
PY=${PF_PY_ENV}/bin/python
export HF_HOME=${PF_HF_HOME}
mkdir -p runs logs

go () {  # gpu name method lr extra...
  local gpu=$1 name=$2 m=$3 lr=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u train_bon.py \
    --method "$m" --lr "$lr" --out_dir runs/"$name" "$@" \
    > logs/"$name".log 2>&1 &
  echo "launched $name (method=$m lr=$lr $*) on gpu $gpu"
}

if [ "$HOST" = "host-b" ]; then
  go 0 bonbon_1e-7 bonbon 1e-7
  go 1 bonbon_3e-7 bonbon 3e-7
  go 2 bonbon_1e-6 bonbon 1e-6
  go 3 bonbon_3e-6 bonbon 3e-6
  go 4 bond_1e-7   bond   1e-7
  go 5 bond_3e-7   bond   3e-7
  go 6 bond_1e-6   bond   1e-6
  go 7 bond_3e-6   bond   3e-6
else
  go 2 jbond_1e-7  jbond  1e-7
  go 3 jbond_3e-7  jbond  3e-7
  go 4 jbond_1e-6  jbond  1e-6
  go 5 jbond_3e-6  jbond  3e-6
  # ablations at the paper's own lr: alpha balance, and length normalisation
  go 6 bonbon_a0.5_3e-6 bonbon 3e-6 --alpha 0.5
  go 7 bond_tok_3e-6    bond   3e-6 --len_norm token
fi
wait
echo "bon sweep done on ${HOST:-host-c}"
