#!/bin/bash
# (a) powmaxrl: A=(r-p)/p^lam, alpha-fair family, lam=1 is MaxRL
# (b) geopassk: geometric weights over k -> A=(r-p)*c/(c+g*p)^2, bounded at p=0
# (c) difkl:    beta(p)=p^kappa mix of reverse-KL (sharpen) and forward-KL (cover)
# lr 1e-5 is where corrected MaxRL peaked; these are all reweightings of the same signal.
cd ${PF_ROOT}
PY=${PF_PY_ENV}/bin/python
export HF_HOME=${PF_HF_HOME} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
go () { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PY -u train.py --steps 120 --out_dir runs/"$name" \
    "$@" > logs/"$name".log 2>&1 < /dev/null &
  disown; echo "launched $name gpu $gpu: $*"; sleep 2; }
if [ "$HOST" = "host-a" ]; then
  go 0 pow15_1e-5  --objective powmaxrl --pow_lambda 1.5 --lr 1e-5
  go 1 pow20_1e-5  --objective powmaxrl --pow_lambda 2.0 --lr 1e-5
  go 2 geo8_1e-5   --objective geopassk --k_eff 8  --lr 1e-5
  go 6 geo32_1e-5  --objective geopassk --k_eff 32 --lr 1e-5
  go 7 difkl1_1e-5 --objective difkl --kappa 1.0 --lr 1e-5
else
  go 4 difkl2_1e-5 --objective difkl --kappa 2.0 --lr 1e-5
  go 5 pow15_3e-6  --objective powmaxrl --pow_lambda 1.5 --lr 3e-6
  go 6 geo8_3e-6   --objective geopassk --k_eff 8 --lr 3e-6
fi
wait
