#!/bin/bash
# Source-only 3x3 matrix: the senior's FrozenBN source models (iter 9999) evaluated on every center's TEST set.
# Diagonal = in-domain reproduction checks; off-diagonal = the 6 cross-domain source-only lower bounds.
# usage: GPU=0 nohup bash eval_source_only.sh > eval_source_only.log 2>&1 &
export LANG=C TMPDIR=${TAC_SCRATCH:-/tmp}/tmp HOME=${TAC_SCRATCH:-/tmp}/tmp_home XDG_CACHE_HOME=${TAC_SCRATCH:-/tmp}/tmp_home/.cache
export FVCORE_CACHE=${TAC_SCRATCH:-/tmp}/tmp/fvcore_cache TORCH_HOME=${TAC_SCRATCH:-/tmp}/tmp/torch_home MPLCONFIGDIR=${TAC_SCRATCH:-/tmp}/tmp/mpl
export CUDA_VISIBLE_DEVICES=${GPU:-0}
source ${TAC_SCRATCH:-/tmp}/miniconda3/etc/profile.d/conda.sh
conda activate ${TAC_SCRATCH:-/tmp}/miniconda3/envs/ttaod || exit 1
cd "${TAC_ROOT:-.}" || exit 1
declare -A CK=([c1]=fs_c1_fz_focal [c2]=fs_c2_fz [c3]=fs_c3_fz)
echo "python=$(which python) GPU=$CUDA_VISIBLE_DEVICES"
for src in c1 c2 c3; do
  for tgt in c1 c2 c3; do
    out=output/so_${src}_to_${tgt}
    echo "===== [$(date '+%H:%M:%S')] source $src (${CK[$src]}) -> target ${tgt}_test  out=$out ====="
    python train_net.py --eval-only --num-gpus 1 --config-file checkpoints/${CK[$src]}/config.yaml \
      MODEL.WEIGHTS checkpoints/${CK[$src]}/model_0009999.pth OUTPUT_DIR "$out" \
      DATASETS.TEST "('fetus_4c_${tgt}_test',)" TEST.TTA False > "${out}.log" 2>&1
    echo "rc=$?"
    grep 'Loaded [0-9]* images' "$out/log.txt" | tail -1 | cut -c1-160
    grep 'copypaste: [0-9]' "$out/log.txt" | tail -1
  done
done
echo "SO_MATRIX_DONE"
