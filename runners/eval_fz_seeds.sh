#!/bin/bash
# Source-only 3x3 for our FrozenBN models (output/fs_<c>_fz_s<seed>, iter 9999) on every TEST set, plus
# heatmap-fusion TTA (TEST.TTA True) on the 6 cross-domain directions for the same models.
# usage: SEEDS="42" MODE=source|heatmap GPU=0 nohup bash eval_fz_seeds.sh > eval_fz_source_42.log 2>&1 &
export LANG=C TMPDIR=${TAC_SCRATCH:-/tmp}/tmp HOME=${TAC_SCRATCH:-/tmp}/tmp_home XDG_CACHE_HOME=${TAC_SCRATCH:-/tmp}/tmp_home/.cache
export FVCORE_CACHE=${TAC_SCRATCH:-/tmp}/tmp/fvcore_cache TORCH_HOME=${TAC_SCRATCH:-/tmp}/tmp/torch_home MPLCONFIGDIR=${TAC_SCRATCH:-/tmp}/tmp/mpl
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
SEEDS=${SEEDS:-"42 43 44"}; MODE=${MODE:?set MODE=source|heatmap}
# dataset: DSN=4c (Heart-4cc, domains c1/c2/c3, default) | abdomen | spine (domains ge/ph/sa); DP = name prefix for ckpt/prior/outputs
DSN=${DSN:-4c}; if [ "$DSN" = 4c ]; then DP=; DOMS="c1 c2 c3"; else DP=${DSN}_; DOMS="ge ph sa"; fi; SRCS=${SRCS:-$DOMS}
source ${TAC_SCRATCH:-/tmp}/miniconda3/etc/profile.d/conda.sh
conda activate ${TAC_SCRATCH:-/tmp}/miniconda3/envs/ttaod || exit 1
cd "${TAC_ROOT:-.}" || exit 1
echo "python=$(which python) GPU=$CUDA_VISIBLE_DEVICES MODE=$MODE SEEDS=[$SEEDS] start=$(date '+%F %T')"
for s in $SEEDS; do
  for src in $SRCS; do
    ck=output/fs_${DP}${src}_fz_s${s}
    [ -f "$ck/model_0009999.pth" ] || { echo "MISSING $ck"; continue; }
    for tgt in $DOMS; do
      if [ "$MODE" = "heatmap" ]; then
        [ "$src" = "$tgt" ] && continue
        out=output/tta_heatmap_${DP}s${s}_${src}_to_${tgt}; tta=True
      else
        out=output/so_fz_${DP}s${s}_${src}_to_${tgt}; tta=False
      fi
      if [ -f "$out/result_ap.txt" ]; then echo "SKIP $out (done)"; continue; fi
      echo "===== [$(date '+%H:%M:%S')] seed $s  $MODE  fz $src -> ${tgt}_test  out=$out ====="
      python train_net.py --eval-only --num-gpus 1 --config-file "$ck/config.yaml" \
        MODEL.WEIGHTS "$ck/model_0009999.pth" OUTPUT_DIR "$out" \
        DATASETS.TEST "('fetus_${DSN}_${tgt}_test',)" TEST.TTA $tta SEMISUPNET.Trainer baseline SEED 42 > "${out}.log" 2>&1
      echo "rc=$?  [$(date '+%H:%M:%S')]"
      grep -E 'Traceback|Error' "${out}.log" | tail -1 | cut -c1-120
      [ -f "$out/result_ap.txt" ] && tail -1 "$out/result_ap.txt" | cut -c1-80
    done
  done
done
echo "EVAL_FZ_${MODE}_DONE end=$(date '+%F %T')"
