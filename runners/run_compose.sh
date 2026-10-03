#!/bin/bash
# Composition experiments: parameter-adapting TTA baselines on TAC-canonicalised inputs.
# Uses the val-chosen baseline configs (output/baselines/best_val.json) + CANON_FACTORS from the tacr2 run of the
# same split/seed/direction. Output dirs: output/compose/<method>+tac_s<seed>_<dir>_<split>
# usage: METHODS="amrod cotta iouf" SPLIT=test SEED=42 GPU=5 nohup bash run_compose.sh > compose_gpu5.log 2>&1 &
export LANG=C TMPDIR=${TAC_SCRATCH:-/tmp}/tmp HOME=${TAC_SCRATCH:-/tmp}/tmp_home XDG_CACHE_HOME=${TAC_SCRATCH:-/tmp}/tmp_home/.cache
export FVCORE_CACHE=${TAC_SCRATCH:-/tmp}/tmp/fvcore_cache TORCH_HOME=${TAC_SCRATCH:-/tmp}/tmp/torch_home MPLCONFIGDIR=${TAC_SCRATCH:-/tmp}/tmp/mpl
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
SEED=${SEED:-42}; SPLIT=${SPLIT:-val}; DIRS=${DIRS:-"c1:c2 c1:c3 c2:c1 c2:c3 c3:c1 c3:c2"}; METHODS=${METHODS:-"amrod cotta iouf"}
source ${TAC_SCRATCH:-/tmp}/miniconda3/etc/profile.d/conda.sh
conda activate ${TAC_SCRATCH:-/tmp}/miniconda3/envs/ttaod || exit 1
cd "${TAC_ROOT:-.}" || exit 1
source <(grep -E '^(V\[|AM=|CO=|declare -A V)' run_baselines.sh)   # reuse the variant definitions
mkdir -p output/compose
echo "GPU=$CUDA_VISIBLE_DEVICES SEED=$SEED SPLIT=$SPLIT DIRS=[$DIRS] METHODS=[$METHODS] start=$(date '+%F %T')"
for m in $METHODS; do
  case $m in amrod) KEY=SEMISUPNET.AMROD.CANON_FACTORS;; iouf) KEY=SEMISUPNET.IOUF.CANON_FACTORS;; cotta) KEY=SEMISUPNET.DISTILL.CANON_FACTORS;;
    src) KEY=SEMISUPNET.IOUF.CANON_FACTORS; V[src_canon1v]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.ITERS 0";;   # control: source model on canonicalised input, 1 view, no update
    *) echo "unknown $m"; continue;; esac
  for d in $DIRS; do
    src=${d%%:*}; tgt=${d##*:}; ck=output/fs_${src}_fz_s${SEED}
    if [ "$m" = src ]; then v=src_canon1v; else v=$(python -c "import json; print(json.load(open('output/baselines/best_val.json'))['$m']['${src}_to_${tgt}'])"); fi
    fac=output/canon/tacr2_s${SEED}_${src}_to_${tgt}_${SPLIT}/canon_factors.json
    [ -f "$fac" ] || { echo "MISSING factors $fac"; continue; }
    out=output/compose/${m}+tac_s${SEED}_${src}_to_${tgt}_${SPLIT}
    [ -f "$out/result_ap.txt" ] && { echo "SKIP $out"; continue; }
    echo "===== [$(date '+%H:%M:%S')] $m+tac ($v)  $src -> ${tgt}_${SPLIT}  out=$out ====="
    python train_net.py --eval-only --num-gpus 1 --config-file "$ck/config.yaml" MODEL.WEIGHTS "$ck/model_0009999.pth" \
      OUTPUT_DIR "$out" DATASETS.TEST "('fetus_4c_${tgt}_${SPLIT}',)" SEED 42 TEST.TTA False \
      SEMISUPNET.DISTILL.ANAT_PRIOR output/anat_prior_${src}.json ${V[$v]} $KEY "$fac" > "${out}.log" 2>&1
    echo "rc=$?  [$(date '+%H:%M:%S')]"
    grep -E 'Traceback|Error' "${out}.log" | tail -1 | cut -c1-160
    [ -f "$out/result_ap.txt" ] && tail -1 "$out/result_ap.txt" | cut -c1-70
  done
done
echo "COMPOSE_DONE GPU=$GPU end=$(date '+%F %T')"
