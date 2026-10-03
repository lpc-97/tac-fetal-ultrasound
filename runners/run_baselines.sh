#!/bin/bash
# Published TTA-OD baselines (re-implemented): IoU-Filter (iouf), AMROD (amrod), CoTTA-det (via distill trainer).
# Equal-budget protocol: 4 named configs per method swept on VAL; then TEST with the val-best config per direction.
# usage (sweep):  VARIANTS="iouf_ep_lr1e-3 ..." SPLIT=val SEED=42 GPU=3 nohup bash run_baselines.sh > bl_gpu3.log 2>&1 &
# usage (best):   METHOD=iouf BEST=output/baselines/best_val.json SPLIT=test SEED=43 GPU=3 nohup bash run_baselines.sh > ... &
export LANG=C TMPDIR=${TAC_SCRATCH:-/tmp}/tmp HOME=${TAC_SCRATCH:-/tmp}/tmp_home XDG_CACHE_HOME=${TAC_SCRATCH:-/tmp}/tmp_home/.cache
export FVCORE_CACHE=${TAC_SCRATCH:-/tmp}/tmp/fvcore_cache TORCH_HOME=${TAC_SCRATCH:-/tmp}/tmp/torch_home MPLCONFIGDIR=${TAC_SCRATCH:-/tmp}/tmp/mpl
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
SEED=${SEED:-42}; SPLIT=${SPLIT:-val}; DIRS=${DIRS:-"c1:c2 c1:c3 c2:c1 c2:c3 c3:c1 c3:c2"}
source ${TAC_SCRATCH:-/tmp}/miniconda3/etc/profile.d/conda.sh
conda activate ${TAC_SCRATCH:-/tmp}/miniconda3/envs/ttaod || exit 1
cd "${TAC_ROOT:-.}" || exit 1
declare -A V
# IoU-Filter: paper = episodic, 5 iters, lr 1e-3, all params, no RoI-cls loss
V[iouf_ep_lr1e-3]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.001"
V[iouf_ep_lr1e-4]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.0001"
V[iouf_on_lr1e-3]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.001 SEMISUPNET.IOUF.EPISODIC False SEMISUPNET.IOUF.ITERS 1"
V[iouf_on_lr1e-4]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.0001 SEMISUPNET.IOUF.EPISODIC False SEMISUPNET.IOUF.ITERS 1"
# AMROD: paper hyper-parameters + grad clipping 10 (diverges without it here); output = teacher (paper) or student
AM="SEMISUPNET.Trainer amrod SEMISUPNET.AMROD.CLIP_GRAD 10.0"
V[amrod_lr1e-3]="$AM SEMISUPNET.AMROD.LR 0.001"
V[amrod_lr1e-4]="$AM SEMISUPNET.AMROD.LR 0.0001"
V[amrod_lr1e-3_mt99]="$AM SEMISUPNET.AMROD.LR 0.001 SEMISUPNET.AMROD.MT 0.99"
V[amrod_lr1e-3_student]="$AM SEMISUPNET.AMROD.LR 0.001 SEMISUPNET.AMROD.OUTPUT student_after"
# CoTTA-det: EMA teacher (0.999) with augmentation-averaged (18-view WBF) predictions as output and pseudo-labels
# (fixed threshold 0.5), student trained on them, stochastic restoration p=0.01
CO="SEMISUPNET.Trainer distill SEMISUPNET.DISTILL.TEACHER wbf SEMISUPNET.DISTILL.TEACHER_EMA True SEMISUPNET.DISTILL.EMA_KEEP 0.999 SEMISUPNET.DISTILL.GATE_SCORE 0.5 SEMISUPNET.DISTILL.GATE_AGREE 0.0 SEMISUPNET.DISTILL.ANAT_ON False SEMISUPNET.DISTILL.RESTORE_P 0.01 SEMISUPNET.DISTILL.HYBRID_OUT True SEMISUPNET.DISTILL.SCHEDULE every"
V[cotta_all_lr1e-3]="$CO SEMISUPNET.DISTILL.PARAMS all SEMISUPNET.DISTILL.LR 0.001"
V[cotta_all_lr1e-4]="$CO SEMISUPNET.DISTILL.PARAMS all SEMISUPNET.DISTILL.LR 0.0001"
V[cotta_heads_lr1e-3]="$CO SEMISUPNET.DISTILL.PARAMS heads SEMISUPNET.DISTILL.LR 0.001"
V[cotta_heads_lr1e-4]="$CO SEMISUPNET.DISTILL.PARAMS heads SEMISUPNET.DISTILL.LR 0.0001"
# WHW (Yoo et al., CVPR 2024): parallel adapters + KL global/foreground feature alignment to source statistics
# (__STATS__ -> output/stats/whw_<src>_s<seed>.pt, collected by collect_whw_stats.sh); paper lrs 1e-3 (COCO) / 1e-4 (KITTI, SHIFT)
WH="SEMISUPNET.Trainer whw SEMISUPNET.WHW.STATS __STATS__"
V[whw_lr1e-3]="$WH SEMISUPNET.WHW.LR 0.001"
V[whw_lr1e-4]="$WH SEMISUPNET.WHW.LR 0.0001"
V[whw_skip_lr1e-3]="$WH SEMISUPNET.WHW.LR 0.001 SEMISUPNET.WHW.SKIP stat-period-ema"
V[whw_full_lr1e-4]="$WH SEMISUPNET.WHW.WHERE full SEMISUPNET.WHW.LR 0.0001"
# CD-Buffer (Song et al., CVPR 2026): subtractive (mask) + additive (adapter) buffers, L1 statistics alignment; paper lr 1e-4 Adam
CB="SEMISUPNET.Trainer cdb SEMISUPNET.CDB.STATS __CDBSTATS__"
V[cdb_paper]="$CB"
V[cdb_before]="$CB SEMISUPNET.CDB.OUTPUT before"
V[cdb_lr1e-3]="$CB SEMISUPNET.CDB.LR 0.001 SEMISUPNET.CDB.BN_LR 0.001"
V[cdb_lr1e-5]="$CB SEMISUPNET.CDB.LR 0.00001 SEMISUPNET.CDB.BN_LR 0.00001"
V[cdb_light]="$CB SEMISUPNET.CDB.LIGHT True"
# SGP / PruningTTA (Wang et al., CVPR 2025): BN-gamma adaptation with sensitivity-guided sparsity + pruning; paper lr 5e-3 Adam
SG="SEMISUPNET.Trainer sgp SEMISUPNET.SGP.STATS_WHW __STATS__ SEMISUPNET.SGP.STATS_CDB __CDBSTATS__"
V[sgp_paper]="$SG"
V[sgp_lr1e-3]="$SG SEMISUPNET.SGP.LR 0.001"
V[sgp_lr1e-4]="$SG SEMISUPNET.SGP.LR 0.0001"
V[sgp_noprune]="$SG SEMISUPNET.SGP.LR 0.001 SEMISUPNET.SGP.THR 0.0 SEMISUPNET.SGP.LAMBDA 0.0"
V[sgp_abs_lr1e-3]="$SG SEMISUPNET.SGP.LR 0.001 SEMISUPNET.SGP.THR_MODE abs"
# O-SFDA data acquisition (Shi et al., ECCV-W 2024): mean teacher (conf 0.9, EMA 0.996, SGD 1e-3) on key frames selected by AUF/ARC clustering
OS="SEMISUPNET.Trainer osfda"
V[osfda_paper]="$OS"
V[osfda_lr1e-4]="$OS SEMISUPNET.OSFDA.LR 0.0001"
V[osfda_warm200]="$OS SEMISUPNET.OSFDA.WARMUP_LABELS 200"
V[osfda_allframes]="$OS SEMISUPNET.OSFDA.GAMMA 1.01"
# VLOD-TTA objective (Belal et al., ECCV 2026): IoU-weighted entropy on adapters, 1 step / image, episodic
VL="SEMISUPNET.Trainer vlod"
V[vlod_lr1e-3]="$VL SEMISUPNET.VLOD.LR 0.001"
V[vlod_lr1e-4]="$VL SEMISUPNET.VLOD.LR 0.0001"
V[vlod_lr1e-2]="$VL SEMISUPNET.VLOD.LR 0.01"
V[vlod_online_lr1e-4]="$VL SEMISUPNET.VLOD.LR 0.0001 SEMISUPNET.VLOD.EPISODIC False"
# BufferTTA (Kim et al., arXiv 2510.21271; CD-Buffer's additive-only baseline): buffer layers + entropy minimisation, Adam 1e-4
BT="SEMISUPNET.Trainer buftta"
V[buftta_paper]="$BT"
V[buftta_before]="$BT SEMISUPNET.BUFTTA.OUTPUT before"
V[buftta_lr1e-3]="$BT SEMISUPNET.BUFTTA.LR 0.001"
V[buftta_light]="$BT SEMISUPNET.BUFTTA.LIGHT True"
mkdir -p output/baselines
echo "python=$(which python) GPU=$CUDA_VISIBLE_DEVICES SEED=$SEED SPLIT=$SPLIT DIRS=[$DIRS] VARIANTS=[${VARIANTS:-best:$METHOD}] start=$(date '+%F %T')"
# detector: DET=fs (Faster R-CNN, default) | DET=rt (RetinaNet models rt_<c>_fz_s<seed>/model_0019999.pth, prior anat_prior_rt_<c>.json,
# outputs output/baselines/rt_<v>_..., best_val file output/baselines/best_val_rt.json)
DET=${DET:-fs}; if [ "$DET" = rt ]; then CKPT=model_0019999.pth; PT=rt_; OP=rt_; else CKPT=model_0009999.pth; PT=; OP=; fi
# dataset: DSN=4c (Heart-4cc, domains c1/c2/c3, default) | abdomen | spine (domains ge/ph/sa); DP = name prefix for ckpt/prior/outputs
DSN=${DSN:-4c}; if [ "$DSN" = 4c ]; then DP=; else DP=${DSN}_; fi
# RetinaNet-compatible variants (no RoI head: keep the classification loss in IoU-Filter; CoTTA-det adapts all params)
V[rt_iouf_ep_lr1e-3]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.001 SEMISUPNET.IOUF.NO_ROI_CLS False"
V[rt_iouf_ep_lr1e-4]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.0001 SEMISUPNET.IOUF.NO_ROI_CLS False"
V[rt_iouf_on_lr1e-3]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.001 SEMISUPNET.IOUF.NO_ROI_CLS False SEMISUPNET.IOUF.EPISODIC False SEMISUPNET.IOUF.ITERS 1"
V[rt_iouf_ep_lr1e-5]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.00001 SEMISUPNET.IOUF.NO_ROI_CLS False"
V[rt_iouf_on_lr1e-4]="SEMISUPNET.Trainer iouf SEMISUPNET.IOUF.LR 0.0001 SEMISUPNET.IOUF.NO_ROI_CLS False SEMISUPNET.IOUF.EPISODIC False SEMISUPNET.IOUF.ITERS 1"
V[rt_cotta_all_lr1e-3]="$CO SEMISUPNET.DISTILL.PARAMS all SEMISUPNET.DISTILL.LR 0.001"
V[rt_cotta_all_lr1e-4]="$CO SEMISUPNET.DISTILL.PARAMS all SEMISUPNET.DISTILL.LR 0.0001"
V[rt_cotta_all_lr1e-5]="$CO SEMISUPNET.DISTILL.PARAMS all SEMISUPNET.DISTILL.LR 0.00001"
V[rt_cotta_all_lr1e-3_ema99]="$CO SEMISUPNET.DISTILL.PARAMS all SEMISUPNET.DISTILL.LR 0.001 SEMISUPNET.DISTILL.EMA_KEEP 0.99"
for d in $DIRS; do
  src=${d%%:*}; tgt=${d##*:}; ck=output/${DET}_${DP}${src}_fz_s${SEED}
  if [ -n "$BEST" ]; then
    vlist=$(python -c "import json; print(json.load(open('$BEST'))['${METHOD}']['${src}_to_${tgt}'])")
  else
    vlist=$VARIANTS
  fi
  for v in $vlist; do
    [ -v V[$v] ] || { echo "UNKNOWN VARIANT $v"; continue; }
    out=output/baselines/${DP}${v}_s${SEED}_${src}_to_${tgt}_${SPLIT}
    [ -f "$out/result_ap.txt" ] && { echo "SKIP $out"; continue; }
    echo "===== [$(date '+%H:%M:%S')] $v  $src -> ${tgt}_${SPLIT}  out=$out ====="
    args=${V[$v]//__STATS__/output/stats/whw_${DP}${src}_s${SEED}.pt}; args=${args//__CDBSTATS__/output/stats/cdb_${DP}${src}_s${SEED}.pt}
    python train_net.py --eval-only --num-gpus 1 --config-file "$ck/config.yaml" MODEL.WEIGHTS "$ck/$CKPT" \
      OUTPUT_DIR "$out" DATASETS.TEST "('fetus_${DSN}_${tgt}_${SPLIT}',)" SEED 42 TEST.TTA False \
      SEMISUPNET.DISTILL.ANAT_PRIOR output/anat_prior_${PT}${DP}${src}.json $args > "${out}.log" 2>&1
    echo "rc=$?  [$(date '+%H:%M:%S')]"
    grep -E 'Traceback|Error' "${out}.log" | tail -1 | cut -c1-160
    [ -f "$out/result_ap.txt" ] && tail -1 "$out/result_ap.txt" | cut -c1-70
  done
done
echo "BASELINES_DONE GPU=$GPU end=$(date '+%F %T')"
