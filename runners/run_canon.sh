#!/bin/bash
# TAC (canonicalisation) variants x directions on a split.
# usage: VARIANTS="tac tac_aspect" DIRS="c1:c2 ..." SPLIT=val SEED=42 GPU=6 nohup bash run_canon.sh > canon_gpu6.log 2>&1 &
export LANG=C TMPDIR=${TAC_SCRATCH:-/tmp}/tmp HOME=${TAC_SCRATCH:-/tmp}/tmp_home XDG_CACHE_HOME=${TAC_SCRATCH:-/tmp}/tmp_home/.cache
export FVCORE_CACHE=${TAC_SCRATCH:-/tmp}/tmp/fvcore_cache TORCH_HOME=${TAC_SCRATCH:-/tmp}/tmp/torch_home MPLCONFIGDIR=${TAC_SCRATCH:-/tmp}/tmp/mpl
export CUDA_VISIBLE_DEVICES=${GPU:?set GPU}
SEED=${SEED:-42}; SPLIT=${SPLIT:-val}; DIRS=${DIRS:-"c1:c2 c1:c3 c2:c1 c2:c3 c3:c1 c3:c2"}; VARIANTS=${VARIANTS:?set VARIANTS}
# detector: DET=fs (Faster R-CNN, default: fs_<c>_fz_s<seed>/model_0009999.pth, anat_prior_<c>.json, out canon/<v>_...)
#           DET=rt (RetinaNet: rt_<c>_fz_s<seed>/model_0019999.pth, anat_prior_rt_<c>.json, out canon/rt_<v>_...)
DET=${DET:-fs}; if [ "$DET" = rt ]; then CKPT=model_0019999.pth; PT=rt_; OP=rt_; else CKPT=model_0009999.pth; PT=; OP=; fi
# dataset: DSN=4c (Heart-4cc, domains c1/c2/c3, default) | abdomen | spine (domains ge/ph/sa); DP = name prefix for ckpt/prior/outputs
DSN=${DSN:-4c}; if [ "$DSN" = 4c ]; then DP=; else DP=${DSN}_; fi
source ${TAC_SCRATCH:-/tmp}/miniconda3/etc/profile.d/conda.sh
conda activate ${TAC_SCRATCH:-/tmp}/miniconda3/envs/ttaod || exit 1
cd "${TAC_ROOT:-.}" || exit 1
declare -A V
T="SEMISUPNET.Trainer canon"
V[tac]="$T"
V[tac_flip0]="$T SEMISUPNET.CANON.FLIP0 True"
V[tac_aspect]="$T SEMISUPNET.CANON.ASPECT True"
V[tac_aspect_flip0]="$T SEMISUPNET.CANON.ASPECT True SEMISUPNET.CANON.FLIP0 True"
V[tac_rounds2]="$T SEMISUPNET.CANON.ROUNDS 2"
V[tacr2]="$T SEMISUPNET.CANON.ROUNDS 2"        # alias used by the composition experiments (factor dump is always written)
# ---- Information Fusion ablations: fusion strategy on the same 3 views, and number of views
V[tac_fnms]="$T SEMISUPNET.CANON.FUSION nms"
V[tac_fheat]="$T SEMISUPNET.CANON.FUSION heatmap"
V[tac_flast]="$T SEMISUPNET.CANON.FUSION last"                                   # canonical view only, no fusion (1 view used)
V[tac_v2]="$T SEMISUPNET.CANON.FLIP False"                                        # pass1 + canonical (2 views)
V[tac_v7]="$T SEMISUPNET.CANON.EXTRA_SCALES (0.85,1.15)"                          # + 2 extra scales x flip = 7 views
V[tac_v11]="$T SEMISUPNET.CANON.EXTRA_SCALES (0.8,0.9,1.1,1.25)"                  # 11 views
V[tac_v7_noar]="$T SEMISUPNET.CANON.EXTRA_SCALES (0.85,1.15) SEMISUPNET.CANON.ANAT_RESCORE False"
# ---- hyper-parameter sensitivity (val, seed 42): one knob at a time around the defaults
V[hp_su0.3]="$T SEMISUPNET.CANON.SIGMA_U 0.3"
V[hp_su0.5]="$T SEMISUPNET.CANON.SIGMA_U 0.5"
V[hp_su2]="$T SEMISUPNET.CANON.SIGMA_U 2.0"
V[hp_su100]="$T SEMISUPNET.CANON.SIGMA_U 100.0"
V[hp_od0]="$T SEMISUPNET.CANON.OUTLIER_DENSITY 0.0"
V[hp_od0.05]="$T SEMISUPNET.CANON.OUTLIER_DENSITY 0.05"
V[hp_od0.5]="$T SEMISUPNET.CANON.OUTLIER_DENSITY 0.5"
V[hp_od1]="$T SEMISUPNET.CANON.OUTLIER_DENSITY 1.0"
V[hp_ms0.1]="$T SEMISUPNET.CANON.MIN_SCORE 0.1"
V[hp_ms0.5]="$T SEMISUPNET.CANON.MIN_SCORE 0.5"
V[hp_ms0.7]="$T SEMISUPNET.CANON.MIN_SCORE 0.7"
V[hp_so0.05]="$T SEMISUPNET.CANON.SIGMA_OBS 0.05"
V[hp_so0.2]="$T SEMISUPNET.CANON.SIGMA_OBS 0.2"
V[hp_so0.4]="$T SEMISUPNET.CANON.SIGMA_OBS 0.4"
V[hp_wbf0.45]="$T SEMISUPNET.CANON.WBF_IOU 0.45"
V[hp_wbf0.65]="$T SEMISUPNET.CANON.WBF_IOU 0.65"
V[hp_wbf0.75]="$T SEMISUPNET.CANON.WBF_IOU 0.75"
V[hp_lam0.25]="$T SEMISUPNET.CANON.ANAT_LAM 0.25"
V[hp_lam1]="$T SEMISUPNET.CANON.ANAT_LAM 1.0"
V[hp_lam2]="$T SEMISUPNET.CANON.ANAT_LAM 2.0"
V[hp_gam0]="$T SEMISUPNET.CANON.ANAT_GAMMA 0.0"
V[hp_gam0.6]="$T SEMISUPNET.CANON.ANAT_GAMMA 0.6"
V[hp_gam1]="$T SEMISUPNET.CANON.ANAT_GAMMA 1.0"
V[hp_irls1]="$T SEMISUPNET.CANON.IRLS_ITERS 1"
V[hp_irls6]="$T SEMISUPNET.CANON.IRLS_ITERS 6"
V[tac_noshrink]="$T SEMISUPNET.CANON.SIGMA_U 100.0"
V[tac_shrink03]="$T SEMISUPNET.CANON.SIGMA_U 0.3"
V[tac_nooutlier]="$T SEMISUPNET.CANON.OUTLIER_DENSITY 0.0"
V[tac_outlier1]="$T SEMISUPNET.CANON.OUTLIER_DENSITY 1.0"
V[tac_noar]="$T SEMISUPNET.CANON.ANAT_RESCORE False"
V[tac_nopass1]="$T SEMISUPNET.CANON.FUSE_PASS1 False"
V[tac_nocalib]="$T SEMISUPNET.CANON.S0_CALIB False"
V[tac_minscore05]="$T SEMISUPNET.CANON.MIN_SCORE 0.5"
V[tac_sobs02]="$T SEMISUPNET.CANON.SIGMA_OBS 0.2"
V[tac_full]="$T SEMISUPNET.CANON.FLIP0 True SEMISUPNET.CANON.ROUNDS 2"
V[tac_full_aspect]="$T SEMISUPNET.CANON.FLIP0 True SEMISUPNET.CANON.ROUNDS 2 SEMISUPNET.CANON.ASPECT True"
# ---- reviewer experiments (2026-09-21): scale-estimator ablation (same 3 views / fusion, only the estimator differs)
V[est_median]="$T SEMISUPNET.CANON.ESTIMATOR median"        # median of per-candidate votes (heuristic)
V[est_confmean]="$T SEMISUPNET.CANON.ESTIMATOR confmean"    # confidence-weighted mean of the votes (heuristic)
V[est_noclass]="$T SEMISUPNET.CANON.ESTIMATOR noclass"      # IRLS with a class-agnostic (pooled) size prior
# ---- externally supplied factors; __DIR__ -> <src>_to_<tgt>, __SEED__ -> seed (substituted in the loop)
FF="SEMISUPNET.CANON.FACTORS_FILE"
V[fac_oracle]="$T $FF output/analysis/oracle_factors___DP____DIR___test.json"                    # TAC-3 pipeline with the ORACLE (GT-derived) factor
V[fac_oracle_1v]="$T $FF output/analysis/oracle_factors___DP____DIR___test.json SEMISUPNET.CANON.FUSION last SEMISUPNET.CANON.FLIP False"   # single view at the oracle scale
V[fac_global]="$T $FF output/analysis/global_factors___DP____DIR___s__SEED__.json"                # stream-level constant factor (median of TAC factors)
V[fac_global_1v]="$T $FF output/analysis/global_factors___DP____DIR___s__SEED__.json SEMISUPNET.CANON.FUSION last SEMISUPNET.CANON.FLIP False"
V[fac_valscale]="$T $FF output/analysis/valscale_factors___DP____DIR__.json"                      # val-selected fixed scale per direction
V[fac_valscale_1v]="$T $FF output/analysis/valscale_factors___DP____DIR__.json SEMISUPNET.CANON.FUSION last SEMISUPNET.CANON.FLIP False"
V[tac_1v]="$T SEMISUPNET.CANON.FUSION last SEMISUPNET.CANON.FLIP False"                            # single view at the TAC scale (2 passes, 1 output view)
V[p1dump]="$T SEMISUPNET.CANON.FUSION last SEMISUPNET.CANON.FLIP False SEMISUPNET.CANON.DUMP_PASS1 True"   # first-pass candidate dump for the posterior-exactness analysis
# ---- uncertainty-aware TAC (patch_canon4.py, 2026-09-23): bracket views at u +- clip(K*s, MIN, MAX) from the posterior std s of u
UB="$T SEMISUPNET.CANON.ADAPT bracket SEMISUPNET.CANON.ADAPT_MIN 0.15 SEMISUPNET.CANON.ADAPT_MAX 0.5"
V[u7_k10]="$UB SEMISUPNET.CANON.ADAPT_K 1.0"
V[u7_k17]="$UB SEMISUPNET.CANON.ADAPT_K 1.732"
V[u7_k25]="$UB SEMISUPNET.CANON.ADAPT_K 2.5"
V[u7_k17_min0]="$T SEMISUPNET.CANON.ADAPT bracket SEMISUPNET.CANON.ADAPT_MIN 0.0 SEMISUPNET.CANON.ADAPT_MAX 0.5 SEMISUPNET.CANON.ADAPT_K 1.732"
V[u7_k17_max07]="$T SEMISUPNET.CANON.ADAPT bracket SEMISUPNET.CANON.ADAPT_MIN 0.15 SEMISUPNET.CANON.ADAPT_MAX 0.7 SEMISUPNET.CANON.ADAPT_K 1.732"
V[u7_k17_w]="$UB SEMISUPNET.CANON.ADAPT_K 1.732 SEMISUPNET.CANON.POST_WEIGHTS True SEMISUPNET.CANON.PASS1_WEIGHT 0.25"      # + posterior-weighted WBF
V[u7_k17_w1]="$UB SEMISUPNET.CANON.ADAPT_K 1.732 SEMISUPNET.CANON.POST_WEIGHTS True SEMISUPNET.CANON.PASS1_WEIGHT 1.0"
V[v7_w]="$T SEMISUPNET.CANON.EXTRA_SCALES (0.85,1.15) SEMISUPNET.CANON.POST_WEIGHTS True SEMISUPNET.CANON.PASS1_WEIGHT 0.25"    # fixed bracket + posterior weights only
UC="$T SEMISUPNET.CANON.ADAPT count SEMISUPNET.CANON.ADAPT_MIN 0.15 SEMISUPNET.CANON.ADAPT_MAX 0.5 SEMISUPNET.CANON.ADAPT_K 1.732"
V[ua_t06]="$UC SEMISUPNET.CANON.ADAPT_TAU 0.06"                                                  # count mode: 3 views if s <= TAU, else 7
V[ua_t08]="$UC SEMISUPNET.CANON.ADAPT_TAU 0.08"
V[ua_t10]="$UC SEMISUPNET.CANON.ADAPT_TAU 0.10"
V[ua_t12]="$UC SEMISUPNET.CANON.ADAPT_TAU 0.12"
V[ua_t15]="$UC SEMISUPNET.CANON.ADAPT_TAU 0.15"
V[ua_t20]="$UC SEMISUPNET.CANON.ADAPT_TAU 0.20"
V[ua_t10_w]="$UC SEMISUPNET.CANON.ADAPT_TAU 0.10 SEMISUPNET.CANON.POST_WEIGHTS True SEMISUPNET.CANON.PASS1_WEIGHT 0.25"
mkdir -p output/canon
echo "python=$(which python) GPU=$CUDA_VISIBLE_DEVICES SEED=$SEED SPLIT=$SPLIT DIRS=[$DIRS] VARIANTS=[$VARIANTS] start=$(date '+%F %T')"
for v in $VARIANTS; do
  [ -v V[$v] ] || { echo "UNKNOWN VARIANT $v"; continue; }
  for d in $DIRS; do
    src=${d%%:*}; tgt=${d##*:}; ck=output/${DET}_${DP}${src}_fz_s${SEED}
    out=output/canon/${OP}${DP}${v}_s${SEED}_${src}_to_${tgt}_${SPLIT}
    [ -f "$out/result_ap.txt" ] && { echo "SKIP $out"; continue; }
    echo "===== [$(date '+%H:%M:%S')] $v  $src -> ${tgt}_${SPLIT}  out=$out ====="
    opts=${V[$v]}; opts=${opts//__DIR__/${src}_to_${tgt}}; opts=${opts//__SEED__/${SEED}}; opts=${opts//__DP__/${DP}}
    python train_net.py --eval-only --num-gpus 1 --config-file "$ck/config.yaml" MODEL.WEIGHTS "$ck/$CKPT" \
      OUTPUT_DIR "$out" DATASETS.TEST "('fetus_${DSN}_${tgt}_${SPLIT}',)" SEED 42 TEST.TTA False \
      SEMISUPNET.CANON.ANAT_PRIOR output/anat_prior_${PT}${DP}${src}.json ${opts} > "${out}.log" 2>&1
    echo "rc=$?  [$(date '+%H:%M:%S')]"
    grep -E 'Traceback|Error' "${out}.log" | tail -1 | cut -c1-160
    grep -o '\[CANON\] summary.*' "$out/log.txt" 2>/dev/null | tail -1 | cut -c1-200
    [ -f "$out/result_ap.txt" ] && tail -1 "$out/result_ap.txt" | cut -c1-70
  done
done
echo "CANON_DONE GPU=$GPU end=$(date '+%F %T')"
