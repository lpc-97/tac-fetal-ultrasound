#!/bin/bash
cd "${TAC_ROOT:-.}" || exit 1
for d in output/canon/u7_*_s42_*_val output/canon/v7_w_s42_*_val; do if [ -d "$d" ] && [ ! -f "$d/result_ap.txt" ]; then rm -rf "$d" "$d.log"; echo "removed $d"; fi; done
setsid nohup env VARIANTS="u7_k17 u7_k17_min0 u7_k17_max07 u7_k17_w u7_k17_w1 v7_w" SPLIT=val SEED=42 GPU=6 bash run_canon.sh > uval_gpu6.log 2>&1 < /dev/null &
sleep 2; echo LAUNCHED6
