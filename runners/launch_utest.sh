#!/bin/bash
# Test-phase queue for the uncertainty-aware TAC variants: VARIANTS x seeds {42,43,44} x six heart directions on the test split,
# with PASSES retry passes (run_canon.sh skips finished runs), detached with setsid so that the ssh channel closes.
# usage: VARIANTS="u7_k17 ua_t10" GPU=6 SPLIT=test PASSES=3 bash launch_utest.sh
cd "${TAC_ROOT:-.}" || exit 1
VARIANTS=${VARIANTS:?set VARIANTS}; GPU=${GPU:?set GPU}; SPLIT=${SPLIT:-test}; PASSES=${PASSES:-3}; SEEDS=${SEEDS:-"42 43 44"}
LOG=utest_gpu${GPU}.log
cat > .utest_gpu${GPU}.sh <<EOF
#!/bin/bash
cd "${TAC_ROOT:-.}" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
for pass in \$(seq 1 ${PASSES}); do
  echo "##### pass \$pass \$(date '+%F %T')"
  for s in ${SEEDS}; do
    VARIANTS="${VARIANTS}" SPLIT=${SPLIT} SEED=\$s GPU=${GPU} bash run_canon.sh
  done
  n=\$(for v in ${VARIANTS}; do for s in ${SEEDS}; do for d in c1_to_c2 c1_to_c3 c2_to_c1 c2_to_c3 c3_to_c1 c3_to_c2; do [ -f output/canon/\${v}_s\${s}_\${d}_${SPLIT}/result_ap.txt ] || echo x; done; done; done | wc -l)
  echo "##### pass \$pass done, missing=\$n"
  [ "\$n" = 0 ] && break
  sleep 600
done
echo "UTEST_DONE GPU=${GPU} missing=\$n \$(date '+%F %T')"
EOF
setsid nohup bash .utest_gpu${GPU}.sh > ${LOG} 2>&1 < /dev/null &
sleep 2; echo "LAUNCHED GPU=${GPU} VARIANTS=[${VARIANTS}] SPLIT=${SPLIT} log=${LOG}"
