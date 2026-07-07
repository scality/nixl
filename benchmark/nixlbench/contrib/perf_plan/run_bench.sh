#!/usr/bin/env bash
# Run one nixlbench OBJ/scality_ai_connector (RDMA) cell and append its output
# to a combined log. Also samples whole-node CPU with mpstat during the run and
# emits a NIXLBENCH_CPU line (authoritative %usr/%sys for the client node).
#
# Edit the CONFIG block (or export the vars), then call once per test cell:
#   ./run_bench.sh OP  SIZE_MiB  THREADS  CLIENTS(GPUs)  [SEG=VRAM|DRAM] [NUM_ITER] [WARMUP]
#   e.g. ./run_bench.sh WRITE 1 8 2 VRAM
#
# All runs append to $LOG. Parse it later with fill_sheet.py.
set -u

# ---------------- CONFIG (edit me) ----------------
BIN=${BIN:-/usr/local/nixlbench/bin/nixlbench}
ENDPOINT=${ENDPOINT:-http://localhost:10000/bpchord}   # connector base URL (also aws --endpoint-url for READ seeding)
BUCKET=${BUCKET:-bpchord}                               # S3 bucket (needed for READ seeding)
ACCESS_KEY=${ACCESS_KEY:-${AWS_ACCESS_KEY_ID:-}}        # needed for READ seeding
SECRET_KEY=${SECRET_KEY:-${AWS_SECRET_ACCESS_KEY:-}}
REGION=${REGION:-us-east-1}
RDMA_NICS=${RDMA_NICS:-}                                # optional, e.g. 10.10.40.208,10.10.48.208,...
DC_KEY=${DC_KEY:-}                                      # optional hex, e.g. 0xffeeddcc
LOG=${LOG:-$PWD/nixlbench_results.log}
# --------------------------------------------------

export NIXLBENCH_EMIT_RESULT=1   # enable the machine-readable NIXLBENCH_RESULT line

OP=$1; SIZE_MIB=$2; THREADS=$3; CLIENTS=$4; SEG=${5:-VRAM}
NUM_ITER=${6:-64}; WARMUP=${7:-16}

BLOCK=$(( SIZE_MIB * 1024 * 1024 ))
# total_buffer_size must be >= block * max(threads, clients); give 2x headroom, min 1 GiB.
maxpar=$(( THREADS > CLIENTS ? THREADS : CLIENTS ))
TOTBUF=$(( BLOCK * maxpar * 2 ))
[ $TOTBUF -lt $((1024*1024*1024)) ] && TOTBUF=$((1024*1024*1024))

MODE=SG
[ "$CLIENTS" -gt 1 ] && MODE=MG

cmd=( "$BIN"
  --backend=OBJ
  --obj_accelerated_enable
  --obj_accelerated_type=scality_ai_connector
  --obj_endpoint_override="$ENDPOINT"
  --obj_bucket_name="$BUCKET"
  --obj_region="$REGION"
  --initiator_seg_type="$SEG"
  --op_type="$OP"
  --mode="$MODE"
  --num_initiator_dev="$CLIENTS"
  --num_threads="$THREADS"
  --start_block_size="$BLOCK"
  --max_block_size="$BLOCK"
  --total_buffer_size="$TOTBUF"
  --warmup_iter="$WARMUP"
  --num_iter="$NUM_ITER"
)
[ -n "$ACCESS_KEY" ] && cmd+=( --obj_access_key="$ACCESS_KEY" )
[ -n "$SECRET_KEY" ] && cmd+=( --obj_secret_key="$SECRET_KEY" )
[ -n "$RDMA_NICS" ] && cmd+=( --obj_rdma_nics="$RDMA_NICS" )
[ -n "$DC_KEY" ]    && cmd+=( --obj_rdma_dc_key="$DC_KEY" )
[ "$OP" = "WRITE" ] && cmd+=( --obj_unique_keys )

{
  echo "======== CELL op=$OP size=${SIZE_MIB}MiB threads=$THREADS clients=$CLIENTS seg=$SEG iter=$NUM_ITER $(date -Is) ========"
  echo "CMD: ${cmd[*]}"
} | tee -a "$LOG"

# whole-node CPU sampler in the background (best-effort; ignored if mpstat absent)
MPSTAT_LOG=$(mktemp)
if command -v mpstat >/dev/null 2>&1; then
  ( LC_ALL=C mpstat 2 > "$MPSTAT_LOG" 2>/dev/null ) &
  MPPID=$!
fi

"${cmd[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

if [ -n "${MPPID:-}" ]; then
  kill "$MPPID" 2>/dev/null; wait "$MPPID" 2>/dev/null
  # Average %usr and %sys over all "all" interval rows. Locate columns relative
  # to the "all" token so AM/PM time formats don't shift the fields.
  awk '$0 ~ /(^|[[:space:]])all([[:space:]])/ {
         for (i=1;i<=NF;i++) if ($i=="all") { u+=$(i+1); s+=$(i+3); n++; break }
       }
       END { if (n>0) printf "NIXLBENCH_CPU usr=%.3f sys=%.3f samples=%d\n", u/n, s/n, n }' \
       "$MPSTAT_LOG" | tee -a "$LOG"
fi
rm -f "$MPSTAT_LOG"

echo "CELL_EXIT rc=$rc op=$OP size=${SIZE_MIB}MiB threads=$THREADS clients=$CLIENTS" | tee -a "$LOG"
echo "" | tee -a "$LOG"
exit $rc
