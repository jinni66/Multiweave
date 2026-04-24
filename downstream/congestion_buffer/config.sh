# =========================
# Multiscale Eval Config
# =========================

TRACE_LEN=2000

QUEUE_CAPACITY=1000
SERVICE_RATE=2

BUFFER_SIZES="1 2 4 6 8 16 32 64 128 256 512"

BIN_SIZE=10

RESULT_DIR="results"

echo "Config loaded:"
echo "TRACE_LEN=$TRACE_LEN"
echo "QUEUE_CAPACITY=$QUEUE_CAPACITY"
echo "SERVICE_RATE=$SERVICE_RATE"