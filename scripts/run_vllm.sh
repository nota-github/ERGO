CUDA_VISIBLE_DEVICES=0 \
vllm serve nota-ai/ERGO-7B \
  --port 8008 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --served-model-name "ergo" \
  --trust-remote-code \
  --disable-log-requests