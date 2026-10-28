#!/bin/bash
# Serves the 4-bit AWQ build on a 24 GB card.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
# 27B abliterado INT4 (AWQ) em GPU de 24 GB: MTP vs plain. Velocidade 1-16 usuarios + GSM8K/HumanEval (guarda de qualidade)
cd /workspace/eco; . /venv/main/bin/activate; mkdir -p out /workspace/models
hf download twolven/Qwen3.8-27B-abliterated-AWQ-MTP --local-dir /workspace/models/awq > out/dl.log 2>&1
echo DOWNLOAD_DONE
for V in mtp3 plain; do
  SPEC=""; [ "$V" = mtp3 ] && SPEC='{"method":"mtp","num_speculative_tokens":3}'
  T0=$(date +%s)
  if [ -n "$SPEC" ]; then
    nohup vllm serve /workspace/models/awq --served-model-name qwen-abliterated --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.94 --max-model-len 12288 --max-num-seqs 16 --reasoning-parser qwen3 --speculative-config "$SPEC" > out/vllm_$V.log 2>&1 &
  else
    nohup vllm serve /workspace/models/awq --served-model-name qwen-abliterated --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.94 --max-model-len 12288 --max-num-seqs 16 --reasoning-parser qwen3 > out/vllm_$V.log 2>&1 &
  fi
  PID=$!
  ok=0; for i in $(seq 1 150); do [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:18000/health)" = 200 ] && ok=1 && break; kill -0 $PID 2>/dev/null || break; sleep 5; done
  echo "variant=$V ok=$ok load_s=$(( $(date +%s) - T0 ))"
  if [ $ok = 1 ]; then
    python llm_bench.py --base http://127.0.0.1:18000/v1 --model qwen-abliterated --out out/llm_$V --levels 1,4,8,16 --dur 40 --think off
    python intel.py --base http://127.0.0.1:18000/v1 --model qwen-abliterated --label awq_$V --out out/intel_awq --only gsm8k,humaneval --n_gsm 100 --workers 16
  else tail -20 out/vllm_$V.log | cut -c1-220; fi
  kill $PID; sleep 10; pkill -f "vllm serve"; sleep 5
done
echo SCRIPT_DONE
