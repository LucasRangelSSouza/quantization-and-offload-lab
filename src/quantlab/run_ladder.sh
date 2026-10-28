#!/bin/bash
# The ladder session: downloads each build, serves it with identical limits, runs capability checks, then tears down.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
# Sessao C (RTX PRO 6000): ESCADA DE QUANTIZACAO do 27B (AWQ -> NVFP4 -> FP8 -> BF16): inteligencia + comportamento, juiz fixo no fim; depois 9B: vLLM x SGLang (cuda graph).
cd /workspace/eco; . /venv/main/bin/activate; mkdir -p out/ld /workspace/models out/intel
log() { echo "STEP $1 $(date +%s)"; }
waitup() { for i in $(seq 1 240); do [ "$(curl -s -o /dev/null -w %{http_code} http://127.0.0.1:18000/health)" = 200 ] && return 0; pgrep -f "vllm serve" >/dev/null || return 1; sleep 5; done; return 1; }
dl_retry() { # dir repo : baixa com deteccao de travamento (sem crescer 120 s) e ate 3 tentativas
  for t in 1 2 3; do
    hf download "$2" --local-dir /workspace/models/$1 > out/ld/dl_$1_$t.log 2>&1 &
    P=$!; last=0; stall=0
    while kill -0 $P 2>/dev/null; do
      sleep 15; cur=$(du -sb /workspace/models/$1 2>/dev/null | cut -f1)
      if [ "$cur" = "$last" ]; then stall=$((stall+15)); else stall=0; fi; last=$cur
      if [ $stall -ge 120 ]; then echo "STALL $1 try $t" >> out/ld/stall.log; kill $P 2>/dev/null; break; fi
    done
    if wait $P 2>/dev/null; then touch /workspace/models/.ok_$1; return 0; fi
  done
  touch /workspace/models/.fail_$1; return 1
}
log dl
( dl_retry awq twolven/Qwen3.8-27B-abliterated-AWQ-MTP; dl_retry target Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4; dl_retry bf16 Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16; dl_retry q9 Qwen/Qwen3.5-9B ) &  # em cadeia: o menor chega primeiro e a GPU ja trabalha enquanto o resto baixa
waitfile() { for i in $(seq 1 480); do [ -e /workspace/models/.ok_$1 ] && return 0; [ -e /workspace/models/.fail_$1 ] && return 1; sleep 5; done; return 1; }
serve() { # dir quant_args
  pkill -f "vllm serve"; sleep 6
  nohup vllm serve /workspace/models/$1 --served-model-name qwen-abliterated --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.90 --max-model-len 8192 --max-num-seqs 24 \
    --trust-remote-code --reasoning-parser qwen3 $2 > out/ld/vllm_$1_$3.log 2>&1 &
  waitup
}
rung() { # nome dir quant_args
  log $1
  if ! waitfile $2; then echo "DOWNLOAD_FALHOU $1"; return; fi
  if serve $2 "$3" $1; then
    python intel.py --base http://127.0.0.1:18000/v1 --model qwen-abliterated --label ladder_$1 --out out/intel --workers 24 > out/ld/intel_$1.log 2>&1
    python behavior.py --base http://127.0.0.1:18000/v1 --model qwen-abliterated --key x --out out/ld/beh_$1 --label ladder_$1 --workers 8 > out/ld/beh_$1.log 2>&1
    tail -5 out/ld/intel_$1.log
  else
    echo "SERVE_FALHOU $1"; tail -15 out/ld/vllm_$2_$1.log | cut -c1-200
  fi
}
rung awq awq ""
rung nvfp4 target ""
rung fp8 bf16 "--quantization fp8"
rung bf16 bf16 ""
# juiz fixo: o BF16 (ultimo servido) re-julga as geracoes das quatro configuracoes
if curl -s -o /dev/null http://127.0.0.1:18000/health; then
  log rejudge
  python rejudge.py --base http://127.0.0.1:18000/v1 --model qwen-abliterated --dirs out/ld/beh_awq,out/ld/beh_nvfp4,out/ld/beh_fp8,out/ld/beh_bf16 --out out/ld/rejudge.jsonl > out/ld/rejudge.log 2>&1
  tail -1 out/ld/rejudge.log
fi
pkill -f "vllm serve"; sleep 6
# ---- 9B: vLLM x SGLang na MESMA GPU (cuda graph ligado no SGLang)
log engines9b
if waitfile q9; then
  nohup vllm serve /workspace/models/q9 --served-model-name q9 --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.85 --max-model-len 8192 --max-num-seqs 32 --reasoning-parser qwen3 > out/ld/vllm_q9.log 2>&1 &
  if waitup; then
    python llm_bench.py --base http://127.0.0.1:18000/v1 --model q9 --out out/ld/e9_vllm --levels 1,8,16,32 --dur 25 --think off > out/ld/e9_vllm.log 2>&1
  fi
  pkill -f "vllm serve"; sleep 8
  pip install -q uv >/dev/null 2>&1; uv venv /workspace/se --python 3.12 -q && . /workspace/se/bin/activate
  ( time uv pip install -q "sglang[all]" httpx ) > out/ld/sglang_install.log 2>&1
  nohup python -m sglang.launch_server --model-path /workspace/models/q9 --served-model-name q9 --host 127.0.0.1 --port 18000 --mem-fraction-static 0.85 --context-length 8192 --cuda-graph-max-bs 32 --reasoning-parser qwen3 > out/ld/sglang.log 2>&1 &
  waitup && python llm_bench.py --base http://127.0.0.1:18000/v1 --model q9 --out out/ld/e9_sglang --levels 1,8,16,32 --dur 25 --think off > out/ld/e9_sglang.log 2>&1
  grep -i -m3 "cuda graph" out/ld/sglang.log | cut -c1-200
  pkill -f sglang.launch_server
fi
log end
echo SCRIPT_DONE
