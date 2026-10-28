# Closed-loop streaming load generator used for the speed side of the ladder.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
"""Bench do LLM (OpenAI-compativel, streaming). Roda NA VM da GPU. Grava 1 linha por requisicao + tempos de cada chunk.
uso: python llm_bench.py --base http://127.0.0.1:18000/v1 --model qwen-abliterated --key KEY --out DIR --levels 1,2,4,8,16 --dur 60"""
import argparse, asyncio, json, os, random, time, uuid
import httpx

CODE = '''def process(items, flt=None):
    out = []
    for i in items:
        if flt is None or flt(i):
            out.append({"id": i["id"], "v": i.get("v", 0) * 2})
    return out
'''
FILL = ("# modulo de utilidades\n" + CODE) * 1


def build(kind, rnd):
    if kind == "chat":
        return [{"role": "user", "content": rnd.choice([
            "Explique em 5 linhas o que e um indice invertido e quando usar.",
            "Escreva um e-mail curto cobrando uma fatura vencida, tom cordial.",
            "Quais as diferencas entre fila e pilha? Responda em portugues, objetivo."])}], 256
    if kind == "code":
        return [{"role": "user", "content": "Refatore para usar compreensao de lista, adicione type hints e 3 testes pytest:\n" + CODE}], 512
    if kind == "json":
        return [{"role": "user", "content": 'Retorne SOMENTE um JSON {"nome":str,"itens":[{"sku":str,"qtd":int}],"total":float} para um pedido de padaria com 4 itens.'}], 200
    if kind == "long8k":
        body = (FILL * 180)[:30000]
        return [{"role": "user", "content": body + "\n\nResuma em 3 linhas o que este arquivo faz."}], 256
    raise ValueError(kind)

MIX = [("chat", .35), ("code", .30), ("json", .20), ("long8k", .15)]


def pick(rnd):
    r, a = rnd.random(), 0
    for k, w in MIX:
        a += w
        if r <= a:
            return k
    return "chat"


async def one(client, args, level, rid, kind, rnd, out, think):
    msgs, maxt = build(kind, rnd)
    body = {"model": args.model, "messages": msgs, "max_tokens": maxt, "temperature": 0.7, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.perf_counter(); chunks = []; ttft = None; usage = None; err = None; status = None; first_content = None
    try:
        async with client.stream("POST", args.base + "/chat/completions", json=body, headers={"Authorization": "Bearer " + args.key}, timeout=600) as r:
            status = r.status_code
            async for line in r.aiter_lines():
                if not line.startswith("data:"): continue
                d = line[5:].strip()
                if d == "[DONE]": break
                now = time.perf_counter() - t0
                try: j = json.loads(d)
                except Exception: continue
                if j.get("usage"): usage = j["usage"]
                ch = j.get("choices") or []
                if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning") or ch[0].get("delta", {}).get("reasoning_content")):
                    if ttft is None: ttft = now
                    chunks.append(round(now, 4))
    except Exception as e:
        err = repr(e)[:200]
    total = time.perf_counter() - t0
    rec = {"request_id": rid, "level": level, "kind": kind, "thinking": think, "status": status, "error": err, "ttft_s": ttft,
           "total_s": round(total, 4), "usage": usage, "n_chunks": len(chunks), "chunk_times_s": chunks, "t_end_wall": time.time()}
    with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(rec) + "\n")
    return rec


async def level(args, n, out, think):
    rnd = random.Random(args.seed + n)
    stop = time.time() + args.dur
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=n + 4)) as c:
        async def worker(w):
            k = 0
            while time.time() < stop:
                await one(c, args, n, f"L{n}-w{w}-{k}-{uuid.uuid4().hex[:6]}", pick(rnd), rnd, out, think)
                k += 1
        t0 = time.time()
        await asyncio.gather(*[worker(w) for w in range(n)])
        return time.time() - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--model"); ap.add_argument("--key", default="x"); ap.add_argument("--out")
    ap.add_argument("--levels", default="1,2,4,8,16"); ap.add_argument("--dur", type=int, default=60); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--think", default="off,on")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "requests.jsonl")
    # warmup
    async with httpx.AsyncClient() as c:
        await one(c, args, 0, "warmup", "chat", random.Random(1), os.path.join(args.out, "warmup.jsonl"), False)
    for th in args.think.split(","):
        for n in [int(x) for x in args.levels.split(",")]:
            dt = await level(args, n, out, th == "on")
            print("level", n, "think", th, "wall", round(dt, 1), flush=True)

asyncio.run(main())
