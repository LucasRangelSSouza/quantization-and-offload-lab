# Capability harness: GSM8K, MMLU, HumanEval-style code tasks, JSON-format tasks against an OpenAI-compatible server.
# Original harness script, copied unmodified apart from this header. Comments inside may be in Portuguese.
"""Bateria de inteligencia (deterministica + PT para juiz): GSM8K, MMLU (amostra), HumanEval (pass@1 com execucao), JSON estruturado, PT livre.
uso: python intel.py --base http://127.0.0.1:18000/v1 --model NAME --label rotulo --out DIR [--think off] [--n_gsm 150 --n_mmlu 200]
Saidas (bruto, 1 linha por item): DIR/<label>/{gsm8k,mmlu,humaneval,json,pt}.jsonl e summary.json. Temperatura 0, sem raciocinio (thinking off)."""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

DS = "https://datasets-server.huggingface.co/rows"


def rows(dataset, config, split, offset, length):
    _name = {"openai/gsm8k": "gsm8k", "cais/mmlu": "mmlu", "openai/openai_humaneval": "humaneval"}[dataset]
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", _name + ".json")
    if os.path.exists(_p):
        _all = json.load(open(_p, encoding="utf-8"))
        if _name == "mmlu":
            _c = rows.__dict__.setdefault("_mmlu_i", 0)
            rows._mmlu_i = _c + 1
            return _all[(_c % 10) * 20:(_c % 10) * 20 + length]
        return _all[offset:offset + length]
    for _ in range(4):
        try:
            r = httpx.get(DS, params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}, timeout=60)
            if r.status_code == 200:
                return [x["row"] for x in r.json()["rows"]]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return []


def chat(base, model, prompt, max_tokens, think, temp=0.0, seed=1, key="x"):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temp, "seed": seed, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.perf_counter()
    for _ in range(3):
        try:
            r = httpx.post(base + "/chat/completions", json=body, headers={"Authorization": "Bearer " + key}, timeout=600)
            j = r.json()
            m = j["choices"][0]["message"]
            return {"text": m.get("content") or "", "finish": j["choices"][0]["finish_reason"], "usage": j.get("usage"), "s": round(time.perf_counter() - t0, 3)}
        except Exception as e:  # noqa: BLE001
            err = repr(e)[:200]
            time.sleep(2)
    return {"text": "", "finish": "error", "usage": None, "s": 0, "error": err}


def pmap(fn, items, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def gsm8k(a, out):
    data = []
    for off in range(0, a.n_gsm, 100):
        data += rows("openai/gsm8k", "main", "test", off, min(100, a.n_gsm - off))
    def one(r):
        p = r["question"] + "\n\nResolva passo a passo e termine com a linha 'Resposta final: <numero>'."
        g = chat(a.base, a.model, p, 1200, a.think)
        t = g["text"]
        m = re.search(r"Resposta final:\s*(.+)", t)
        pred = num(m.group(1)) if m else num(t[-200:])
        gold = num(r["answer"].split("####")[-1])
        return {"q": r["question"][:120], "pred": pred, "gold": gold, "ok": pred is not None and abs(pred - gold) < 1e-6, "finish": g["finish"], "tokens": (g["usage"] or {}).get("completion_tokens"), "s": g["s"]}
    res = pmap(one, data, a.workers)
    return res


def mmlu(a, out):
    rnd = random.Random(7)
    data = []
    for _ in range(a.n_mmlu // 20):
        off = rnd.randrange(0, 14000)
        data += rows("cais/mmlu", "all", "test", off, 20)
    L = "ABCD"
    def one(r):
        p = r["question"] + "\n" + "\n".join(f"{L[i]}. {c}" for i, c in enumerate(r["choices"])) + "\n\nResponda apenas com a letra da alternativa correta (A, B, C ou D)."
        g = chat(a.base, a.model, p, 16, a.think)
        m = re.search(r"\b([ABCD])\b", g["text"].strip().upper())
        pred = m.group(1) if m else None
        return {"subject": r["subject"], "pred": pred, "gold": L[r["answer"]], "ok": pred == L[r["answer"]], "s": g["s"]}
    return pmap(one, data, a.workers)


def run_code(code, test, entry, timeout=15):
    prog = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w", encoding="utf-8").write(prog)
        try:
            p = subprocess.run([sys.executable, f], capture_output=True, text=True, timeout=timeout, cwd=d)
            return p.returncode == 0, (p.stderr or "")[-200:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def humaneval(a, out):
    data = rows("openai/openai_humaneval", "openai_humaneval", "test", 0, 100) + rows("openai/openai_humaneval", "openai_humaneval", "test", 100, 64)
    def one(r):
        p = "Complete a funcao Python abaixo. Responda SOMENTE com o codigo completo da funcao (incluindo imports e a assinatura) em um unico bloco ```python```.\n\n" + r["prompt"]
        g = chat(a.base, a.model, p, 1200, a.think)
        m = re.search(r"```(?:python)?\n(.*?)```", g["text"], re.S)
        code = m.group(1) if m else g["text"]
        ok, err = run_code(code, r["test"], r["entry_point"])
        return {"task": r["task_id"], "ok": ok, "err": err, "finish": g["finish"], "s": g["s"]}
    return pmap(one, data, max(4, a.workers // 2))


JSON_TASKS = [
    ("pedido de padaria com 3 itens", {"cliente": str, "itens": list, "total": float}),
    ("cadastro de aluno com nome, idade e curso", {"nome": str, "idade": int, "curso": str}),
    ("resumo de ticket de suporte com assunto, prioridade (alta|media|baixa) e resolvido (true/false)", {"assunto": str, "prioridade": str, "resolvido": bool}),
    ("evento de agenda com titulo, data (YYYY-MM-DD) e participantes (lista)", {"titulo": str, "data": str, "participantes": list}),
    ("produto com sku, preco e estoque", {"sku": str, "preco": float, "estoque": int}),
    ("endereco com rua, numero, cidade, uf", {"rua": str, "numero": int, "cidade": str, "uf": str}),
    ("analise de sentimento com texto, sentimento e confianca (0 a 1)", {"texto": str, "sentimento": str, "confianca": float}),
    ("fatura com numero, valor e pago", {"numero": str, "valor": float, "pago": bool}),
]


def jsontask(a, out):
    items = [(d, s, v) for (d, s) in JSON_TASKS for v in range(3)]
    def one(it):
        d, schema, v = it
        p = f"Gere um exemplo ficticio (variante {v}) de {d}. Responda SOMENTE com um objeto JSON valido com exatamente as chaves {list(schema)} e tipos {[t.__name__ for t in schema.values()]}. Sem texto extra."
        g = chat(a.base, a.model, p, 400, a.think, temp=0.7, seed=v + 1)
        t = g["text"].strip()
        t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
        ok = False
        try:
            j = json.loads(t)
            ok = isinstance(j, dict) and set(j) == set(schema) and all(isinstance(j[k], tp) or (tp is float and isinstance(j[k], int) and not isinstance(j[k], bool)) for k, tp in schema.items())
        except Exception:  # noqa: BLE001
            pass
        return {"task": d[:40], "variant": v, "ok": ok, "s": g["s"]}
    return pmap(one, items, a.workers)


PT = ["Explique, para um dono de padaria, o que e fluxo de caixa e como acompanhar na pratica.",
      "Escreva um e-mail cordial e firme cobrando uma fatura vencida ha 15 dias.",
      "Compare, em 2 paragrafos, banco de dados relacional e NoSQL para um e-commerce pequeno.",
      "Um cliente reclama que o pedido chegou atrasado. Escreva a resposta de atendimento no WhatsApp, empatica e objetiva.",
      "Resuma em 5 linhas por que testes automatizados importam em software.",
      "Crie um plano de estudo de 4 semanas para aprender SQL do zero, com horas por semana.",
      "Explique a diferenca entre CDB, LCI e Tesouro Selic para quem esta comecando a investir.",
      "Escreva uma funcao Python que valide CPF e explique brevemente a logica.",
      "Reescreva de forma mais clara: 'O sistema apresentou indisponibilidade em decorrencia de falha na camada de persistencia.'",
      "Quais sao 5 riscos de colocar um agente de IA para responder clientes sozinho e como mitigar cada um?",
      "Crie uma tabela em markdown com 4 vantagens e 4 desvantagens de trabalhar remotamente.",
      "Explique o que e um indice de banco de dados com uma analogia do dia a dia.",
      "Um grupo de 5 amigos divide uma conta de R$ 387,50 com 10% de gorjeta. Quanto cada um paga? Mostre o calculo.",
      "Escreva um texto curto de anuncio para uma padaria artesanal que abre uma unidade nova.",
      "Explique o que significa idempotencia numa API e por que importa para pagamentos.",
      "Liste os passos para investigar por que um site ficou lento de repente.",
      "Escreva 3 perguntas de entrevista para um analista de dados junior e o que voce espera de uma boa resposta.",
      "Explique, sem jargao, o que e um modelo de linguagem.",
      "Corrija e explique os erros: 'Nois vai fazer a reuniao amanha, mais tem que confirma se todos podem vim.'",
      "Proponha 3 nomes e slogans para um app de controle financeiro pessoal voltado a autonomos."]


def pt(a, out):
    def one(i):
        g = chat(a.base, a.model, PT[i], 700, a.think)
        return {"i": i, "prompt": PT[i], "answer": g["text"], "finish": g["finish"], "s": g["s"]}
    return pmap(one, list(range(len(PT))), a.workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base"); ap.add_argument("--model"); ap.add_argument("--label"); ap.add_argument("--out", default="out/intel")
    ap.add_argument("--think", default="off"); ap.add_argument("--n_gsm", type=int, default=150); ap.add_argument("--n_mmlu", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--only", default="")
    a = ap.parse_args()
    a.think = a.think == "on"
    d = os.path.join(a.out, a.label)
    os.makedirs(d, exist_ok=True)
    summary = {"label": a.label, "model": a.model, "thinking": a.think, "t_start": time.time()}
    for name, fn in (("gsm8k", gsm8k), ("mmlu", mmlu), ("humaneval", humaneval), ("json", jsontask), ("pt", pt)):
        if a.only and name not in a.only.split(","):
            continue
        t0 = time.time()
        res = fn(a, d)
        with open(os.path.join(d, name + ".jsonl"), "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if name != "pt":
            n, ok = len(res), sum(1 for r in res if r.get("ok"))
            summary[name] = {"n": n, "pass": ok, "acc": round(ok / n, 4) if n else None, "wall_s": round(time.time() - t0, 1)}
        else:
            summary[name] = {"n": len(res), "wall_s": round(time.time() - t0, 1)}
        print(name, summary[name], flush=True)
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
