import argparse
import asyncio
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import orjson
import yaml

SYSTEM_PROMPT = "Output only JSON with a single `question` field; do not include reasoning or extra text."
ANCHOR_SYSTEM_PROMPT = "Return only the clinical anchors as plain text bullets. No JSON."
ANCHOR_PROMPT = """Extract clinical anchors from the text.
Return a compact list with: conditions, population, interventions/exposures, outcomes, key labs/imaging, mechanisms, confounders, and specialty focus (nephrology preferred, otherwise internal medicine).
Text:
{text}
"""

MEDICAL_KEYWORDS = {
    "diagnosis", "differential", "pathophysiology", "management", "treatment", "therapy",
    "prognosis", "risk", "outcome", "mortality", "morbidity", "trial", "cohort", "meta-analysis",
    "biomarker", "guideline", "mechanism", "renal", "kidney", "nephrology", "glomerular",
    "tubular", "proteinuria", "creatinine", "egfr", "dialysis", "transplant", "aki", "ckd",
    "electrolyte", "acid-base", "hypertension", "diabetes", "lupus", "vasculitis",
    "immunosuppression", "steroid", "biopsy", "urinalysis", "imaging"
}

DEPTH_KEYWORDS = {
    "mechanism", "pathway", "pathophysiology", "hemodynamic", "diagnostic strategy",
    "differential", "risk stratification", "external validity", "confounding", "bias",
    "subgroup", "hazard ratio", "relative risk", "odds ratio", "number needed",
    "management", "treatment decision", "benefit", "harm", "trade-off"
}

NEPHRO_KEYWORDS = {
    "renal", "kidney", "nephrology", "glomerular", "tubular", "proteinuria",
    "creatinine", "egfr", "aki", "ckd", "dialysis", "transplant",
    "electrolyte", "acid-base", "bicarbonate", "potassium", "sodium",
    "phosphate", "calcium", "pth", "urinalysis", "hematuria"
}


def now_warsaw():
    return datetime.now(ZoneInfo("Europe/Warsaw"))


def load_prompts(path: Path):
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^#\s*(\d+)\s*$", text)
    prompts = {}
    for i in range(1, len(parts), 2):
        pid = int(parts[i])
        body = parts[i + 1].strip()
        prompts[pid] = body
    return [prompts[i] for i in sorted(prompts) if i in (1, 2, 3, 4)]


def normalize(val):
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return "\n".join(str(x) for x in val if x is not None)
    return orjson.dumps(val).decode("utf-8")


def combine(*parts):
    return "\n".join(p for p in (p.strip() for p in parts if p) if p)


def select_input_text(source_file, obj):
    name = Path(source_file).name
    if name in ("meta_analysis_clean.jsonl", "systematic_review_clean.jsonl"):
        ct = obj.get("context_truncated")
        if ct is True or str(ct).lower() == "true":
            return combine(normalize(obj.get("abstract_text")), normalize(obj.get("sections_clean")))
        ctx = normalize(obj.get("context"))
        if ctx:
            return ctx
        return combine(normalize(obj.get("abstract_text")), normalize(obj.get("sections_clean")))
    if name == "original.jsonl":
        sections = normalize(obj.get("sections"))
        if sections:
            return sections
        abstract = normalize(obj.get("abstract"))
        if abstract:
            return abstract
        return normalize(obj.get("tables"))
    if name == "trials.jsonl":
        base = normalize(obj.get("sections")) or normalize(obj.get("abstract"))
        extras = combine(normalize(obj.get("outcomes")), normalize(obj.get("tables")))
        return combine(base, extras)
    return normalize(obj)


def estimate_tokens(text):
    return len(text) / 4.0


def truncate_text(text, max_tokens):
    max_chars = int(max_tokens * 4)
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


def extract_json_str(text):
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    return None


def validate_question(q, min_chars, require_nephro):
    if not q or not isinstance(q, str):
        return False
    if len(q.strip()) < min_chars:
        return False
    if not q.strip().endswith("?"):
        return False
    low = q.lower()
    if any(x in low for x in ("above", "the text", "provided text", "given text")):
        return False
    if not any(k in low for k in MEDICAL_KEYWORDS):
        return False
    if not any(k in low for k in DEPTH_KEYWORDS):
        return False
    if require_nephro and not any(k in low for k in NEPHRO_KEYWORDS):
        return False
    return True


def build_messages(prompt, system_prompt=SYSTEM_PROMPT):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def safe_json_loads(s):
    return orjson.loads(s)


class Runner:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.model = cfg["models"][args.model]
        self.hw = cfg["hardware_profiles"][args.hardware]
        self.prompts = load_prompts(Path(args.prompts))
        if len(self.prompts) != 4:
            raise ValueError("expected 4 prompts")
        ts = now_warsaw().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(args.outputs) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_out_dir = self.run_dir / self.model["served_model_name"]
        self.model_out_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.snapshot.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
        (self.run_dir / "prompts.snapshot.md").write_text(Path(args.prompts).read_text(encoding="utf-8"), encoding="utf-8")
        meta = {
            "model_key": args.model,
            "hardware_key": args.hardware,
            "base_url": args.base_url,
            "inputs": args.inputs,
            "concurrency": args.concurrency,
            "max_retries": args.max_retries,
            "min_question_chars": args.min_question_chars,
            "require_nephro": args.require_nephro,
        }
        try:
            import vllm
            meta["vllm_version"] = getattr(vllm, "__version__", "unknown")
        except Exception:
            meta["vllm_version"] = "unknown"
        (self.run_dir / "run_meta.json").write_text(orjson.dumps(meta).decode("utf-8"), encoding="utf-8")

        self.locks = defaultdict(asyncio.Lock)
        self.stats = defaultdict(int)
        self.latencies = []
        self.prompt_tokens = []
        self.completion_tokens = []

    def output_path(self, source_file):
        name = Path(source_file).name
        return self.model_out_dir / f"{name}.questions.jsonl"

    def log_path(self):
        return self.log_dir / f"{self.model['served_model_name']}.jsonl"

    def get_sampling(self):
        return self.model.get("sampling_defaults", {})

    def get_max_new_tokens(self):
        sd = self.get_sampling()
        return sd.get("max_new_tokens") or self.args.max_new_tokens

    def get_max_model_len(self):
        return self.model.get("max_model_len")

    def should_anchor(self, text):
        max_model_len = self.get_max_model_len()
        max_new = self.get_max_new_tokens() or 0
        if not max_model_len:
            return False
        limit = max_model_len - max_new - 1024
        return estimate_tokens(text) > limit

    async def write_line(self, path, line):
        async with self.locks[str(path)]:
            await asyncio.to_thread(self._append_line, path, line)

    @staticmethod
    def _append_line(path, line):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    async def log_event(self, payload):
        line = orjson.dumps(payload).decode("utf-8") + "\n"
        await self.write_line(self.log_path(), line)


async def call_chat(client, base_url, model_name, messages, sampling, max_tokens, extra_body=None, timeout=300):
    payload = {
        "model": model_name,
        "messages": messages,
    }
    if sampling.get("temperature") is not None:
        payload["temperature"] = sampling["temperature"]
    if sampling.get("top_p") is not None:
        payload["top_p"] = sampling["top_p"]
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if extra_body:
        payload["extra_body"] = extra_body
    url = base_url.rstrip("/") + "/chat/completions"
    r = await client.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


async def generate_anchor(runner, client, text):
    sampling = runner.get_sampling()
    max_tokens = min(512, runner.get_max_new_tokens() or 512)
    max_model_len = runner.get_max_model_len()
    if max_model_len:
        max_input_tokens = max_model_len - max_tokens - 1024
        text = truncate_text(text, max_input_tokens)
    prompt = ANCHOR_PROMPT.format(text=text)
    t0 = time.perf_counter()
    data = await call_chat(
        client,
        runner.args.base_url,
        runner.model["served_model_name"],
        build_messages(prompt, ANCHOR_SYSTEM_PROMPT),
        sampling,
        max_tokens,
        extra_body=runner.args.extra_body,
    )
    latency = (time.perf_counter() - t0) * 1000
    content = data["choices"][0]["message"]["content"]
    await runner.log_event({"type": "anchor", "latency_ms": latency})
    return content


async def generate_question(runner, client, prompt_id, prompt, input_text, source_file, line_index):
    max_tokens = runner.get_max_new_tokens()
    if max_tokens is None:
        max_tokens = 1024
    sampling = runner.get_sampling()
    model_name = runner.model["served_model_name"]

    for attempt in range(runner.args.max_retries + 1):
        t0 = time.perf_counter()
        try:
            filled = prompt.replace("{text}", input_text)
            data = await call_chat(
                client,
                runner.args.base_url,
                model_name,
                build_messages(filled),
                sampling,
                max_tokens,
                extra_body=runner.args.extra_body,
            )
            latency = (time.perf_counter() - t0) * 1000
            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            json_str = extract_json_str(choice)
            if not json_str:
                raise ValueError("non_json")
            obj = safe_json_loads(json_str)
            if list(obj.keys()) != ["question"]:
                raise ValueError("extra_keys")
            q = obj.get("question")
            if not validate_question(q, runner.args.min_question_chars, runner.args.require_nephro):
                raise ValueError("quality")

            runner.stats["success"] += 1
            runner.latencies.append(latency)
            if usage.get("prompt_tokens"):
                runner.prompt_tokens.append(usage["prompt_tokens"])
            if usage.get("completion_tokens"):
                runner.completion_tokens.append(usage["completion_tokens"])

            await runner.log_event({
                "type": "question",
                "status": "ok",
                "source_file": Path(source_file).name,
                "line_index": line_index,
                "article_id": f"{Path(source_file).name}:{line_index}",
                "prompt_id": prompt_id,
                "attempt": attempt,
                "latency_ms": latency,
                "usage": usage,
            })
            return q
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            runner.stats["failed"] += 1
            await runner.log_event({
                "type": "question",
                "status": "error",
                "source_file": Path(source_file).name,
                "line_index": line_index,
                "article_id": f"{Path(source_file).name}:{line_index}",
                "prompt_id": prompt_id,
                "attempt": attempt,
                "latency_ms": latency,
                "error": str(e),
            })
            if attempt >= runner.args.max_retries:
                return None


async def process_article(runner, client, source_file, line_index, raw_line):
    runner.stats["records"] += 1
    try:
        obj = orjson.loads(raw_line)
    except Exception:
        runner.stats["bad_json"] += 1
        return

    raw_text = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, (bytes, bytearray)) else raw_line
    input_text = select_input_text(source_file, obj)
    if not input_text:
        input_text = raw_text

    if runner.should_anchor(raw_text) or runner.should_anchor(input_text):
        try:
            input_text = await generate_anchor(runner, client, input_text)
        except Exception:
            runner.stats["anchor_failed"] += 1

    for i, p in enumerate(runner.prompts, 1):
        q = await generate_question(runner, client, i, p, input_text, source_file, line_index)
        if q:
            out = {
                "source_file": Path(source_file).name,
                "line_index": line_index,
                "prompt_id": i,
                "model_id": runner.model["model_id"],
                "question": q,
            }
            line = orjson.dumps(out).decode("utf-8") + "\n"
            await runner.write_line(runner.output_path(source_file), line)
        else:
            runner.stats["dropped"] += 1


async def run(runner):
    limits = httpx.Limits(max_connections=runner.args.concurrency, max_keepalive_connections=runner.args.concurrency)
    timeout = httpx.Timeout(300)
    start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        queue = asyncio.Queue(maxsize=runner.args.concurrency * 2)

        async def worker():
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    break
                src, idx, line = item
                await process_article(runner, client, src, idx, line)
                queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(runner.args.concurrency)]

        for src in runner.args.inputs:
            src_path = Path(src)
            if not src_path.exists():
                continue
            with src_path.open("rb") as f:
                for idx, line in enumerate(f):
                    if runner.args.max_lines and idx >= runner.args.max_lines:
                        break
                    await queue.put((src, idx, line))

        for _ in workers:
            await queue.put(None)
        await queue.join()
        await asyncio.gather(*workers)

    duration = time.perf_counter() - start
    total_prompt = sum(runner.prompt_tokens)
    total_completion = sum(runner.completion_tokens)
    summary = {
        "records": runner.stats["records"],
        "bad_json": runner.stats["bad_json"],
        "anchor_failed": runner.stats["anchor_failed"],
        "success": runner.stats["success"],
        "failed": runner.stats["failed"],
        "dropped": runner.stats["dropped"],
        "avg_latency_ms": sum(runner.latencies) / len(runner.latencies) if runner.latencies else 0,
        "avg_prompt_tokens": sum(runner.prompt_tokens) / len(runner.prompt_tokens) if runner.prompt_tokens else 0,
        "avg_completion_tokens": sum(runner.completion_tokens) / len(runner.completion_tokens) if runner.completion_tokens else 0,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "duration_s": duration,
        "throughput_completion_tokens_per_s": (total_completion / duration) if duration > 0 else 0,
    }
    (runner.run_dir / "summary.json").write_text(orjson.dumps(summary).decode("utf-8"), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--inputs", nargs="+", default=[
        "input/systematic_review_clean.jsonl",
        "input/meta_analysis_clean.jsonl",
        "input/original.jsonl",
        "input/trials.jsonl",
    ])
    ap.add_argument("--prompts", default="prompts.md")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--min-question-chars", type=int, default=160)
    ap.add_argument("--require-nephro", action="store_true")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--extra-body", type=orjson.loads, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runner = Runner(cfg, args)
    asyncio.run(run(runner))


if __name__ == "__main__":
    main()
