import argparse
import shlex
import subprocess
from pathlib import Path

import yaml


def build_cmd(cfg, model_key, hw_key, extra_args):
    if model_key not in cfg.get("models", {}):
        raise KeyError(f"model '{model_key}' not found in config")
    if hw_key not in cfg.get("hardware_profiles", {}):
        raise KeyError(f"hardware profile '{hw_key}' not found in config")

    model = cfg["models"][model_key]
    hw = cfg["hardware_profiles"][hw_key]

    cmd = ["vllm", "serve", model["model_id"]]

    if model.get("served_model_name"):
        cmd += ["--served-model-name", str(model["served_model_name"])]
    if model.get("max_model_len"):
        cmd += ["--max-model-len", str(model["max_model_len"])]
    if model.get("dtype"):
        cmd += ["--dtype", str(model["dtype"])]
    if model.get("quantization"):
        cmd += ["--quantization", str(model["quantization"])]
    if hw.get("tensor_parallel_size"):
        cmd += ["--tensor-parallel-size", str(hw["tensor_parallel_size"])]
    if hw.get("pipeline_parallel_size") and int(hw["pipeline_parallel_size"]) > 1:
        cmd += ["--pipeline-parallel-size", str(hw["pipeline_parallel_size"])]
    if hw.get("max_num_seqs"):
        cmd += ["--max-num-seqs", str(hw["max_num_seqs"])]
    if hw.get("max_num_batched_tokens"):
        cmd += ["--max-num-batched-tokens", str(hw["max_num_batched_tokens"])]
    if hw.get("gpu_memory_utilization"):
        cmd += ["--gpu-memory-utilization", str(hw["gpu_memory_utilization"])]

    if model.get("enable_reasoning"):
        cmd.append("--enable-reasoning")
    if model.get("reasoning_parser"):
        cmd += ["--reasoning-parser", str(model["reasoning_parser"])]
    if model.get("tool_call_parser"):
        cmd += ["--tool-call-parser", str(model["tool_call_parser"])]
    if model.get("chat_template"):
        cmd += ["--chat-template", str(model["chat_template"])]
    if model.get("generation_config"):
        cmd += ["--generation-config", str(model["generation_config"])]

    if extra_args:
        cmd += shlex.split(extra_args)

    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--extra-args")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    cmd = build_cmd(cfg, args.model, args.hardware, args.extra_args)

    if args.dry_run:
        print(" ".join(shlex.quote(c) for c in cmd))
        return

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
