#!/usr/bin/env python3
"""Build the held-out eval kernel notebook from data/heldout-eval-v1.jsonl.

Eval design (see data/heldout-eval-v1-report.md): 25 fresh questions, never
trained on, zero overlap with DPO prompts / self-test questions. The kernel
runs them on the BASE model and on the v15 DPO adapter (loaded from the
training kernel's latest successful output via kernel_sources), greedy
decoding, same ChatML + system prompt as the self-test. Auto-scores with
must_contain / must_not_contain; manual scoring happens at check time.
"""
import json

EVAL = [json.loads(l) for l in open("../data/heldout-eval-v1.jsonl", encoding="utf-8") if l.strip()]
assert len(EVAL) == 25, len(EVAL)

MD = """## Held-out 评测 v1：底座 vs v15（DPO 后）

- **目的**：自测 15 题已不能当评判标准——训练数据就是照着它们的毛病建的。本评测用 25 道从未进过训练的新题（见 `data/heldout-eval-v1.jsonl`），量 v15 的真实水平，并与底座在同一 kernel、同一解码条件下对比。
- **做法**：先跑底座（Qwen2.5-7B-Instruct，未微调），再加载 v15 的 DPO 适配器跑一遍；贪心解码（do_sample=False，max_new_tokens 320），保证可复现；ChatML 与 system 提示与自测相同。
- **评分**：kernel 内自动检查（must_contain 全出现、must_not_contain 全不出现）；must_contain 为空的题只拦禁用串，最终结论由核查人工读答案判定。
- **预期**：约 10–15 分钟。结果存 `outputs/heldout-eval-v1-results.json`。
"""

CELL2 = """# 2. 配置与评测题（题目硬编码在笔记本里，不走数据集挂载，避开路径问题）
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # 4-bit 底座
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 320
SYS = "你是一位精通中国历代官制的学者，回答要准确、克制，有出处意识，不确定的地方明说不确定。"
# v15 的 DPO 适配器：训练 kernel 最新成功版（v15）的输出，以 kernel_sources 挂载。
# 顶层目录本身就是合法的适配器目录（含 adapter_model.safetensors 与 adapter_config.json）。
ADAPTER_CANDIDATES = [
    "/kaggle/input/officialdom-qwen2-5-7b-sft-batch-1/outputs/officialdom-qwen2.5-7b-sft-v6-dpo",
    "/kaggle/input/officialdom-qwen2-5-7b-sft-batch-1/officialdom-qwen2.5-7b-sft-v6-dpo",
]
EVAL = __EVAL_JSON__
print("评测题：", len(EVAL), "道")
from collections import Counter as _Counter
print("分类：", dict(_Counter(r["category"] for r in EVAL)))
"""

CELL3 = """# 3. 定位 v15 的 DPO 适配器（候选路径 + 全盘搜索；找不到就大声报错）
import json, os

def _list_input(root="/kaggle/input"):
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        print("  " * level + os.path.basename(dirpath) + "/")
        for f in filenames:
            print("  " * (level + 1) + f)

print("=== /kaggle/input 实际挂载内容 ===")
_list_input()
print("=== DPO 适配器路径检查 ===")
ADAPTER_PATH = None
for p in ADAPTER_CANDIDATES:
    print(("OK  " if os.path.exists(os.path.join(p, "adapter_model.safetensors")) else "缺失") + " " + p)
    if os.path.exists(os.path.join(p, "adapter_model.safetensors")):
        ADAPTER_PATH = p
        break
if ADAPTER_PATH is None:
    print("=== 候选路径全缺失，全盘搜索 DPO 适配器 ===")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if "adapter_model.safetensors" in filenames and "ref" not in dirpath.split(os.sep):
            if "dpo" in dirpath:
                ADAPTER_PATH = dirpath
                print("搜到：", ADAPTER_PATH)
                break
if ADAPTER_PATH is None:
    raise FileNotFoundError("v15 的 DPO 适配器不在任何候选路径，见上方挂载清单。kernel_sources 是否挂了训练 kernel 的最新成功版输出？")
print("使用适配器：", ADAPTER_PATH)
print("适配器配置：", json.load(open(os.path.join(ADAPTER_PATH, "adapter_config.json"))))
"""

CELL4 = """# 4. 底座评测：加载底座（未微调），贪心跑 25 题
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def _run_eval(tag):
    answers = {}
    for r in EVAL:
        prompt = (f"<|im_start|>system\\n{SYS}<|im_end|>\\n"
                  f"<|im_start|>user\\n{r['question']}<|im_end|>\\n<|im_start|>assistant\\n")
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                             do_sample=False)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        ans = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        answers[r["id"]] = ans
        print(f"\\n===== [{tag}] {r['id']} =====\\n问：{r['question']}\\n答：{ans}\\n")
        print(f"（生成 token 数：{len(gen_ids)}，答案字数：{len(ans)}）")
    torch.cuda.empty_cache()
    return answers

print("########## 底座开始评测 ##########")
base_answers = _run_eval("底座")
print("底座评测完成。")
del model
torch.cuda.empty_cache()
print("底座模型已释放。")
"""

CELL5 = """# 5. v15（DPO 后）评测：加载 DPO 适配器（与 v15 第 4 格相同做法），贪心跑 25 题
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("DPO 适配器加载完成（可训练参数：", _trainable, "；评测只用推理，不训练）")
FastLanguageModel.for_inference(model)

print("########## v15（DPO 后）开始评测 ##########")
dpo_answers = _run_eval("v15-DPO")
print("v15 评测完成。")
"""

CELL6 = """# 6. 自动打分表 + 保存结果
import json as _json

def _auto(ans, r):
    ok_must = all(s in ans for s in r["must_contain"])
    ok_not = all(s not in ans for s in r["must_not_contain"])
    manual = (len(r["must_contain"]) == 0 and len(r["must_not_contain"]) == 0)
    return ok_must and ok_not, manual

print("\\n===== 自动打分表（✓=自动通过，✗=自动不通过，?=须人工读答案） =====")
print(f"{'题号':<5}{'类别':<12}{'底座':<6}{'v15':<6}")
tot_base = tot_dpo = 0
for r in EVAL:
    b_ok, b_man = _auto(base_answers[r["id"]], r)
    d_ok, d_man = _auto(dpo_answers[r["id"]], r)
    b_s = "?" if b_man else ("✓" if b_ok else "✗")
    d_s = "?" if d_man else ("✓" if d_ok else "✗")
    if b_ok and not b_man:
        tot_base += 1
    if d_ok and not d_man:
        tot_dpo += 1
    print(f"{r['id']:<5}{r['category']:<12}{b_s:<6}{d_s:<6}  {r['question'][:40]}")
print(f"\\n自动通过（不含须人工的题）：底座 {tot_base}/25，v15 {tot_dpo}/25")
print("注意：'?' 的题（must_contain 与 must_not_contain 均为空）自动检查不判分，须人工读答案；'✓' 也不等于全对，只表示必含的串都出现、禁用的串都没出现，最终结论以人工评分为准。")

os.makedirs("outputs", exist_ok=True)
results = {
    "eval_version": "heldout-v1",
    "questions": EVAL,
    "base_answers": base_answers,
    "dpo_answers": dpo_answers,
    "auto_score": {"base": tot_base, "dpo": tot_dpo, "total": len(EVAL)},
}
with open("outputs/heldout-eval-v1-results.json", "w", encoding="utf-8") as f:
    _json.dump(results, f, ensure_ascii=False, indent=2)
print("\\n结果已存到 outputs/heldout-eval-v1-results.json")
print("=== outputs/ 全部文件 ===")
for _root, _dirs, _files in os.walk("outputs"):
    for _f in _files:
        print(os.path.join(_root, _f))
print("完成。")
"""

def set_cell(cells, i, source):
    cells[i]["source"] = source.splitlines(keepends=True)
    cells[i]["outputs"] = []
    cells[i]["execution_count"] = None

nb = {
    "cells": [],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.10.0"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}
nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(keepends=True)})
nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                    "outputs": [],
                    "source": "# 1. 安装（Kaggle Notebook 里跑一次即可）\n%pip install -q \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n%pip install -q trl peft accelerate bitsandbytes datasets\n".splitlines(keepends=True)})
for src in [CELL2, CELL3, CELL4, CELL5, CELL6]:
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                        "source": src.splitlines(keepends=True)})

# embed the eval set into cell 2
c2 = "".join(nb["cells"][2]["source"])
c2 = c2.replace("__EVAL_JSON__", json.dumps(EVAL, ensure_ascii=False))
nb["cells"][2]["source"] = c2.splitlines(keepends=True)

json.dump(nb, open("qwen2.5-7b-eval-heldout.ipynb", "w"), ensure_ascii=False, indent=1)
print("wrote qwen2.5-7b-eval-heldout.ipynb")

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        _py = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("%", "!")))
        compile(_py, f"cell-{i}", "exec")
        print(f"cell {i}: compile OK ({len(src)} chars)")
print("all code cells compile OK")

meta = {
    "id": "norsky/officialdom-qwen2-5-7b-eval-heldout",
    "title": "Officialdom Qwen2.5-7B held-out eval",
    "code_file": "qwen2.5-7b-eval-heldout.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": ["norsky/officialdom-qwen2-5-7b-sft-batch-1"],
}
json.dump(meta, open("kernel-metadata.json", "w"), ensure_ascii=False, indent=2)
print("wrote kernel-metadata.json")
