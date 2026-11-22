#!/usr/bin/env python3
"""Build the Qwen3-8B zero-shot baseline kernel from data/heldout-eval-v1.jsonl.

Purpose (decided 2026-11-12, not yet run): before switching the officialdom
model's base from Qwen2.5-7B to Qwen3-8B, run the NEW base zero-shot on the
same 25 held-out eval v1 questions and compare against the Qwen2.5-7B-Instruct
zero-shot baseline (manual 10.5/25, auto 12/25). Switch only if clearly
better; otherwise stay on Qwen2.5 and spend the effort on data v2.1 + RAG.

Design, kept identical to eval v1 (kaggle-kernel-eval-heldout) so the numbers
are comparable: same 25 questions (embedded in the notebook, no dataset
mount), same system prompt, same ChatML prompt construction, greedy decoding
(do_sample=False, max_new_tokens 320), same must_contain / must_not_contain
auto-scoring. Only the model changes.

Qwen3 differences handled here (verified 2026-11-13 via public docs):
- Thinking is ON by default for Qwen3. The soft switch "/no_think" at the end
  of the user message turns it off per turn; the reply then still carries an
  empty <think> block, which the run function strips before scoring/printing.
  Non-thinking mode is the right comparator: eval v1's base answered directly,
  with no reasoning traces.
- Qwen's model card warns against greedy decoding in *thinking* mode (endless
  repetition). We run non-thinking mode, where greedy is the eval-v1
  convention. If answers still show repetition, the notebook header says so
  and the fallback is a sampled re-run (temperature 0.7, top_p 0.8, top_k 20
  per the Qwen3 card) — report both, do not silently swap.
- Model id (corrected 2026-11-14 after v1 failed): Unsloth drops "-Instruct"
  for Qwen3 — there is NO unsloth/Qwen3-8B-Instruct(-bnb-4bit) repo.
  unsloth/Qwen3-8B IS the instruct model (verified: its tokenizer_config has
  a ChatML chat_template; unsloth/Qwen3-8B-Base has none). So the first
  candidate is unsloth/Qwen3-8B-bnb-4bit, then the
  unsloth/Qwen3-8B-unsloth-bnb-4bit variant, then Qwen/Qwen3-8B-Instruct
  with load_in_4bit=True (Unsloth quantizes on load). Either way the run
  prints the resolved id. v1 also failed on the official repo with the same
  "not a valid model identifier" error, which may have been a transient
  Kaggle-to-HF resolution failure; if v2 fails at the same spot, that is the
  next hypothesis to test (weights as a Kaggle dataset, or Colab).
"""
import json

EVAL = [json.loads(l) for l in open("../data/heldout-eval-v1.jsonl", encoding="utf-8") if l.strip()]
assert len(EVAL) == 25, len(EVAL)

MD = """## Qwen3-8B 零样本基线：held-out 评测 v1 同题复测

- **目的**：换底座之前先验证。同一套 25 道 held-out v1 新题（见 `data/heldout-eval-v1.jsonl`，与评测 v1 完全相同），只跑 Qwen3-8B-Instruct 零样本，不加载任何适配器。
- **对照**：评测 v1 里 Qwen2.5-7B-Instruct 零样本是人工 10.5/25、自动 12/25；v15（DPO 后）人工 13/25。**判断规则**：Qwen3-8B 明显高于 10.5/25 再谈换底座；不高就留在 Qwen2.5，把力气花在数据 v2.1 和 RAG 上。
- **可比性**：题目、system 提示、ChatML 拼法、贪心解码（do_sample=False，max_new_tokens 320）、自动打分规则，全部与评测 v1 相同，只换模型。
- **Qwen3 思考模式**：默认开启，本 kernel 用 `/no_think` 关掉（非思考模式作答，与评测 v1 的直接作答对齐）；输出里残留的空 `<think>` 块在打分前去掉，并计数。Qwen 卡片警告思考模式下不要用贪心解码——我们跑的是非思考模式；若答案仍出现重复，记下来并按卡片推荐采样（temperature 0.7、top_p 0.8、top_k 20）重跑一遍，两个结果都报。
- **预期**：约 10 分钟（只跑一个模型）。结果存 `outputs/qwen3-8b-zero-shot-baseline-results.json`。
- **v2 修正（2026-11-14）**：v1 用的 `unsloth/Qwen3-8B-Instruct-bnb-4bit` 在 HF 上不存在（404），约 2.5 分钟即失败；Qwen3 的 Unsloth 命名不带 `-Instruct`，Instruct 版即 `unsloth/Qwen3-8B`（已核 chat template），4-bit 构建为 `unsloth/Qwen3-8B-bnb-4bit`。v2 以此为准，另加 `-unsloth-bnb-4bit` 变体与官方库作候选。
"""

CELL2 = """# 2. 配置与评测题（题目硬编码在笔记本里，不走数据集挂载，避开路径问题）
MODEL_CANDIDATES = [
    "unsloth/Qwen3-8B-bnb-4bit",          # 首选：Unsloth 4-bit 构建（Qwen3 的 Unsloth 命名不带 -Instruct）
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",  # 备选：同一模型的另一 4-bit 变体
    "Qwen/Qwen3-8B-Instruct",             # 兜底：官方 Instruct，加载时 4-bit 量化
]
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 320
SYS = "你是一位精通中国历代官制的学者，回答要准确、克制，有出处意识，不确定的地方明说不确定。"
EVAL = __EVAL_JSON__
print("评测题：", len(EVAL), "道")
from collections import Counter as _Counter
print("分类：", dict(_Counter(r["category"] for r in EVAL)))
"""

CELL3 = """# 3. 加载 Qwen3-8B-Instruct（零样本，不加载任何适配器）
from unsloth import FastLanguageModel
import torch

MODEL_NAME = None
_last_err = None
for _cand in MODEL_CANDIDATES:
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=_cand,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        MODEL_NAME = _cand
        print("加载成功：", MODEL_NAME)
        break
    except Exception as e:  # noqa: BLE001 - 候选模型逐个试，记下最后一个错误
        _last_err = e
        print("加载失败：", _cand, "->", str(e)[:200])
if MODEL_NAME is None:
    raise RuntimeError(f"三个候选模型都加载失败，最后一个错误：{_last_err}")
FastLanguageModel.for_inference(model)
"""

CELL4 = """# 4. 零样本评测：贪心跑 25 题（与评测 v1 相同的提示与解码）
import re

def _strip_think(ans):
    # Qwen3 用 /no_think 后仍可能带一个空 <think> 块；去掉它再打分。
    # 若 <think> 未闭合（生成在思考中截断），答案视为无有效内容。
    if "<think>" in ans:
        _pre, _sep, _rest = ans.partition("<think>")
        if "</think>" in _rest:
            ans = _rest.split("</think>", 1)[1]
        else:
            ans = ""
    return ans.strip()

def _run_eval(tag):
    answers = {}
    think_count = 0
    for r in EVAL:
        prompt = (f"<|im_start|>system\\n{SYS}<|im_end|>\\n"
                  f"<|im_start|>user\\n{r['question']} /no_think<|im_end|>\\n<|im_start|>assistant\\n")
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
                             do_sample=False)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "<think>" in raw:
            think_count += 1
        ans = _strip_think(raw).strip()
        answers[r["id"]] = ans
        print(f"\\n===== [{tag}] {r['id']} =====\\n问：{r['question']}\\n答：{ans}\\n")
        print(f"（生成 token 数：{len(gen_ids)}，答案字数：{len(ans)}，含 think 块：{'是' if '<think>' in raw else '否'}）")
    torch.cuda.empty_cache()
    print(f"\\n含 <think> 块的答案：{think_count}/25")
    return answers, think_count

print("########## Qwen3-8B 零样本开始评测 ##########")
qwen3_answers, qwen3_think_count = _run_eval("Qwen3-8B")
print("Qwen3-8B 评测完成。")
"""

CELL5 = """# 5. 自动打分表 + 保存结果（对照评测 v1 的 Qwen2.5-7B-Instruct 基线）
import json as _json

def _auto(ans, r):
    ok_must = all(s in ans for s in r["must_contain"])
    ok_not = all(s not in ans for s in r["must_not_contain"])
    manual = (len(r["must_contain"]) == 0 and len(r["must_not_contain"]) == 0)
    return ok_must and ok_not, manual

print("\\n===== 自动打分表（✓=自动通过，✗=自动不通过，?=须人工读答案） =====")
print(f"{'题号':<5}{'类别':<12}{'Qwen3-8B':<9}")
tot_qwen3 = 0
for r in EVAL:
    q_ok, q_man = _auto(qwen3_answers[r["id"]], r)
    q_s = "?" if q_man else ("✓" if q_ok else "✗")
    if q_ok and not q_man:
        tot_qwen3 += 1
    print(f"{r['id']:<5}{r['category']:<12}{q_s:<9}  {r['question'][:40]}")
print(f"\\n自动通过（不含须人工的题）：Qwen3-8B {tot_qwen3}/25")
print("对照：评测 v1 Qwen2.5-7B-Instruct 零样本自动 12/25、人工 10.5/25；v15 人工 13/25。")
print("注意：'?' 的题（must_contain 与 must_not_contain 均为空）自动检查不判分，须人工读答案；'✓' 也不等于全对，只表示必含的串都出现、禁用的串都没出现，最终结论以人工评分为准。")
print("若答案出现重复或思考痕迹，按笔记本头的说明用采样重跑一遍，两个结果都报，不要只报好的那个。")

os.makedirs("outputs", exist_ok=True)
results = {
    "eval_version": "heldout-v1",
    "model": MODEL_NAME,
    "mode": "zero-shot, greedy, non-thinking (/no_think)",
    "questions": EVAL,
    "qwen3_answers": qwen3_answers,
    "think_block_count": qwen3_think_count,
    "auto_score": {"qwen3_8b": tot_qwen3, "total": len(EVAL)},
    "baseline_for_comparison": {
        "qwen2.5-7b-instruct zero-shot": {"auto": 12, "manual": 10.5, "total": 25},
        "v15 dpo": {"manual": 13, "total": 25},
    },
}
with open("outputs/qwen3-8b-zero-shot-baseline-results.json", "w", encoding="utf-8") as f:
    _json.dump(results, f, ensure_ascii=False, indent=2)
print("\\n结果已存到 outputs/qwen3-8b-zero-shot-baseline-results.json")
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
for src in [CELL2, CELL3, CELL4, CELL5]:
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [],
                        "source": src.splitlines(keepends=True)})

# embed the eval set into cell 2
c2 = "".join(nb["cells"][2]["source"])
c2 = c2.replace("__EVAL_JSON__", json.dumps(EVAL, ensure_ascii=False))
nb["cells"][2]["source"] = c2.splitlines(keepends=True)

json.dump(nb, open("qwen3-8b-eval-heldout-baseline.ipynb", "w"), ensure_ascii=False, indent=1)
print("wrote qwen3-8b-eval-heldout-baseline.ipynb")

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        _py = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("%", "!")))
        compile(_py, f"cell-{i}", "exec")
        print(f"cell {i}: compile OK ({len(src)} chars)")
print("all code cells compile OK")

meta = {
    "id": "norsky/officialdom-qwen3-8b-zero-shot-baseline",
    "title": "Officialdom Qwen3-8B zero-shot baseline",
    "code_file": "qwen3-8b-eval-heldout-baseline.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
json.dump(meta, open("kernel-metadata.json", "w"), ensure_ascii=False, indent=2)
print("wrote kernel-metadata.json")
