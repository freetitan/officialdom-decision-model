#!/usr/bin/env python3
"""Build v15 (DPO-only) notebook from the v14 notebook.

v14 failed because DPO_PATH was hardcoded to the old-style mount path and the
DPO cell had no fallback. v15 skips SFT entirely: it loads the SFT adapter
v14 already saved (checkpoint-624, via kernel_sources self-input) and runs
only DPO (48 steps) + self-test + GGUF export (~30 min).
"""
import json

SRC = "qwen2.5-7b-sft-batch1.ipynb"
nb = json.load(open(SRC))
cells = nb["cells"]
assert len(cells) == 11, len(cells)

MD_V15 = """## v15：只跑 DPO（约半小时），不重跑 SFT

- **背景**：v14 的 SFT 阶段已跑完（624 步，loss 0.6497），但 DPO 阶段第一步就因 DPO 数据路径写错而失败（FileNotFoundError，DPO 0 步）。v14 的 SFT 适配器已完整保存在 `outputs/officialdom-qwen2.5-7b-sft-v6-dpo-sft/checkpoint-624`。
- **本轮做法**：跳过 SFT，直接加载 v14 存下的 SFT 适配器（本内核上一版的输出，以 kernel_sources 挂载），只跑 DPO（192 对，48 步）+ 自测 15 题 + GGUF 导出。
- **先看自测**：与 v11/v13 相同的 15 题，重点看 Q10（2026 年赤字率）终于答不答 2026 自己的数字、Q14 还倒不倒退、Q2/Q9 的法条还对不对。
- **史实要准**：下一步是加 RAG——史料切块入库，问答时先检索原文再回答；微调管风格与路子，RAG 管出处。
"""

CELL2 = """# 2. 配置：只改这里
#    v15：只跑 DPO，不重跑 SFT。SFT 适配器用 v14 存下的 checkpoint-624（本内核上一版输出，kernel_sources 挂载）。
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # 4-bit 底座，QLoRA 用（适配器配置里也指这个底座）
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

# v14 失败教训：路径不要只写一种。Kaggle 的挂载路径在不同版本里有两种写法
# （/kaggle/input/datasets/norsky/<slug>/ 与 /kaggle/input/<slug>/），v14 的 DPO 就是因为只写了旧式路径而失败。
# 下面所有输入都用“候选路径 + 全盘搜索”解析，找不到就大声报错、不猜。
SFT_ADAPTER_CANDIDATES = [
    "/kaggle/input/officialdom-qwen2-5-7b-sft-batch-1/outputs/officialdom-qwen2.5-7b-sft-v6-dpo-sft/checkpoint-624",
    "/kaggle/input/officialdom-qwen2-5-7b-sft-batch-1/officialdom-qwen2.5-7b-sft-v6-dpo-sft/checkpoint-624",
]
EVAL_CANDIDATES = [
    "/kaggle/input/datasets/norsky/officialdom-sft-batch6/sft-batch2-eval.jsonl",
    "/kaggle/input/officialdom-sft-batch6/sft-batch2-eval.jsonl",
    "/kaggle/input/officialdom-sft-batch-6/sft-batch2-eval.jsonl",
]
DPO_CANDIDATES = [
    "/kaggle/input/officialdom-dpo-round1/dpo-round1-pairs.jsonl",          # v14 实测挂载位置
    "/kaggle/input/datasets/norsky/officialdom-dpo-round1/dpo-round1-pairs.jsonl",  # 旧式写法
]
DPO_OUTPUT_DIR = "outputs/officialdom-qwen2.5-7b-sft-v6-dpo"  # v15：DPO 适配器与检查点（与 v14 同名，v14 没写出任何东西）
DPO_GGUF_DIR = "outputs/officialdom-qwen2.5-7b-sft-v6-dpo-gguf"  # v15：DPO 模型的 GGUF
"""

CELL3 = """# 3. 输入定位：先打印 /kaggle/input 的实际挂载清单，再解析评测集路径（只用于自测，不参与训练）
#    v15 不加载 SFT 训练数据（2,490 条），SFT 已在 v14 跑完。
import json, os

def _list_input(root="/kaggle/input"):
    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, "").count(os.sep)
        print("  " * level + os.path.basename(dirpath) + "/")
        for f in filenames:
            print("  " * (level + 1) + f)

def _resolve(candidates, filename, what):
    for p in candidates:
        print(("OK  " if os.path.exists(p) else "缺失") + " " + p)
        if os.path.exists(p):
            print("使用：", p)
            return p
    print(f"=== 候选路径全缺失，全盘搜索 {filename} ===")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if filename in filenames:
            p = os.path.join(dirpath, filename)
            print("搜到：", p)
            return p
    raise FileNotFoundError(f"{what}不在任何候选路径，见上方挂载清单。")

print("=== /kaggle/input 实际挂载内容 ===")
_list_input()
print("=== 评测集路径检查 ===")
EVAL_PATH = _resolve(EVAL_CANDIDATES, "sft-batch2-eval.jsonl", "batch-2 评测集")
EVAL_DIR = os.path.dirname(EVAL_PATH)
EVAL3_PATH = os.path.join(EVAL_DIR, "sft-batch3-eval.jsonl")
EVAL4_PATH = os.path.join(EVAL_DIR, "sft-batch4-eval.jsonl")
EVAL5_PATH = os.path.join(EVAL_DIR, "sft-batch5-eval.jsonl")
EVAL6_PATH = os.path.join(EVAL_DIR, "sft-batch6-eval.jsonl")
for _p in [EVAL3_PATH, EVAL4_PATH, EVAL5_PATH, EVAL6_PATH]:
    print(("OK  " if os.path.exists(_p) else "缺失") + " " + _p)

def load_examples(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
"""

CELL4 = """# 4. 加载底座 + v14 的 SFT 适配器（不重新训练 SFT）
#    Unsloth 做法：把适配器目录直接传给 FastLanguageModel.from_pretrained，
#    它按 adapter_config.json 里的底座名加载底座、再挂上适配器，返回的模型可继续训练。
#    适配器目录 = v14 输出的 checkpoint-624（含 adapter_model.safetensors、adapter_config.json、tokenizer 文件）。
#    找不到就大声报错；加载后核对可训练参数数（应为 40,370,176，与 v5–v14 一致）。
from unsloth import FastLanguageModel
import torch

print("=== SFT 适配器路径检查 ===")
ADAPTER_PATH = None
for p in SFT_ADAPTER_CANDIDATES:
    print(("OK  " if os.path.exists(os.path.join(p, "adapter_model.safetensors")) else "缺失") + " " + p)
    if os.path.exists(os.path.join(p, "adapter_model.safetensors")):
        ADAPTER_PATH = p
        break
if ADAPTER_PATH is None:
    print("=== 候选路径全缺失，全盘搜索 checkpoint-624/adapter_model.safetensors ===")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if os.path.basename(dirpath) == "checkpoint-624" and "adapter_model.safetensors" in filenames:
            ADAPTER_PATH = dirpath
            print("搜到：", ADAPTER_PATH)
            break
if ADAPTER_PATH is None:
    raise FileNotFoundError("v14 的 SFT 适配器（checkpoint-624）不在任何候选路径，见上方挂载清单。kernel_sources 是否挂了本内核上一版输出？")
print("使用适配器：", ADAPTER_PATH)
import json as _json
print("适配器配置：", _json.load(open(os.path.join(ADAPTER_PATH, "adapter_config.json"))))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("可训练参数：", _trainable)
assert _trainable > 0, "适配器加载后可训练参数为 0，DPO 无法训练——停止。"
print("期望 40,370,176；实际", _trainable, "一致" if _trainable == 40370176 else "（不一致，请核对）")

try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("gradient checkpointing 已开启")
except Exception as _e:
    print("gradient checkpointing 开启失败（继续）：", repr(_e))
"""

CELL5 = """# 5. 跳过 SFT（v14 已跑完，适配器已加载）
#    本格只做确认，不训练。DPO 在第 7 格。
print("SFT 已跳过：v14 的 SFT 适配器（checkpoint-624，624 步，loss 0.6497）已加载。")
print("下一步：DPO（192 对，48 步）。")
"""

CELL6 = """# 6. DPO 数据：192 对偏好（prompt / chosen / rejected），见 data/dpo-round1-report.md
#    prompt = ChatML 的 system+user 部分（到 "<|im_start|>assistant\\n" 为止）；
#    chosen / rejected = 答案 + "<|im_end|>"。
#    chosen 是核对过的正确答案，rejected 是模型在 v9–v13 自测里实际答错的同款错误（按日志片段重建）。
#    v15 修复：v14 在这里失败——DPO_PATH 只写了旧式路径。改用候选路径 + 全盘搜索。
import json as _json
from datasets import Dataset
print("=== DPO 数据路径检查 ===")
DPO_PATH = None
for p in DPO_CANDIDATES:
    print(("OK  " if os.path.exists(p) else "缺失") + " " + p)
    if os.path.exists(p):
        DPO_PATH = p
        break
if DPO_PATH is None:
    print("=== 候选路径全缺失，全盘搜索 dpo-round1-pairs.jsonl ===")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        if "dpo-round1-pairs.jsonl" in filenames:
            DPO_PATH = os.path.join(dirpath, "dpo-round1-pairs.jsonl")
            print("搜到：", DPO_PATH)
            break
if DPO_PATH is None:
    raise FileNotFoundError("DPO 偏好对文件不在任何候选路径，见上方挂载清单。数据集 slug 为 officialdom-dpo-round1。")
print("使用：", DPO_PATH)
DPO_SYS = "你是一位精通中国历代官制的学者，回答要准确、克制，有出处意识，不确定的地方明说不确定。"
def _dpo_prompt(instruction):
    return ("<|im_start|>system\\n" + DPO_SYS + "<|im_end|>\\n"
            f"<|im_start|>user\\n{instruction}<|im_end|>\\n"
            "<|im_start|>assistant\\n")
_dpo_rows = [_json.loads(l) for l in open(DPO_PATH, encoding="utf-8") if l.strip()]
print("DPO 偏好对：", len(_dpo_rows), "条")
from collections import Counter as _Counter
print("分类统计：", dict(_Counter(r["category"] for r in _dpo_rows)))
# 训练集只留 prompt / chosen / rejected 三列（其余列 DPO 不用）
dpo_dataset = Dataset.from_dict({
    "prompt": [_dpo_prompt(r["prompt"]) for r in _dpo_rows],
    "chosen": [r["chosen"] + "<|im_end|>" for r in _dpo_rows],
    "rejected": [r["rejected"] + "<|im_end|>" for r in _dpo_rows],
})
# 干跑验证：第一对的 prompt 必须是 prompt+chosen 的前缀（边界无 token 合并），答案以 <|im_end|> 结尾
_p0 = dpo_dataset[0]["prompt"]; _c0 = dpo_dataset[0]["chosen"]
_p_ids = tokenizer(_p0, add_special_tokens=False)["input_ids"]
_pc_ids = tokenizer(_p0 + _c0, add_special_tokens=False)["input_ids"]
print("干跑：第 1 对 prompt", len(_p_ids), "tokens，chosen", len(_pc_ids) - len(_p_ids), "tokens")
assert _pc_ids[:len(_p_ids)] == _p_ids, "DPO 边界不对：prompt 不是 prompt+chosen 的前缀"
assert _pc_ids[-1] == 151645, "chosen 结尾不是 <|im_end|>"
print("DPO 干跑通过：边界正确，chosen 以 <|im_end|> 结尾")
"""

def set_cell(i, source):
    cells[i]["source"] = source.splitlines(keepends=True)
    # clear stale outputs so the pushed notebook is clean
    cells[i]["outputs"] = []
    cells[i]["execution_count"] = None

set_cell(0, MD_V15)
set_cell(2, CELL2)
set_cell(3, CELL3)
set_cell(4, CELL4)
set_cell(5, CELL5)
set_cell(6, CELL6)
# cell 7: DPO training code unchanged; only update the header comment
c7 = "".join(cells[7]["source"])
c7 = c7.replace(
    "# 7. DPO 训练：从刚训完的 SFT 模型（同一会话内的 LoRA）继续，不重载底座。",
    "# 7. DPO 训练：从加载的 v14 SFT 适配器继续（第 4 格已加载），不重跑 SFT。",
)
cells[7]["source"] = c7.splitlines(keepends=True)
cells[7]["outputs"] = []
cells[7]["execution_count"] = None
set_cell(8, MD_V15)
# cells 1 (install), 9 (self-test), 10 (GGUF export) unchanged except clearing outputs
for i in (1, 9, 10):
    cells[i]["outputs"] = []
    cells[i]["execution_count"] = None

json.dump(nb, open(SRC, "w"), ensure_ascii=False, indent=1)
print("wrote", SRC)

# syntax-check every code cell (skip IPython magic lines like %pip)
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        _py = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("%", "!")))
        compile(_py, f"cell-{i}", "exec")
        print(f"cell {i}: compile OK ({len(src)} chars)")
print("all code cells compile OK")
