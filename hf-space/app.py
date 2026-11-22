"""公共管理决策大模型 v15 — Hugging Face Space（Gradio + ZeroGPU）.

模型文件（GGUF）放在同账号下的私有模型仓库里，Space 启动时用 HF_TOKEN
（Space secret）把它下载到本地磁盘；聊天页本身是公开的，但权重不公开。
模型跑在 CPU 上（llama-cpp-python 用 CPU wheel；CUDA 版在 ZeroGPU 容器里
缺 libcudart.so.12，导包就失败，见 2026-11-21 日志）。answer() 仍挂
@spaces.GPU——ZeroGPU 硬件要求启动时至少检测到一个 GPU 函数，否则直接判
Runtime error；模型本身不占 GPU，只是每次对话按时长消耗 GPU 时间。
"""

import os

import gradio as gr
import spaces
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "unimaster/officialdom-v15")
MODEL_FILE = os.environ.get(
    "MODEL_FILE", "Qwen2.5-7B-Instruct.Q4_K_M.gguf"
)
HF_TOKEN = os.environ.get("HF_TOKEN")  # Space secret, 用于读私有模型仓库

_llm = None
_model_path = None


def _prefetch_model():
    """Space 启动时（CPU 环境）先把 GGUF 下载到磁盘，别在 GPU 计时里下载."""
    global _model_path
    if _model_path is not None:
        return _model_path
    try:
        _model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID, filename=MODEL_FILE, token=HF_TOKEN
        )
        print(f"模型文件就绪: {_model_path}")
    except Exception as e:  # token 未设或仓库不对时先记下来，首次对话再报清楚
        print(f"模型预下载失败（首次对话时会重试）: {e}")
        _model_path = None
    return _model_path


# 启动即预下载，冷启动多花一两分钟，但每次对话不用再等下载。
_prefetch_model()

SYSTEM_PROMPT = (
    "你是一位熟悉中国官场规则、制度与决策程序的助手，帮助使用者分析具体情境、"
    "识别约束与风险、给出可执行的处理建议。回答用中文，条理清晰，引用制度依据时注明出处。"
)


def _load_model():
    global _llm
    if _llm is not None:
        return _llm
    from llama_cpp import Llama

    path = _prefetch_model()
    if path is None:
        raise RuntimeError(
            "模型文件下载失败：请检查 Space 的 HF_TOKEN secret 与 MODEL_REPO_ID 是否正确。"
        )
    # CPU-only wheel（requirements.txt 走 CPU wheel 索引）：n_gpu_layers=0，
    # 模型跑在 CPU 上。注意：Space 硬件是 ZeroGPU，启动时必须至少有一个
    # @spaces.GPU 装饰的函数，否则报 "No @spaces.GPU function detected"
    # 直接判 Runtime error——所以 answer() 仍挂 @spaces.GPU（它只满足启动检查，
    # 模型本身不占 GPU）。代价是每次对话按调用时长消耗每天 5 分钟的 GPU 时间
    # （短回答约 30 秒，约够 10 次/天）；想不限次数可把 Space 硬件改成免费的
    # cpu-basic（Settings → Hardware），再去掉装饰器。7B Q4 在 CPU 上约每秒
    # 几到十几 token，回答是流式输出，短问题十几二十秒能答完。
    _llm = Llama(model_path=path, n_gpu_layers=0, n_ctx=4096, verbose=False)
    return _llm


def _to_messages(history, message):
    """兼容 Gradio 新旧两种 history 格式（messages / tuples）。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        if isinstance(turn, dict):  # messages 格式
            messages.append(turn)
        else:  # tuples 格式：[user, assistant]
            user_msg, bot_msg = turn
            messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})
    return messages


@spaces.GPU(duration=120)
def answer(message, history):
    llm = _load_model()
    messages = _to_messages(history, message)
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.3,
        top_p=0.9,
        repeat_penalty=1.15,
        max_tokens=512,
        stream=True,
    )
    partial = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            partial += delta["content"]
            yield partial


demo = gr.ChatInterface(
    fn=answer,
    title="公共管理决策大模型（演示）",
    description=(
        "基于 Qwen2.5-7B 微调的官场决策支持模型 v15（DPO）。"
        "免费演示：ZeroGPU 每天只有 5 分钟 GPU 时间，请简短提问。"
    ),
    examples=[
        "某县遭遇水灾，作为县长，赈灾工作应如何组织？",
        "一名官员在哪些情形下需要回避？具体程序是什么？",
        "2026 年赤字率比 2025 年变化了多少？",
    ],
)

with demo:
    gr.Markdown(
        "<p style='text-align:center;font-size:13px;color:#6b7280'>"
        "西南交通大学数字治理研究中心；西南交通大学智能政策实验室<br>"
        "联系方式：<a href='mailto:crip@swjtu.edu.cn'>crip@swjtu.edu.cn</a></p>"
    )

if __name__ == "__main__":
    demo.launch()
