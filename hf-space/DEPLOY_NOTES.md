# HF Space 部署笔记（做官模型 v15）

## 架构

```
用户（国内，无 VPN）
  → 自定义域名（Cloudflare）
  → Cloudflare Worker 反代（cloudflare-worker.js）
  → https://unimaster-officialdom-v15.hf.space（公开 Space）
  → Gradio 聊天页（app.py）
  → 私有模型仓库 unimaster/officialdom-v15 里的 GGUF（HF_TOKEN secret 读取）
```

- **模型权重始终私有**：公开的是 Space 聊天页，不是权重。Space 用 secret 里的
  token 去读私有仓库，访客看不到 token，也下不到权重。
- **换版本不用重建 Space**：新版本单独建一个模型仓库，Space 里改 `MODEL_REPO_ID`
  变量、重启，验证通过后再删旧仓库（见末尾）。

## 文件清单（本目录）

| 文件 | 用途 |
| --- | --- |
| `app.py` | Gradio 聊天页；ZeroGPU 上加载 GGUF、流式输出 |
| `requirements.txt` | llama-cpp-python（cu124 CUDA wheel）+ gradio 等 |
| `README.md` | Space 元数据（sdk: gradio） |
| `cloudflare-worker.js` | Worker 反代脚本，部署时只改 `BACKEND_HOST` 一行 |

## 部署步骤（拿到写权限 token 后）

1. **建私有模型仓库** `unimaster/officialdom-v15`（Settings → New Model，
   Private），上传 `Qwen2.5-7B-Instruct.Q4_K_M.gguf`（4.68GB）。
2. **创建 Space** `unimaster/officialdom-v15`：SDK 选 Gradio，硬件选
   **ZeroGPU**（免费档），把 `app.py`、`requirements.txt`、`README.md`
   三个文件传上去。
3. **设 Space secrets**（Settings → Variables and secrets）：
   - `HF_TOKEN`：读私有模型仓库的 token（部署用的写 token 就能读）。
   - `MODEL_REPO_ID`（可选）：默认 `unimaster/officialdom-v15`。
   - `MODEL_FILE`（可选）：默认 `Qwen2.5-7B-Instruct.Q4_K_M.gguf`。
4. **验证**：等构建完成，打开 Space，问一个短问题，看回答是否流式返回；
   再看 Space 日志里 llama.cpp 是否成功 offload 到 GPU。
5. **部署 Worker**：Cloudflare Dashboard → Workers & Pages → 新建 Worker，
   贴入 `cloudflare-worker.js`（改 `BACKEND_HOST`），Deploy 后在 Worker 的
   Domains 里绑定自定义域名。
6. **验证国内访问**：用国内网络（或请国内的人）打开自定义域名，测一次完整对话，
   重点看流式输出顺不顺、有没有中途断掉。

## 已知风险与备选

1. **CUDA wheel 不能用（2026-11-21 已证实）**：`llama-cpp-python==0.3.19` 的
   cu124 wheel 在 ZeroGPU 容器里导包即失败——`libcudart.so.12: cannot open
   shared object file`（容器里没有 CUDA 12 运行时库）。已改回 CPU wheel，并去掉
   `@spaces.GPU`、设 `n_gpu_layers=0`，模型跑在 CPU 上。cu124 的核还是 CUDA
   12.4 编译的，即便补上 libcudart，Blackwell（sm_120）也没有对应核，照样退回
   CPU，所以 CPU wheel 是正解，不是权宜之计。**注意**：CPU wheel 必须走
   `https://abetlen.github.io/llama-cpp-python/whl/cpu` 这个 extra-index-url
   （requirements.txt 已写）。直接从默认 PyPI 装 0.3.19 时 pip 找不到预编译轮子，
   会下载 59.6MB 源码包现场编译 llama.cpp，构建要多花十几分钟。代价是慢（7B Q4
   在 CPU 上每秒几到十几 token，短回答十几二十秒）；且每次 @spaces.GPU 调用仍按
   时长计入每天 5 分钟 ZeroGPU 额度（见第 3 条），并非不限次数。
2. **ZeroGPU 必须保留一个 @spaces.GPU 函数**：2026-11-21 曾把装饰器去掉想省
   GPU 时间，结果 Space 启动即判 Runtime error——"No @spaces.GPU function
   detected during startup"。ZeroGPU 硬件要求启动时至少检测到一个 GPU 函数，
   与模型跑不跑 GPU 无关。已把装饰器加回（模型仍是 CPU 推理）。不要改纯 CPU：2026-11-21 实测，免费档 cpu-basic 不覆盖 Gradio Space，删 Space 重建时报 402，需 PRO 订阅；免费用户既不能降级也不能新建 cpu-basic 的 Gradio Space。此路已证伪，不得再试；Space 已还原为 ZeroGPU。
3. **免费 ZeroGPU 每天 5 分钟 GPU 时间**：CPU 推理的每次对话仍按调用时长占
   GPU 时间（短回答约 30 秒，约够 10 次/天）。这是当前演示的上限。
3. **冷启动**：Space 闲置会休眠，首次打开要等几分钟（下载模型 + 加载）。
4. **Worker 免费档**：每天 10 万次请求，轻度演示够用。
5. **国内实测是必须的**：websocket 流式经 Worker 转发一般能用，但要实测顺畅度。

## ⚠️ token 吊销注意

部署用的写 token 不要在部署后立即吊销——Space 的 `HF_TOKEN` secret 靠它读模型，
吊销后 Space 就读不到权重了。如需吊销，先在 HF 上另建一个**只读** token，把 Space
的 `HF_TOKEN` secret 换成只读 token，重启验证通过后再吊销写 token。

## 换版本流程（v16 出来后）

1. 新版本单独建私有仓库，如 `unimaster/officialdom-v16`，上传新 GGUF。
2. Space 的 `MODEL_REPO_ID` 变量改成新仓库名，重启 Space。
3. 在 Space 里提问验证新版本正常。
4. 验证通过后，旧仓库在 Settings 里删除（可选，留着当存档也行）。
5. 回滚：把 `MODEL_REPO_ID` 改回旧仓库名即可。
