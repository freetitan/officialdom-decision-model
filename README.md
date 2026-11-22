# 公共管理决策大模型（Officialdom Decision Model）

A Chinese officialdom specialist model: Qwen2.5-7B fine-tuned with Unsloth QLoRA (SFT + DPO)
for decision support for leading cadres in complex officialdom situations — not factual recall.

以 Qwen2.5-7B 为基座、用 Unsloth QLoRA 做 SFT + DPO 微调的中文官场决策模型：
目标是在复杂官场情境中辅助领导干部决策，而不是回答史实知识。

## Honest results / 诚实的结果

On 25 freshly written held-out decision scenarios, human-scored (4-dimension rubric):

- Base model (Qwen2.5-7B-Instruct): 10.5 / 25
- This model (v15, SFT + DPO): 13 / 25

Known limitations, reported as-is:

- DPO gains do not generalize: fixes on self-test questions fail on new scenarios.
- Confidently wrong on some factual years (e.g. imperial-exam years from the CBDB data),
  where the base model refuses — more dangerous than a refusal for decision support.
- Inconsistent on the same fiscal figures across questions.
- Q10 still fails after five rounds of fixes; Q14 stays self-contradictory.

This is a proof of concept, not a production system. The next lever is retrieval (RAG),
not more SFT + DPO on the same data.

## Layout / 目录结构

- `officialdom-model/` — training code: the starter notebook, Kaggle kernels for SFT,
  held-out evaluation, the Qwen3-8B baseline, and a smoke test. Kernel metadata files
  contain only pointers to private Kaggle datasets, not data.
- `hf-space/` — the Hugging Face Space (Gradio + ZeroGPU) that serves the model,
  including the Cloudflare Worker that fronts it under a custom domain.
- `demo-worker/` — a separate demo site (Cloudflare Worker + Workers AI Llama 3.3 70B)
  with keyword retrieval over training-material excerpts. The corpus itself is NOT in
  this repo; `build_corpus.py` rebuilds it from local training data, which is not included.

## Data statement / 数据声明

**No training data and no model weights are included in this repo.**
Training data (8.9 GB, including classical novels, CBDB records, government documents,
and a contemporary novel corpus used for training only) and the 4.68 GB GGUF weights
live elsewhere. The weights are served from a private Hugging Face repository.
Nothing in this repo can reproduce the weights without the data.

## Reproduce on Kaggle / 在 Kaggle 上复现

1. Upload your training data as private Kaggle datasets and update the dataset slugs
   in the kernel metadata files.
2. Run `officialdom-model/kaggle-kernel-smoke-test/` first (one small step, CPU/GPU check).
3. Run the SFT kernel (`kaggle-kernel-sft-batch1/`), then the DPO step in `build_v15.py`.
   Reference hardware: Kaggle free tier, Tesla T4 x2.
4. Evaluate with `kaggle-kernel-eval-heldout/` (25 fresh questions, human scoring).

## Evaluation method / 评测方法

Held-out evaluation v1: 25 newly written decision scenarios, 5 categories x 5,
four-dimension human scoring (constraint identification, conflict surfacing,
actionable steps, accountability-risk flagging). Automated scoring was tried and
found misleading (it ranked the base model above the fine-tuned model, opposite
to human scores), so human scores are the reported numbers.

## Deployment / 部署结构

- Weights: private Hugging Face model repository, read by the Space via a token.
- Demo: public Gradio Space on ZeroGPU (`@spaces.GPU`, CPU inference, `n_gpu_layers=0`).
- Custom domain: Cloudflare Worker reverse proxy in `hf-space/cloudflare-worker.js`.
- The `demo-worker/` site is a separate Llama-based demo with RAG over training excerpts.

## License

MIT. See `LICENSE`.

## Contact

西南交通大学数字治理研究中心；西南交通大学智能政策实验室
crip@swjtu.edu.cn
