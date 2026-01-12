<div align="center">

# Baichuan-M3-235B

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/baichuan-inc/Baichuan-M3-235B)
[![M3 GPTQ-4bit](https://img.shields.io/badge/🤗%20M3%20GPTQ--4bit-Model-orange)](https://huggingface.co/baichuan-inc/Baichuan-M3-235B-GPTQ-INT4)
[![Tech Blog](https://img.shields.io/badge/📗%20Tech%20Blog-Blog-green)](https://www.baichuan-ai.com/blog/baichuan-M3)

<h4 align="center">
    <p>
        <a href="https://github.com/baichuan-inc/Baichuan-M3-235B/blob/main/README.md">中文</a> |
        <b>English</b>
    <p>
</h4>

**From Inquiry to Decision: Building Trustworthy Medical AI**

</div>

## 🌟 Model Overview

**Baichuan-M3** is Baichuan AI's new-generation medical-enhanced large language model, a major milestone following [Baichuan-M2](https://github.com/baichuan-inc/Baichuan-M2-32B).

In contrast to prior approaches that primarily focus on static question answering or superficial role-playing, Baichuan-M3 is trained to explicitly model the **clinical decision-making process**, aiming to improve usability and reliability in real-world medical practice. Rather than merely producing "plausible-sounding answers" or high-frequency vague recommendations like "you should see a doctor soon," the model is trained to **proactively acquire critical clinical information**, **construct coherent medical reasoning pathways**, and **systematically constrain hallucination-prone behaviors**.

### Core Highlights

- 🏆 **Surpasses GPT-5.2**: Outperforms OpenAI's latest model across HealthBench, HealthBench-Hard, hallucination evaluation, and BCOSCE, establishing a new SOTA in medical AI
- 🩺 **High-Fidelity Clinical Inquiry**: The only model to rank first across all three BCOSCE dimensions—Clinical Inquiry, Laboratory Testing, and Diagnosis
- 🧠 **Low Hallucination, High Reliability**: Achieves substantially lower hallucination rates than GPT-5.2 through Fact-Aware RL, even without external tools
- ⚡ **Efficient Deployment**: W4 quantization reduces memory to 26% of original; Gated Eagle3 speculative decoding achieves 96% speedup


## 📊 Performance

### HealthBench & Hallucination Evaluation

HealthBench is OpenAI's authoritative medical benchmark, constructed by 262 practicing physicians from 60 countries, comprising 5,000 high-fidelity multi-turn clinical conversations.

<div align="center">
  <img src="images/hb_metrics.png" alt="HealthBench Performance" width="80%">
</div>

Compared to Baichuan-M2, **Baichuan-M3 improves by 28 percentage points on HealthBench-Hard**, reaching 44.4 and surpassing GPT-5.2. It also ranks first on the HealthBench Total leaderboard.

For hallucination evaluation, we decompose long-form responses into fine-grained, verifiable atomic medical claims and validate each against authoritative medical evidence. **Even in a tool-free setting, Baichuan-M3 exhibits lower hallucination rates than GPT-5.2.**

### BCOSCE Evaluation

BCOSCE (BaiChuan Objective Structured Clinical Examination) is our end-to-end clinical decision-making benchmark that simulates the complete clinical workflow from patient encounter to final diagnosis, evaluating models' high-fidelity clinical inquiry capabilities through three stations: History Taking, Ancillary Investigations, and Final Diagnosis.

<div align="center">
  <img src="images/osce_metrics.png" alt="BCOSCE Performance" width="80%">
</div>

Baichuan-M3 **ranks first across all three core dimensions**, outperforming the second-best model by 12.4 points in Clinical Inquiry.

> 📢 The BCOSCE benchmark will be open-sourced soon. Stay tuned.


## 🔬 Technical Features

> 📖 For detailed technical information, please refer to: [Tech Blog](https://www.baichuan-ai.com/blog/baichuan-M3)

### SPAR: Segmented Pipeline Reinforcement Learning

To address reward sparsity and credit assignment challenges in long clinical interactions, we propose **SPAR (Step-Penalized Advantage with Relative baseline)**: it decomposes clinical workflows into four stages—history taking, differential diagnosis, laboratory testing, and final diagnosis—each with independent rewards, combined with OSCE process-level rewards for precise credit assignment, driving the model to construct auditable and complete decision logic.

<div align="center">
  <img src="images/SPAR_schema.jpeg" alt="SPAR Schema" width="80%">
</div>

### Fact-Aware Reinforcement Learning

By integrating factual verification directly into the RL loop, we build an online hallucination detection module that validates model-generated medical claims against authoritative medical evidence in real-time, supported by efficient caching mechanisms for online RL training. A dynamic reward aggregation strategy adaptively balances task learning and factual constraints based on the model's capability stage, significantly enhancing medical factual reliability without sacrificing reasoning depth.

<div align="center">
  <img src="images/FactAwareRL_schema.png" alt="Fact-Aware RL Schema" width="80%">
</div>

### Efficient Training and Inference

Adopts a **three-stage multi-expert fusion** training paradigm (Domain-Specific RL → Offline Distillation → MOPD), combined with **Gated Eagle3 speculative decoding** (96% speedup) and **W4 quantization** (only 26% memory) for efficient deployment.

<div align="center">
  <img src="images/train_pipeline.png" alt="Training Pipeline" width="80%">
</div>


## 🔧 Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-M3-235B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-M3-235B")

messages = [{"role": "user", "content": "I've been having headaches lately, especially worse in the afternoon. What should I do?"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    thinking_mode='on'
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768,
    temperature=0.6
)
response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
print(response)
```

### Deployment

Create an OpenAI-compatible API endpoint using `sglang>=0.4.6.post1` or `vllm>=0.9.0`:

```shell
# SGLang
python -m sglang.launch_server --model-path baichuan-inc/Baichuan-M3-235B --reasoning-parser qwen3

# vLLM
vllm serve baichuan-inc/Baichuan-M3-235B --reasoning-parser qwen3
```

### MTP Speculative Decoding

```shell
python3 -m sglang.launch_server \
    --model baichuan-inc/Baichuan-M3-235B \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path baichuan-inc/Baichuan-M3-235B/draft \
    --speculative-num-steps 6 \
    --speculative-eagle-topk 10 \
    --speculative-num-draft-tokens 32 \
    --mem-fraction 0.9 \
    --reasoning-parser qwen3 \
    --dtype bfloat16
```


## ⚠️ Usage Notices

1. **Medical Disclaimer**: For research and reference only; cannot replace professional medical diagnosis or treatment
2. **Intended Use Cases**: Medical education, health consultation, clinical decision support
3. **Safe Use**: Recommended under guidance of medical professionals


## 📄 License

Licensed under the [Apache License 2.0](LICENSE). Research and commercial use permitted.

## 🤝 Acknowledgements

- Base Model: Qwen3
- Training Framework: verl
- Inference Engines: vLLM, SGLang

Thank you to the open-source community. We commit to continuous contribution and advancement of healthcare AI.

## 📞 Contact Us

- Official Website: [Baichuan AI](https://www.baichuan-ai.com)
- Technical Support: [GitHub](https://github.com/baichuan-inc)


<div align="center">

**Advancing Medical AI from "Answering Correctly" to "Supporting Decisions"**

</div>

## 📚 Citation

```bibtex
@misc{baichuan-m3,
    title={Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making},
    author={Baichuan M3 Team},
    year={2025},
    url={https://github.com/baichuan-inc/Baichuan-M3-235B},
}
```
