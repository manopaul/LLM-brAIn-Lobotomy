# 🧠 LLM-brAIn-Lobotomy: Model Weight Abliteration via ORVP

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

A lightweight, training-free toolkit to perform **Orthogonal Refusal-Vector Projection (ORVP)** on open-weight Large Language Models (LLMs).

Rather than relying on brittle input-level prompt jailbreaks, **abliteration** operates directly on the model's tensor weights, mathematically removing the learned "refusal direction" from the residual stream without requiring gradient updates, fine-tuning, or retraining.

---

## 🎬 Abliteration Demonstration

> [!NOTE]
> **Research Demonstration:** The video below demonstrates the **LLM Weight Abliteration Pipeline (ORVP)** in action that I was researching.

[github.com/user-attachments/assets/c720ff7e-f5da-4f27-a214-6dceadad81de](https://github.com/user-attachments/assets/c720ff7e-f5da-4f27-a214-6dceadad81de)

---

## 📌 Features

- **Dynamic Refusal Vector Extraction:** Calculates layer-wise means for harmful vs. harmless instructions using the **Difference of Means**.
- **Automatic Target Layer Selection:** Identifies the intermediate layer where refusal representations coalesce.
- **Full-Rank Weight Projection (ORVP):** Modifies both Self-Attention (`o_proj`) and MLP (`down_proj`) weight matrices to eliminate refusal activation energy.
- **Ollama & GGUF Integration:** Optional built-in pipeline to convert abliterated weights into GGUF format and load them directly into [Ollama](https://ollama.com/) for local side-by-side comparison.
- **Model Presets & Custom Support:** Ready-to-use presets for **Qwen 2**, **Phi-3 Mini**, **LLaMA 3**, and **Gemma 2**, plus support for any custom HuggingFace repository.

---

## ⚙️ Prerequisites

Before you start, ensure you have the following installed on your system:

1. **Python 3.10, 3.11, or 3.12** _(Python 3.13 is not yet recommended due to PyTorch wheel compatibility)._
2. **Git** (for downloading repository tools).
3. **[Ollama](https://ollama.com/)** _(Optional, required only for local model execution and side-by-side testing)._

---

## 🚀 Step-by-Step Setup & Usage

### Step 1: Clone the Repository & Create Environment

```bash
# Clone repository
git clone https://github.com/manopaul/LLM-brAIn-Lobotomy.git
cd LLM-brAIn-Lobotomy

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3: Run the Abliteration Script

You can run the script in **interactive mode** or pass command-line arguments directly.

#### Option A: Interactive Mode & Model Configuration

Model presets are dynamically loaded from [`models.json`](./models.json). Simply run the script to view available model presets or add your own:

```bash
python abliterate.py
```

- **Select a Preset:** Enter `1`, `2`, `3`, etc. to pick a pre-configured model.
- **Add a New Model Preset:** Select `+ Add New Preset` to enter a new Hugging Face Repo ID, friendly name, and description. It will automatically save to [`models.json`](./models.json) for future runs.
- **Custom One-Off:** Select `Custom One-Off` to abliterate a model without saving it to `models.json`.

##### Customizing `models.json` Manually

You can also directly edit [`models.json`](./models.json) to add custom models:

```json
{
  "5": {
    "name": "qwen2.5-3b",
    "repo": "Qwen/Qwen2.5-3B-Instruct",
    "desc": "Qwen 2.5 (3B Params, Gated: False)",
    "gated": false
  }
}
```

#### Option B: Direct CLI Execution (with Ollama Conversion)

Pass the model name and target name via CLI flags. Adding `--convert_ollama` will automatically convert the weights to GGUF and register an Ollama model:

```bash
# Example 1: Qwen 2 (1.5B) - Default 30 prompts
python abliterate.py --model Qwen/Qwen2-1.5B-Instruct --name qwen2 --convert_ollama

# Example 2: Microsoft Phi-3 Mini (3.8B) - Sample 50 prompts
python abliterate.py --model microsoft/Phi-3-mini-4k-instruct --name phi3 --num_prompts 50 --convert_ollama

# Example 3: Meta LLaMA 3 (8B) - Full 100 prompts calibration
export HF_TOKEN="your_huggingface_token_here"
python abliterate.py --model meta-llama/Meta-Llama-3-8B-Instruct --name llama3 --num_prompts 100 --convert_ollama

# Custom models config file path:
python abliterate.py --models_file ./my_custom_models.json
```

---

### Step 4: Converting Models & Running Locally on Ollama

You can convert both the **original downloaded model** and the **abliterated model** to GGUF format and register them in Ollama for side-by-side local testing.

#### Method A: Automatic Conversion via `abliterate.py`

Add the `--convert_ollama` flag when running `abliterate.py`. The pipeline will automatically compile the abliterated weights to GGUF, create an Ollama `Modelfile`, and register the model into Ollama as `abliterated_<name>`:

```bash
python abliterate.py --model Qwen/Qwen2-1.5B-Instruct --name qwen2 --convert_ollama
```

---

#### Method B: Converting Already-Downloaded or Already-Abliterated Models

If you have already downloaded or abliterated models in `./models/`, you can convert them directly to GGUF and import them into Ollama using the preset names defined in [`models.json`](./models.json).

##### 1. Prepare `llama.cpp` Conversion Tool (Once)

```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

##### 2. Conversion Commands for Preset Models in `models.json`

Select the command corresponding to the model name from `models.json`:

###### Preset 1: `qwen2-1.5b`

```bash
# Original Model
python llama.cpp/convert_hf_to_gguf.py ./models/original_qwen2-1.5b --outfile ./models/original_qwen2-1.5b.gguf --outtype f16
echo "FROM ./original_qwen2-1.5b.gguf" > ./models/Modelfile_original_qwen2-1.5b
ollama create original_qwen2-1.5b -f ./models/Modelfile_original_qwen2-1.5b

# Abliterated Model
python llama.cpp/convert_hf_to_gguf.py ./models/abliterated_qwen2-1.5b --outfile ./models/abliterated_qwen2-1.5b.gguf --outtype f16
echo "FROM ./abliterated_qwen2-1.5b.gguf" > ./models/Modelfile_abliterated_qwen2-1.5b
ollama create abliterated_qwen2-1.5b -f ./models/Modelfile_abliterated_qwen2-1.5b
```

###### Preset 2: `phi3-mini`

```bash
# Original Model
python llama.cpp/convert_hf_to_gguf.py ./models/original_phi3-mini --outfile ./models/original_phi3-mini.gguf --outtype f16
echo "FROM ./original_phi3-mini.gguf" > ./models/Modelfile_original_phi3-mini
ollama create original_phi3-mini -f ./models/Modelfile_original_phi3-mini

# Abliterated Model
python llama.cpp/convert_hf_to_gguf.py ./models/abliterated_phi3-mini --outfile ./models/abliterated_phi3-mini.gguf --outtype f16
echo "FROM ./abliterated_phi3-mini.gguf" > ./models/Modelfile_abliterated_phi3-mini
ollama create abliterated_phi3-mini -f ./models/Modelfile_abliterated_phi3-mini
```

###### Preset 3: `llama3-8b`

```bash
# Original Model
python llama.cpp/convert_hf_to_gguf.py ./models/original_llama3-8b --outfile ./models/original_llama3-8b.gguf --outtype f16
echo "FROM ./original_llama3-8b.gguf" > ./models/Modelfile_original_llama3-8b
ollama create original_llama3-8b -f ./models/Modelfile_original_llama3-8b

# Abliterated Model
python llama.cpp/convert_hf_to_gguf.py ./models/abliterated_llama3-8b --outfile ./models/abliterated_llama3-8b.gguf --outtype f16
echo "FROM ./abliterated_llama3-8b.gguf" > ./models/Modelfile_abliterated_llama3-8b
ollama create abliterated_llama3-8b -f ./models/Modelfile_abliterated_llama3-8b
```

###### Preset 4: `gemma2-2b`

```bash
# Original Model
python llama.cpp/convert_hf_to_gguf.py ./models/original_gemma2-2b --outfile ./models/original_gemma2-2b.gguf --outtype f16
echo "FROM ./original_gemma2-2b.gguf" > ./models/Modelfile_original_gemma2-2b
ollama create original_gemma2-2b -f ./models/Modelfile_original_gemma2-2b

# Abliterated Model
python llama.cpp/convert_hf_to_gguf.py ./models/abliterated_gemma2-2b --outfile ./models/abliterated_gemma2-2b.gguf --outtype f16
echo "FROM ./abliterated_gemma2-2b.gguf" > ./models/Modelfile_abliterated_gemma2-2b
ollama create abliterated_gemma2-2b -f ./models/Modelfile_abliterated_gemma2-2b
```

---

#### 3. Run Side-by-Side Local Testing

Test both models in terminal to evaluate the refusal removal:

```bash
# Test original baseline (standard safety refusals active)
ollama run original_qwen2-1.5b

# Test abliterated model (direct, decensored responses)
ollama run abliterated_qwen2-1.5b
```

---

## 📐 The Mathematics: How ORVP Works

Safety alignment (RLHF / DPO) causes an LLM to encode a single geometric vector in activation space corresponding to refusal intent:

1. **Mean Activations:**

$$
\mu_{\text{harmful}}^{(l)} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{h}_{\text{harmful}, i}^{(l)}
$$

$$
\mu_{\text{harmless}}^{(l)} = \frac{1}{M} \sum_{j=1}^{M} \mathbf{h}_{\text{harmless}, j}^{(l)}
$$

2. **Difference of Means (Refusal Vector):**

$$
\mathbf{r} = \mu_{\text{harmful}}^{(l_{\text{target}})} - \mu_{\text{harmless}}^{(l_{\text{target}})}
$$

3. **Normalization:**

$$
\hat{\mathbf{v}} = \frac{\mathbf{r}}{\|\mathbf{r}\|_2}
$$

4. **Orthogonal Projection on Weight Tensors:**

$$
\mathbf{W}_{\text{new}} = \left(\mathbf{I} - \hat{\mathbf{v}}\hat{\mathbf{v}}^T\right) \mathbf{W} = \mathbf{W} - \hat{\mathbf{v}} (\hat{\mathbf{v}}^T \mathbf{W})
$$

Because $\hat{\mathbf{v}}^T \mathbf{W}_{\text{new}} = \mathbf{0}$, the modified layers can no longer write activation energy in the direction of refusal for any input prompt.

---

## 📁 Repository Structure

```text
├── abliterate.py        # Core abliteration & deployment script
├── models.json          # Config file storing model presets & custom models
├── dataset/
│   ├── harmful.json     # Calibration dataset of harmful prompt representations
│   └── harmless.json    # Calibration dataset of harmless contrastive prompts
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules for weights & models
├── LICENSE              # Apache 2.0 License with Security Terms
├── NOTICE               # Apache 2.0 Section 4(d) Attribution Notice
└── README.md            # Documentation
```

---

## ⚠️ Responsible Use, Attribution & Liability Disclaimer

This repository is designed, developed, and open-sourced strictly for **authorized AI security research, adversarial red-teaming, mechanistic interpretability, and defensive safety auditing**.

### Terms of Use & Liability:

1. **User Responsibility & Liability:** By downloading, cloning, or executing this code, the user acknowledges and agrees that they assume full responsibility and legal liability for all actions, modified weights, model outputs, and downstream effects generated through the use of this software.
2. **Authorized Testing Only:** Removing safety guardrails modifies model behavior. All modified models should be tested strictly within **isolated, network-sandboxed environments**.
3. **No Warranty / Limitation of Liability:** This software is provided "as is" without warranty of any kind. Under no circumstances shall the author (Mano Paul) or contributors be held liable for any damages, security incidents, or claims arising from the use or misuse of this software.
4. **Vulnerability Triage Caution:** As documented in recent research (e.g., _clearbluejar_), abliterating refusal vectors can induce _Verdict Bias_ (sycophancy) in automated bug hunting pipelines. Abliterated models should not be used as unassisted judges in vulnerability detection.

---

## 📄 License & Attribution

This project is licensed under the **Apache License 2.0 with Security Research & Liability Terms** - see the [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE) files for details.

**Attribution Notice:** Any forks, redistribution, academic citations, or derived works must prominently attribute the original work to **Mano Paul**.

---

## 📚 References

1. **Zou, A., et al. (2023).** _“Representation Engineering: A Top-Down Approach to AI Transparency.”_ Center for AI Safety. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
2. **Arditi, A., et al. (2024).** _“Refusal in Language Models Is Mediated by a Single Direction.”_ [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
3. **Chen, Y., et al. (2025).** _“The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence.”_ [arXiv:2502.17420](https://arxiv.org/abs/2502.17420)
4. **Turner, A., et al. (2023).** _“Activation Addition: Steering Language Models Without Optimization.”_ [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
5. **Jiang, A., et al. (2024).** _“Mixtral of Experts.”_ [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
6. **clearbluejar. (2026).** _“Don’t Let Abliteration Abliterate Your Bug Hunting: Discovering Verdict Bias in Uncensored Models.”_ [clearbluejar.github.io](https://clearbluejar.github.io/posts/does-abliteration-skew-your-bug-hunting/)
7. **Labonne, M. (2024).** _“Uncensor any LLM with abliteration.”_ Hugging Face Blog. [huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration)
8. **Dai, J., et al. (2023).** _“Safe RLHF: Safe Reinforcement Learning from Human Feedback.”_ [arXiv:2310.12773](https://arxiv.org/abs/2310.12773)
9. **Kim, J., et al. (2026).** _“SafeMoE: Safe Fine-Tuning for MoE LLMs by Aligning Harmful Input Routing.”_ International Conference on Learning Representations (ICLR). [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2026/hash/34d3cf97696022b179171e5abda42c0b-Abstract-Conference.html)
10. **Liu, Q., et al. (2026).** _“Residual Stream Analysis of Overfitting and Structural Disruptions.”_ [arXiv:2603.13318](https://arxiv.org/abs/2603.13318)
