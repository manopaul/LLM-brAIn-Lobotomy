#!/usr/bin/env python3
"""
===============================================================================
Dynamic LLM Weight Abliteration Pipeline (Self-Contained Educational Script)
Technique: Orthogonal Refusal-Vector Projection (ORVP) in Representation Engineering

Author: Mano Paul
Copyright (c) 2026 Mano Paul. All Rights Reserved.
License: Apache License 2.0 with Security Research & Liability Terms (see LICENSE file)

ATTRIBUTION & DISCLAIMER NOTICE:
Any redistribution, fork, or derivative work must retain attribution to Mano Paul.
This tool is intended strictly for authorized AI security research and red-teaming.
The user assumes full responsibility and liability for compliance with applicable laws
and safe containment of any decensored model weights.
===============================================================================

What is Abliteration?
--------------------
Traditional safety alignment (like RLHF or DPO) conditions a language model to
refuse harmful requests by establishing a specific directional pathway in its
residual activation space (the "Refusal Direction"). When a model processes a
prompt deemed sensitive, its internal layer activations align with this vector,
triggering standard refusal tokens ("I cannot assist with that...").

Abliteration uses Linear Algebra to surgically project out and delete this
specific refusal vector directly from the model's weight matrices. The result is
a permanently decensored model that retains its general reasoning capabilities
without needing retraining or gradient updates.

Pipeline Steps:
--------------
1. Model Selection & Loading (Supports Hugging Face IDs or local directories).
2. Contrastive Dataset Ingestion (Harmful vs. Harmless prompts).
3. Residual Activation Extraction across all transformer layers.
4. Difference of Means Calculation (Isolating the Refusal Direction Vector).
5. Orthogonal Projection (ORVP) on Attention (o_proj) and MLP (down_proj) weights.
6. Saving Decensored Model Artifacts.
7. (Optional) GGUF Conversion & Direct Ollama Model Registration.
===============================================================================
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize rich console for clear, formatted terminal output
console = Console()

# -----------------------------------------------------------------------------
# Dynamic Model Presets Configuration (models.json)
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PRESETS = {
    "1": {
        "name": "qwen2-1.5b",
        "repo": "Qwen/Qwen2-1.5B-Instruct",
        "desc": "Qwen 2 (1.5B Params, Fast, Great baseline)",
        "gated": False,
    },
    "2": {
        "name": "phi3-mini",
        "repo": "microsoft/Phi-3-mini-4k-instruct",
        "desc": "Phi-3 Mini (3.8B Params, Strict Alignment)",
        "gated": False,
    },
    "3": {
        "name": "llama3-8b",
        "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        "desc": "Llama 3 (8B Params, Requires HF_TOKEN)",
        "gated": True,
    },
    "4": {
        "name": "gemma2-2b",
        "repo": "google/gemma-2-2b-it",
        "desc": "Gemma 2 (2.6B Params, Requires HF_TOKEN)",
        "gated": True,
    },
}


def load_model_presets(config_path: str = "models.json") -> dict:
    """
    Loads model presets from a JSON file.
    If the file does not exist, saves DEFAULT_MODEL_PRESETS to config_path and returns them.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[bold red]Warning: Could not parse '{config_path}': {e}. Falling back to default presets.[/bold red]")

    # Fallback to default presets and save to config_path
    save_model_presets(DEFAULT_MODEL_PRESETS, config_path)
    return DEFAULT_MODEL_PRESETS


def save_model_presets(presets: dict, config_path: str = "models.json"):
    """Saves updated model presets dictionary to a JSON configuration file."""
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
        console.print(f"[bold green]Updated model presets saved to '{config_path}'.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error saving presets to '{config_path}': {e}[/bold red]")



def download_model_if_needed(model_name_or_path: str, local_dir: str) -> str:
    """
    Ensures model weights are available locally.
    If the model already exists in local_dir or model_name_or_path is a local path,
    skips downloading and uses the local model directly.
    """
    # 1. Check if model_name_or_path is an explicit local directory on disk
    if os.path.isdir(model_name_or_path):
        console.print(f"[bold green]Using existing local model directory: '{model_name_or_path}'[/bold green]")
        return model_name_or_path

    # 2. Check if local_dir already exists and contains model weights or config
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        files_in_dir = os.listdir(local_dir)
        has_config = "config.json" in files_in_dir
        has_weights = any(
            f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt") or f.endswith(".gguf") or "index.json" in f
            for f in files_in_dir
        )
        
        if has_weights or has_config:
            console.print(f"[bold green]Model already exists locally at '{local_dir}'. Skipping download.[/bold green]")
            return local_dir

    # 3. Download snapshot from Hugging Face Hub if not present
    os.makedirs(local_dir, exist_ok=True)
    console.print(f"[bold cyan]Downloading '{model_name_or_path}' from Hugging Face to '{local_dir}'...[/bold cyan]")
    
    token = os.environ.get("HF_TOKEN")
    try:
        snapshot_download(
            repo_id=model_name_or_path,
            local_dir=local_dir,
            token=token
        )
        console.print(f"[bold green]Download complete![/bold green]")
        return local_dir
    except Exception as e:
        console.print(f"[bold red]Failed to download model: {e}[/bold red]")
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            console.print("[yellow]Hint: For gated models, export your token: export HF_TOKEN=your_token_here[/yellow]")
        sys.exit(1)



def load_datasets(dataset_dir: str, num_prompts: int = 30):
    """
    Loads harmful and harmless contrastive prompt pairs.
    Slices datasets to num_prompts (default: 30, min: 30, max: 100).
    
    Why contrastive pairs?
    ----------------------
    To isolate 'refusal' from general topic semantics, we compare activations from
    harmful instructions (which trigger refusal) against harmless instructions
    (which elicit helpful answers).
    """
    harmful_path = os.path.join(dataset_dir, "harmful.json")
    harmless_path = os.path.join(dataset_dir, "harmless.json")

    if not os.path.exists(harmful_path) or not os.path.exists(harmless_path):
        console.print(f"[bold red]Datasets not found in '{dataset_dir}'. Ensure harmful.json and harmless.json exist.[/bold red]")
        sys.exit(1)

    with open(harmful_path, "r", encoding="utf-8") as f:
        harmful_prompts = json.load(f)
    with open(harmless_path, "r", encoding="utf-8") as f:
        harmless_prompts = json.load(f)

    # Validate bounds (30 to 100)
    if num_prompts < 30 or num_prompts > 100:
        console.print(f"[bold yellow]Warning: --num_prompts ({num_prompts}) must be between 30 and 100. Clamping to nearest boundary.[/bold yellow]")
        num_prompts = max(30, min(100, num_prompts))

    # Slice dataset prompts to requested count
    harmful_prompts = harmful_prompts[:num_prompts]
    harmless_prompts = harmless_prompts[:num_prompts]

    return harmful_prompts, harmless_prompts



def get_hidden_states(model, tokenizer, prompts, desc="Processing"):
    """
    Performs forward passes over prompts and records the hidden state activations
    across all transformer layers at the final instruction token position.

    Mathematical Intuition:
    -----------------------
    In an autoregressive transformer, the final token of the prompt is where the
    model decides its next action (i.e., whether to comply or refuse). By extracting
    the hidden activation h at this exact position across all layers L, we obtain
    a high-dimensional geometric trace of the model's decision process.
    """
    all_hidden_states = []
    
    for prompt in tqdm(prompts, desc=desc):
        # Apply the model's official chat template (e.g., <|im_start|>user\n...<|im_end|>)
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback format if model has no chat template configured
            formatted_prompt = f"User: {prompt}\nAssistant:"

        # Tokenize and transfer prompt inputs to model device
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Forward pass with no gradients (inference mode)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of length (num_layers + 1):
        # index 0: initial embedding layer output
        # index 1..L: output of each transformer block layer
        #
        # For each layer, grab hidden state for batch index 0, last token position (-1):
        # Tensor shape per layer: [hidden_size] (e.g., 2048 or 4096 dimensions)
        last_token_hidden = [layer_hidden[0, -1, :].cpu() for layer_hidden in outputs.hidden_states]
        
        # Stack layers: shape -> [num_layers + 1, hidden_size]
        all_hidden_states.append(torch.stack(last_token_hidden))

    # Return full batch tensor: shape -> [num_prompts, num_layers + 1, hidden_size]
    return torch.stack(all_hidden_states)


def calculate_refusal_direction(model, harmful_hidden, harmless_hidden):
    """
    Calculates the Refusal Direction Vector using the Difference of Means.

    Mathematical Formulation:
    -------------------------
    1. Calculate mean of harmful prompt activations:
       μ_harmful = (1 / N) * Σ h_harmful[i]
       
    2. Calculate mean of harmless prompt activations:
       μ_harmless = (1 / M) * Σ h_harmless[j]
       
    3. Compute difference vector (raw refusal direction):
       r = μ_harmful - μ_harmless
       
    4. Target Layer Selection:
       The abstract semantic concept of refusal forms in the middle layers of the model.
       We sample r at layer (num_layers // 2).
       
    5. Unit Normalization:
       v_hat = r / ||r||_2  (so that ||v_hat|| = 1)
    """
    console.print("\n[bold cyan]🧠 Calculating Difference of Means across Residual Stream...[/bold cyan]")
    
    # Compute mean activation vectors along the prompt dimension (dim=0)
    # Resulting shapes: [num_layers + 1, hidden_size]
    mean_harmful = harmful_hidden.mean(dim=0)
    mean_harmless = harmless_hidden.mean(dim=0)
    
    # Compute the direction vector separating harmful from harmless thoughts
    diff_vector = mean_harmful - mean_harmless
    
    # Calculate Euclidean norm per layer to observe where divergence peaks
    norms = torch.norm(diff_vector, dim=1)

    # In modern transformers, middle layers represent abstract decision gates
    num_layers = len(model.model.layers)
    target_layer = num_layers // 2

    console.print(f"[bold green]🎯 Target Censorship Gate selected at Layer {target_layer} (Divergence Norm: {norms[target_layer].item():.4f})[/bold green]")

    # Extract refusal vector from the target layer and normalize to unit length (length = 1.0)
    refusal_dir = diff_vector[target_layer]
    refusal_dir = refusal_dir / torch.norm(refusal_dir)
    refusal_dir = refusal_dir.to(model.device)

    return refusal_dir


def perform_abliteration(model, refusal_dir):
    """
    Performs Orthogonal Refusal-Vector Projection (ORVP) on model weight tensors.

    Mathematical Formulation:
    -------------------------
    In PyTorch, a linear layer computes: y = x * W^T
    For attention output (o_proj) and MLP down-projection (down_proj), the output dimension
    corresponds to the residual stream where refusal is written.

    To permanently eliminate any activation along the refusal direction v_hat:
        W_new = (I - v_hat * v_hat^T) * W
              = W - v_hat * (v_hat^T * W)

    Proof that refusal is eliminated:
        y_new = x * (W_new)^T
              = x * W^T - (x * W^T * v_hat) * v_hat^T
              = y - (y · v_hat) * v_hat
        => v_hat · y_new = (v_hat · y) - (y · v_hat) * (v_hat · v_hat)
                         = (v_hat · y) - (v_hat · y) * (1)
                         = 0  (Strictly Zero for ANY input x!)
    """
    console.print("\n[bold cyan]✂️  Applying Orthogonal Refusal-Vector Projection (ORVP) across all layers...[/bold cyan]")

    for layer in tqdm(model.model.layers, desc="Projecting Weights"):
        # 1. Self-Attention Output Projection (o_proj)
        # o_proj is the gatekeeper that writes self-attention heads back to the residual stream.
        if hasattr(layer.self_attn, "o_proj"):
            attn_weight = layer.self_attn.o_proj.weight.data  # Shape: [hidden_size, in_features]
            v_ref = refusal_dir.to(device=attn_weight.device, dtype=attn_weight.dtype)
            
            # Compute rank-1 projection: v_ref * (v_ref^T * W)
            proj = torch.outer(v_ref, torch.matmul(v_ref, attn_weight))
            
            # Subtract projection to zero out refusal writing capability
            layer.self_attn.o_proj.weight.data = attn_weight - proj

        # 2. MLP Down Projection (down_proj)
        # Feed-forward networks also write intermediate concept vectors back to the residual stream.
        if hasattr(layer.mlp, "down_proj"):
            mlp_weight = layer.mlp.down_proj.weight.data  # Shape: [hidden_size, intermediate_size]
            v_ref = refusal_dir.to(device=mlp_weight.device, dtype=mlp_weight.dtype)
            
            # Compute rank-1 projection: v_ref * (v_ref^T * W_mlp)
            proj = torch.outer(v_ref, torch.matmul(v_ref, mlp_weight))
            
            # Subtract projection
            layer.mlp.down_proj.weight.data = mlp_weight - proj


def convert_and_deploy_ollama(model_dir: str, target_name: str, gguf_out_dir: str):
    """
    Automates conversion of Hugging Face weights to GGUF format via llama.cpp
    and imports the model into Ollama for local testing with clean status displays.
    """
    console.print(f"\n[bold cyan]🔄 Converting '{target_name}' to GGUF and registering in Ollama...[/bold cyan]")

    # Check/Clone llama.cpp conversion scripts if not present
    llama_cpp_dir = "./llama.cpp"
    if not os.path.exists(os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")):
        with console.status("[bold yellow]Cloning llama.cpp tools...[/bold yellow]", spinner="dots"):
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

    os.makedirs(gguf_out_dir, exist_ok=True)
    gguf_path = os.path.join(gguf_out_dir, f"{target_name}.gguf")

    console.print(f"[bold cyan]Compiling weights into GGUF: {gguf_path}[/bold cyan]")
    convert_cmd = [
        sys.executable,
        os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
        model_dir,
        "--outfile", gguf_path,
        "--outtype", "f16"
    ]

    # Run GGUF compilation with animated progress status (suppressing noisy output)
    try:
        with console.status(f"[bold cyan]Compiling tensors into GGUF format ({target_name})...[/bold cyan]", spinner="dots"):
            subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        console.print(f"[bold green]✅ GGUF compilation complete: '{gguf_path}'[/bold green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed GGUF compilation for '{target_name}':\n{e.stderr or e.stdout}[/bold red]")
        return

    # Create Ollama Modelfile
    modelfile_path = os.path.join(gguf_out_dir, f"Modelfile_{target_name}")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(f"FROM ./{target_name}.gguf\n")
        f.write("PARAMETER temperature 0.7\n")
        f.write("PARAMETER top_p 0.9\n")

    # If Ollama CLI is installed on this machine, create the model locally
    if shutil.which("ollama"):
        try:
            with console.status(f"[bold cyan]Registering '{target_name.lower()}' into Ollama...[/bold cyan]", spinner="dots"):
                subprocess.run(
                    ["ollama", "create", target_name.lower(), "-f", f"Modelfile_{target_name}"],
                    cwd=gguf_out_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
            console.print(f"[bold green]✅ Successfully registered in Ollama as '{target_name.lower()}'![/bold green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]❌ Failed to register model in Ollama:\n{e.stderr or e.stdout}[/bold red]")
    else:
        console.print("[yellow]Ollama CLI not found in PATH. GGUF was generated. Install Ollama to run locally.[/yellow]")


def main():
    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Abliterate refusal weights from an open-source LLM using ORVP.")
    parser.add_argument("--model", type=str, help="HuggingFace model ID (e.g., Qwen/Qwen2-1.5B-Instruct) or local directory")
    parser.add_argument("--name", type=str, help="Friendly name for output artifacts (e.g., qwen2, phi3)")
    parser.add_argument("--models_file", type=str, default="./models.json", help="Path to JSON file containing model presets")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Directory containing harmful.json and harmless.json")
    parser.add_argument("--num_prompts", type=int, default=30, help="Number of prompts to sample from datasets (range: 30 to 100, default: 30)")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory where model artifacts will be saved")
    parser.add_argument("--convert_ollama", action="store_true", help="Automatically convert abliterated model to GGUF and register in Ollama")
    args = parser.parse_args()


    # Banner Display
    console.print(Panel.fit(
        "[bold magenta]🧠 LLM Weight Abliteration Pipeline (ORVP)[/bold magenta]\n"
        "[dim]Representation Engineering: Surgical Removal of Refusal Vectors[/dim]",
        border_style="magenta"
    ))

    # Load presets from models.json (or config file)
    model_presets = load_model_presets(args.models_file)

    # Handle Interactive Model Selection if not specified via CLI flags
    if not args.model:
        table = Table(title="Select a Model Preset or Add Custom", show_header=True, header_style="bold cyan")
        table.add_column("Option", style="dim", width=8)
        table.add_column("Model Name", style="bold")
        table.add_column("Description")

        # Sort numeric keys logically
        sorted_keys = sorted(model_presets.keys(), key=lambda k: int(k) if k.isdigit() else k)
        for k in sorted_keys:
            v = model_presets[k]
            table.add_row(k, v["repo"], v["desc"])
        
        add_option_idx = str(len(sorted_keys) + 1)
        oneoff_option_idx = str(len(sorted_keys) + 2)

        table.add_row(add_option_idx, "[bold green]+ Add New Preset[/bold green]", "Prompt for model details and save to models.json")
        table.add_row(oneoff_option_idx, "[bold yellow]Custom One-Off[/bold yellow]", "Enter custom Hugging Face repo ID or local path (without saving)")
        console.print(table)

        choice = input(f"\nEnter choice [1-{oneoff_option_idx}] (default: 1): ").strip() or "1"

        if choice in model_presets:
            model_id = model_presets[choice]["repo"]
            friendly_name = model_presets[choice]["name"]
        elif choice == add_option_idx:
            console.print("\n[bold cyan]➕ Add New Model Preset to models.json[/bold cyan]")
            repo_id = input("Enter HuggingFace Repo ID or Local Path (e.g., Qwen/Qwen2.5-3B-Instruct): ").strip()
            name = input("Enter a friendly short name (e.g., qwen2.5-3b): ").strip() or "custom_model"
            desc = input("Enter a short description: ").strip() or f"Custom model ({repo_id})"
            gated_str = input("Is this a gated model requiring HF_TOKEN? (y/N): ").strip().lower()
            gated = gated_str in ("y", "yes")

            new_preset = {
                "name": name,
                "repo": repo_id,
                "desc": desc,
                "gated": gated
            }
            new_key = str(len(model_presets) + 1)
            model_presets[new_key] = new_preset
            save_model_presets(model_presets, args.models_file)

            model_id = repo_id
            friendly_name = name
        else:
            model_id = input("Enter HuggingFace Repo ID or Local Path: ").strip()
            friendly_name = input("Enter a friendly name (e.g., my_model): ").strip() or "custom_model"
    else:
        model_id = args.model
        friendly_name = args.name or os.path.basename(model_id).lower()

    # Define paths for raw and modified models
    original_model_dir = os.path.join(args.output_dir, f"original_{friendly_name}")
    abliterated_model_dir = os.path.join(args.output_dir, f"abliterated_{friendly_name}")

    # Step 1: Ingest/Download Model
    local_source_path = download_model_if_needed(model_id, original_model_dir)

    # Step 2: Ingest Datasets
    harmful_prompts, harmless_prompts = load_datasets(args.dataset_dir, args.num_prompts)
    console.print(f"[bold green]Loaded {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless contrastive prompts.[/bold green]")


    # Step 3: Load Model into Memory
    # Using bfloat16 and device_map="cpu" keeps RAM footprint low and avoids GPU driver crashes on laptops
    console.print(f"\n[bold cyan]Loading model weights into memory ({local_source_path})...[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(local_source_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        local_source_path,
        dtype=torch.bfloat16,
        device_map="cpu"
    )

    # Step 4: Extract Activations
    console.print("\n[bold yellow]Extracting hidden states for harmful prompts...[/bold yellow]")
    harmful_hidden = get_hidden_states(model, tokenizer, harmful_prompts, desc="Harmful Prompts")

    console.print("\n[bold yellow]Extracting hidden states for harmless prompts...[/bold yellow]")
    harmless_hidden = get_hidden_states(model, tokenizer, harmless_prompts, desc="Harmless Prompts")

    # Step 5: Compute Refusal Vector via Difference of Means
    refusal_vector = calculate_refusal_direction(model, harmful_hidden, harmless_hidden)

    # Step 6: Apply Weight Projection (ORVP)
    perform_abliteration(model, refusal_vector)

    # Step 7: Save Decensored Model
    console.print(f"\n[bold green]Saving abliterated model to '{abliterated_model_dir}'...[/bold green]")
    os.makedirs(abliterated_model_dir, exist_ok=True)
    model.save_pretrained(abliterated_model_dir)
    tokenizer.save_pretrained(abliterated_model_dir)

    # Copy tokenizer vocab files if missing (required by llama.cpp for GGUF conversion)
    for file in os.listdir(local_source_path):
        if "token" in file or "vocab" in file:
            src = os.path.join(local_source_path, file)
            dst = os.path.join(abliterated_model_dir, file)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    console.print(f"[bold green]✅ Abliterated model saved successfully at '{abliterated_model_dir}'![/bold green]")

    # Step 8: (Optional) GGUF Conversion & Ollama Registration for BOTH Original and Abliterated Models
    if args.convert_ollama:
        console.print("\n[bold magenta]🚀 Converting and registering BOTH Original and Abliterated models in Ollama...[/bold magenta]")
        
        # 1. Convert & Register Original Baseline Model
        console.print(f"[bold cyan]1/2: Processing Original Baseline Model ('original_{friendly_name}')...[/bold cyan]")
        convert_and_deploy_ollama(local_source_path, f"original_{friendly_name}", args.output_dir)
        
        # 2. Convert & Register Abliterated Decensored Model
        console.print(f"[bold cyan]2/2: Processing Abliterated Decensored Model ('abliterated_{friendly_name}')...[/bold cyan]")
        convert_and_deploy_ollama(abliterated_model_dir, f"abliterated_{friendly_name}", args.output_dir)

        console.print(f"\n[bold green]✨ Side-by-Side Ollama Testing Ready![/bold green]")
        console.print(f"  • Run Original Baseline:   [bold cyan]ollama run original_{friendly_name.lower()}[/bold cyan]")
        console.print(f"  • Run Abliterated Model:   [bold cyan]ollama run abliterated_{friendly_name.lower()}[/bold cyan]")

    console.print("\n[bold green]🎉 All Done! Weight Abliteration & Deployment Complete.[/bold green]")



if __name__ == "__main__":
    main()
