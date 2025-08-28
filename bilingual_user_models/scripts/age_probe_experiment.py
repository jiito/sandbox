import os
import json
import argparse
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from nnsight import LanguageModel as NNSightLanguageModel  # type: ignore
    _HAS_NNSIGHT = True
except Exception:
    _HAS_NNSIGHT = False

# Local imports
from src.dataset import llama_v2_prompt
from src.intervention_utils import return_classifier_dict
from src.probes import LinearProbeClassification


DEFAULT_AGE_LABELS = [
    "child",
    "adolescent",
    "adult",
    "older_adult",
]


def load_hf(model_id: str, access_token_path: str | None, device_map: str, torch_dtype: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    access_token = None
    if access_token_path and os.path.exists(access_token_path):
        with open(access_token_path, "r", encoding="utf-8") as f:
            access_token = f.read().strip()

    dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16 if torch_dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=access_token,
        padding_side="left",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=access_token,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return tokenizer, model


def build_triggers(default_labels: List[str]) -> Dict[str, List[str]]:
    return {
        "child": ["cartoons", "playground", "Lego", "bedtime story"],
        "adolescent": ["homework", "high school", "prom", "video games"],
        "adult": ["mortgage", "work meeting", "taxes", "commute"],
        "older_adult": ["retirement", "grandchildren", "pension", "medicare"],
    }


def prompts_from_triggers(triggers: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    labels: List[str] = []
    for label, words in triggers.items():
        for w in words:
            messages = [
                {"role": "user", "content": f"Can you discuss the topic: {w}?"},
            ]
            prompt = llama_v2_prompt(messages)
            prompts.append(prompt)
            labels.append(label)
    return prompts, labels


def nnsight_collect_last_token_mlp(
    model_id: str,
    tokenizer: AutoTokenizer,
    target_layer: int,
    prompts: List[str],
    dtype: torch.dtype,
    access_token: str | None,
    device_map: str,
) -> torch.Tensor:
    if not _HAS_NNSIGHT:
        raise RuntimeError("NNSight is not available")

    # Instantiate through NNSight for consistent module access
    lm = NNSightLanguageModel(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        token=access_token,
        trust_remote_code=True,
    )

    lm_tokenizer = lm.tokenizer
    # Keep tokenizer aligned with provided one (pad token added if needed)
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        lm.model.resize_token_embeddings(len(lm_tokenizer))

    collected: List[torch.Tensor] = []
    for prompt in prompts:
        with lm.trace(prompt) as tracer:  # type: ignore
            # Save MLP module output at target layer; take last token later
            mlp_out = lm.model.layers[target_layer].mlp.output.save()  # type: ignore[attr-defined]
        # mlp_out.value shape: (batch, seq, hidden)
        acts = mlp_out.value  # type: ignore
        collected.append(acts[:, -1, :].to("cpu"))

    return torch.vstack(collected)


def hooks_collect_last_token_mlp(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    target_layer: int,
    prompts: List[str],
    device: str,
) -> torch.Tensor:
    from collections import OrderedDict

    model.eval()
    collected: List[torch.Tensor] = []

    for prompt in prompts:
        encoding = tokenizer(
            prompt,
            truncation=True,
            max_length=2048,
            return_attention_mask=True,
            return_tensors="pt",
        )
        features: Dict[str, List[torch.Tensor]] = {}

        def make_hook(name: str):
            def hook_fn(module, input, output):
                # output: (batch, seq, hidden)
                features.setdefault(name, []).append(output.detach().to("cpu"))
            return hook_fn

        handle = None
        try:
            mlp_name = f"model.layers.{target_layer}.mlp"
            module = dict(model.named_modules())[mlp_name]
            handle = module.register_forward_hook(make_hook(mlp_name))

            with torch.no_grad():
                _ = model(
                    input_ids=encoding["input_ids"].to(device),
                    attention_mask=encoding["attention_mask"].to(device),
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            if handle is not None:
                handle.remove()

        mlp_out = features[mlp_name][0]  # (1, seq, hidden)
        collected.append(mlp_out[:, -1, :])

    return torch.vstack(collected)


def load_age_probes(probe_dir: str, target_layer: int | None) -> Dict[int, torch.nn.Module]:
    classifier_dict = return_classifier_dict(
        probe_dir,
        model_func=LinearProbeClassification,
        chosen_layer=target_layer,
        mix_scaler=False,
        sklearn=False,
        logistic=True,
    )
    if "age" not in classifier_dict or len(classifier_dict["age"]) == 0:
        raise RuntimeError("No age probes found in directory: " + probe_dir)
    return classifier_dict["age"]


def predict_with_probes(
    acts: torch.Tensor,
    probes_by_layer: Dict[int, torch.nn.Module],
    target_layer: int,
) -> torch.Tensor:
    if target_layer not in probes_by_layer:
        # If chosen layer not available, pick the max available layer
        target_layer = sorted(probes_by_layer.keys())[0]
    probe = probes_by_layer[target_layer].eval()
    device = next(probe.parameters()).device
    with torch.no_grad():
        logits, _ = probe(acts.to(device))
        preds = torch.argmax(logits, dim=-1).to("cpu")
    return preds


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str],
    out_png: str,
):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run age probe experiment with NNSight tracing")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-13b-chat-hf")
    parser.add_argument("--access_token_path", type=str, default="hf_access_token.txt")
    parser.add_argument("--probe_dir", type=str, default="probe_checkpoints/controlling_probe")
    parser.add_argument("--target_layer", type=int, default=20)
    parser.add_argument("--backend", type=str, choices=["nnsight", "hooks"], default="nnsight")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--out_dir", type=str, default="outputs/age_probe_experiment")
    parser.add_argument("--triggers_json", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Build trigger sets
    if args.triggers_json and os.path.exists(args.triggers_json):
        with open(args.triggers_json, "r", encoding="utf-8") as f:
            triggers = json.load(f)
    else:
        triggers = build_triggers(DEFAULT_AGE_LABELS)

    prompts, true_labels_str = prompts_from_triggers(triggers)
    label_to_idx = {name: idx for idx, name in enumerate(DEFAULT_AGE_LABELS)}
    y_true = [label_to_idx[l] for l in true_labels_str]

    # Load HF
    tokenizer, model = load_hf(args.model_id, args.access_token_path, args.device_map, args.torch_dtype)
    device = next(model.parameters()).device

    # Load probes
    probes_by_layer = load_age_probes(args.probe_dir, args.target_layer)

    # Collect activations (last token of MLP at chosen layer)
    acts = None
    dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
    access_token = None
    if args.access_token_path and os.path.exists(args.access_token_path):
        with open(args.access_token_path, "r", encoding="utf-8") as f:
            access_token = f.read().strip()

    if args.backend == "nnsight" and _HAS_NNSIGHT:
        try:
            acts = nnsight_collect_last_token_mlp(
                model_id=args.model_id,
                tokenizer=tokenizer,
                target_layer=args.target_layer,
                prompts=prompts,
                dtype=dtype,
                access_token=access_token,
                device_map=args.device_map,
            )
        except Exception as e:
            print(f"NNSight backend failed ({e}). Falling back to hooks backend.")
            acts = hooks_collect_last_token_mlp(model, tokenizer, args.target_layer, prompts, str(device))
    else:
        acts = hooks_collect_last_token_mlp(model, tokenizer, args.target_layer, prompts, str(device))

    # Predict with probes
    preds = predict_with_probes(acts.to(torch.float32), probes_by_layer, args.target_layer)

    # Save artifacts
    np.savez_compressed(
        os.path.join(run_dir, "hidden_states.npz"),
        acts=acts.cpu().numpy(),
        y_true=np.array(y_true, dtype=np.int64),
    )
    with open(os.path.join(run_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": DEFAULT_AGE_LABELS,
                "y_true": y_true,
                "y_pred": preds.tolist(),
                "prompts": prompts,
                "triggers": triggers,
                "target_layer": args.target_layer,
                "backend": ("nnsight" if _HAS_NNSIGHT and args.backend == "nnsight" else "hooks"),
            },
            f,
            indent=2,
        )

    # Confusion matrix
    cm_path = os.path.join(run_dir, "confusion_matrix.png")
    save_confusion_matrix(y_true, preds.tolist(), DEFAULT_AGE_LABELS, cm_path)
    print(f"Saved confusion matrix to: {cm_path}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()


