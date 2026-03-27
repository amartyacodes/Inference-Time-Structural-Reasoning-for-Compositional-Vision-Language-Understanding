"""
qwen3_probe.py
==============
Probing suite for Qwen3-VL-8B-Thinking on Winoground.

MODULE 1 — Attention Probes
    Extract cross-modal attention weights (text→image tokens) for each layer/head.
    Visualize which caption tokens (subject/verb/object) attend to which image patches.
    Compare correct vs. swapped (c0,i0) vs (c1,i0) pairs.

MODULE 2 — Activation Patching (Causal Tracing)
    Cache all residual stream activations on a "clean" (correct) run.
    Corrupt with a swapped caption, then patch one layer at a time.
    Identify which layers causally recover the correct yes/no decision.

Usage:
    # Run on a few Winoground examples, produce all plots
    python qwen3_probe.py --max_samples 20 --output_dir ./probe_results

    # Specific example index
    python qwen3_probe.py --example_idx 0 5 12 --output_dir ./probe_results

    # Only attention probes (skip patching, much faster)
    python qwen3_probe.py --skip_patching --max_samples 50
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    raise ImportError("pip install transformers>=4.57.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Custom colormap (white→teal, good for attention) ──────────
ATTN_CMAP = LinearSegmentedColormap.from_list(
    "attn", ["#ffffff", "#00897b", "#004d40"]
)
PATCH_CMAP = LinearSegmentedColormap.from_list(
    "patch", ["#c62828", "#ffffff", "#1565c0"]   # red=harmful, blue=helpful
)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProbeInput:
    """One (image, caption) pair prepared for probing."""
    image:      Image.Image
    caption:    str
    label:      str          # e.g. "c0_i0", "c1_i0", ...
    input_ids:  torch.Tensor
    attn_mask:  torch.Tensor
    pixel_vals: torch.Tensor
    # token spans (set after tokenisation)
    img_token_start: int = 0
    img_token_end:   int = 0
    text_tokens:     list[str] = field(default_factory=list)


@dataclass
class AttentionResult:
    """Per-layer, per-head attention from text→image tokens."""
    label:           str
    caption:         str
    # [n_layers, n_heads, n_text_tokens, n_img_tokens]
    attn_text2img:   np.ndarray
    text_tokens:     list[str]
    n_img_tokens:    int


@dataclass
class PatchResult:
    """Activation patching result for one example pair."""
    idx:              int
    image_id:         str           # "img0" or "img1"
    caption_clean:    str
    caption_corrupt:  str
    p_yes_clean:      float
    p_yes_corrupt:    float
    # shape: [n_layers] — ΔP(yes) = p_yes_patched[l] - p_yes_corrupt
    layer_effects:    np.ndarray
    # shape: [n_layers] — raw P(yes) after patching each layer
    p_yes_per_layer:  np.ndarray = field(default_factory=lambda: np.array([]))
    # shape: [n_layers, n_heads] — head-level patching (if run)
    head_effects:     Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

def load_model(model_id="Qwen/Qwen3-VL-8B-Thinking", device="cuda:1"):
    log.info(f"Loading {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
        # SDPA/FlashAttention return None for attention weights.
        # eager is required to get actual attention tensors for probing.
        attn_implementation="eager",
    ).eval()
    # Qwen3VLConfig is a multimodal wrapper; LM architecture lives in text_config
    tc = model.config.text_config
    n_layers = tc.num_hidden_layers
    n_heads  = tc.num_attention_heads
    log.info(f"Loaded — layers: {n_layers}, heads: {n_heads}")
    return model, processor


# ═══════════════════════════════════════════════════════════════
# Input Preparation
# ═══════════════════════════════════════════════════════════════

def _build_messages(image: Image.Image, caption: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": (
                    f"Does this image match the caption: '{caption}'?\n"
                    f"Answer with only 'yes' or 'no'."
                )},
            ],
        }
    ]


def prepare_input(
    processor, image: Image.Image, caption: str, label: str, device: torch.device
) -> ProbeInput:
    messages = _build_messages(image, caption)
    enc = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    enc.pop("token_type_ids", None)   # processor may emit this; model doesn't accept it
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"][0]

    # Decode each token for labelling axes
    text_tokens = [
        processor.tokenizer.decode([t], skip_special_tokens=False)
        for t in input_ids.tolist()
    ]

    # Find image token span heuristically:
    # Qwen3-VL uses a special <|image_pad|> or vision token placeholder range.
    # We detect the contiguous block of vision tokens by looking for the
    # repeating image_token_id (151655 for Qwen2/3-VL family).
    img_token_id = getattr(processor.tokenizer, "image_token_id", None)
    if img_token_id is None:
        # fallback: largest contiguous repeated token block
        ids = input_ids.tolist()
        token_counts = {}
        for t in ids:
            token_counts[t] = token_counts.get(t, 0) + 1
        img_token_id = max(token_counts, key=lambda t: token_counts[t])

    ids_np = input_ids.cpu().numpy()
    img_mask = ids_np == img_token_id
    if img_mask.any():
        img_start = int(np.argmax(img_mask))
        img_end   = int(len(img_mask) - np.argmax(img_mask[::-1]))
    else:
        img_start, img_end = 0, 0

    return ProbeInput(
        image=image,
        caption=caption,
        label=label,
        input_ids=input_ids,
        attn_mask=enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0],
        pixel_vals=enc.get("pixel_values", torch.zeros(1)),
        img_token_start=img_start,
        img_token_end=img_end,
        text_tokens=text_tokens,
    )


# ═══════════════════════════════════════════════════════════════
# MODULE 1 — Attention Probes
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_attention(
    model, processor, probe_in: ProbeInput
) -> AttentionResult:
    """
    Run one forward pass with output_attentions=True.
    Returns per-layer/head attention sliced to text→image tokens.
    """
    device = next(model.parameters()).device
    messages = _build_messages(probe_in.image, probe_in.caption)
    enc = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    enc.pop("token_type_ids", None)
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc, output_attentions=True)
    # attentions: tuple of [1, n_heads, seq, seq] per layer
    attentions = out.attentions   # len = n_layers

    # Guard: SDPA/FlashAttention return None per layer; requires attn_implementation="eager"
    if attentions is None or attentions[0] is None:
        raise RuntimeError(
            "attentions is None — the model must be loaded with "
            'attn_implementation="eager" to return attention weights. '
            "Reload with load_model(...) which sets this automatically."
        )

    n_layers = len(attentions)
    n_heads  = attentions[0].shape[1]
    seq_len  = attentions[0].shape[-1]

    s, e = probe_in.img_token_start, probe_in.img_token_end
    n_img = max(e - s, 1)

    # text positions = everything outside the image block
    text_pos = list(range(0, s)) + list(range(e, seq_len))

    # Collect [n_layers, n_heads, n_text, n_img]
    stack = []
    for layer_attn in attentions:
        a = layer_attn[0].float().cpu().numpy()   # [n_heads, seq, seq]
        # text tokens attending to image tokens
        a_t2i = a[:, text_pos, :][:, :, s:e]     # [n_heads, n_text, n_img]
        stack.append(a_t2i)
    attn_arr = np.stack(stack, axis=0)            # [n_layers, n_heads, n_text, n_img]

    text_tokens = [probe_in.text_tokens[i] for i in text_pos]

    return AttentionResult(
        label=probe_in.label,
        caption=probe_in.caption,
        attn_text2img=attn_arr,
        text_tokens=text_tokens,
        n_img_tokens=n_img,
    )


def _clean_token(tok: str) -> str:
    """Make token strings display-friendly."""
    tok = tok.replace("Ġ", " ").replace("▁", " ").strip()
    if len(tok) > 12:
        tok = tok[:11] + "…"
    return tok if tok else "·"


# ── Visualization helpers ─────────────────────────────────────

def plot_attention_heatmap(
    result: AttentionResult,
    out_path: Path,
    layer_indices: Optional[list[int]] = None,
    head_mean: bool = True,
):
    """
    Plot text→image attention averaged over heads (or per-head) for selected layers.
    X-axis: image token index  |  Y-axis: text tokens
    """
    n_layers, n_heads, n_text, n_img = result.attn_text2img.shape
    if layer_indices is None:
        # evenly spaced selection of 6 layers
        layer_indices = list(np.linspace(0, n_layers - 1, 6, dtype=int))

    n_cols = len(layer_indices)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.5, max(4, n_text * 0.28)))
    fig.suptitle(
        f"Text→Image Attention  |  {result.label}\n\"{result.caption[:80]}\"",
        fontsize=9, y=1.01,
    )

    tokens_clean = [_clean_token(t) for t in result.text_tokens]

    for ax, li in zip(axes, layer_indices):
        if head_mean:
            data = result.attn_text2img[li].mean(axis=0)   # [n_text, n_img]
        else:
            data = result.attn_text2img[li].max(axis=0)    # max over heads

        im = ax.imshow(data, aspect="auto", cmap=ATTN_CMAP, vmin=0)
        ax.set_title(f"Layer {li}", fontsize=8)
        ax.set_xlabel("Image token idx", fontsize=7)
        ax.set_yticks(range(n_text))
        ax.set_yticklabels(tokens_clean, fontsize=5)
        ax.tick_params(axis="x", labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved attention heatmap → {out_path}")


def plot_attention_comparison(
    results: list[AttentionResult],
    out_path: Path,
    layer_idx: int = -1,
):
    """
    Side-by-side comparison of text→image attention for multiple (image,caption) combos.
    Useful for comparing (c0,i0) vs (c1,i0) — same image, swapped caption.
    layer_idx: which layer to visualise (-1 = last).
    """
    n = len(results)
    n_layers = results[0].attn_text2img.shape[0]
    li = layer_idx % n_layers

    fig, axes = plt.subplots(1, n, figsize=(n * 4, 6), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        data = res.attn_text2img[li].mean(axis=0)   # [n_text, n_img]
        tokens_clean = [_clean_token(t) for t in res.text_tokens]
        im = ax.imshow(data, aspect="auto", cmap=ATTN_CMAP, vmin=0)
        ax.set_title(f"{res.label}\n\"{res.caption[:50]}\"", fontsize=7)
        ax.set_xlabel("Image token idx", fontsize=7)
        ax.set_yticks(range(len(tokens_clean)))
        ax.set_yticklabels(tokens_clean, fontsize=5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Attention Comparison  (layer {li})", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved attention comparison → {out_path}")


def plot_head_attention_grid(
    result: AttentionResult,
    out_path: Path,
    layer_idx: int,
    top_n_tokens: int = 20,
):
    """
    Grid of all attention heads at a given layer.
    Rows = text tokens (top_n_tokens most attended), Cols = heads.
    """
    n_layers, n_heads, n_text, n_img = result.attn_text2img.shape
    li = layer_idx % n_layers
    layer_data = result.attn_text2img[li]   # [n_heads, n_text, n_img]

    # Select top tokens by mean attention over all heads & img tokens
    mean_per_token = layer_data.mean(axis=(0, 2))   # [n_text]
    top_idx = np.argsort(mean_per_token)[-top_n_tokens:][::-1]
    tokens_clean = [_clean_token(result.text_tokens[i]) for i in top_idx]

    cols = min(n_heads, 8)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.2))
    axes = np.array(axes).flatten()

    for h in range(n_heads):
        ax = axes[h]
        data = layer_data[h][top_idx]   # [top_n, n_img]
        ax.imshow(data, aspect="auto", cmap=ATTN_CMAP, vmin=0)
        ax.set_title(f"H{h}", fontsize=7)
        ax.set_yticks(range(len(tokens_clean)))
        ax.set_yticklabels(tokens_clean, fontsize=5)
        ax.tick_params(axis="x", labelsize=5)

    for ax in axes[n_heads:]:
        ax.axis("off")

    fig.suptitle(
        f"All Heads — Layer {li}  |  {result.label}\n\"{result.caption[:70]}\"",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Saved head grid → {out_path}")


def plot_attention_delta(
    res_correct: AttentionResult,
    res_swapped: AttentionResult,
    out_path: Path,
    layer_indices: Optional[list[int]] = None,
):
    """
    Δ attention = correct_attn − swapped_attn  (mean over heads).
    Highlights which text→image paths change when you swap the caption.
    Blue = more attention in correct, Red = more in swapped.
    """
    n_layers = res_correct.attn_text2img.shape[0]
    if layer_indices is None:
        layer_indices = list(np.linspace(0, n_layers - 1, 6, dtype=int))

    # Align text token count (may differ if captions have different lengths)
    min_text = min(res_correct.attn_text2img.shape[2],
                   res_swapped.attn_text2img.shape[2])
    min_img  = min(res_correct.attn_text2img.shape[3],
                   res_swapped.attn_text2img.shape[3])

    n_cols = len(layer_indices)
    tokens_clean = [_clean_token(t) for t in res_correct.text_tokens[:min_text]]

    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.5, max(4, min_text * 0.28)))
    fig.suptitle(
        f"Δ Attention (correct − swapped)\n"
        f"Correct: \"{res_correct.caption[:60]}\"\n"
        f"Swapped: \"{res_swapped.caption[:60]}\"",
        fontsize=8, y=1.02,
    )

    for ax, li in zip(axes, layer_indices):
        delta = (
            res_correct.attn_text2img[li, :, :min_text, :min_img].mean(0)
          - res_swapped.attn_text2img[li, :, :min_text, :min_img].mean(0)
        )
        vmax = np.abs(delta).max() + 1e-8
        im = ax.imshow(delta, aspect="auto", cmap=PATCH_CMAP,
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"Layer {li}", fontsize=8)
        ax.set_xlabel("Image token idx", fontsize=7)
        ax.set_yticks(range(min_text))
        ax.set_yticklabels(tokens_clean, fontsize=5)
        ax.tick_params(axis="x", labelsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved attention delta → {out_path}")


# ═══════════════════════════════════════════════════════════════
# MODULE 2 — Activation Patching (Causal Tracing)
# ═══════════════════════════════════════════════════════════════

def _resolve_yes_no_ids(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    def get_ids(word):
        candidates = set()
        for v in [word, word.capitalize(), word.upper()]:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if ids:
                candidates.add(ids[0])
        return torch.tensor(list(candidates), dtype=torch.long)
    return get_ids("yes"), get_ids("no")


@torch.no_grad()
def _p_yes(model, processor, enc: dict) -> float:
    out = model(**enc)
    last = out.logits[0, -1].float()
    yes_ids, no_ids = _resolve_yes_no_ids(processor.tokenizer)
    y = torch.logsumexp(last[yes_ids.to(last.device)], dim=0)
    n = torch.logsumexp(last[no_ids.to(last.device)], dim=0)
    return torch.softmax(torch.stack([y, n]), dim=0)[0].item()


def _encode(processor, image: Image.Image, caption: str, device) -> dict:
    messages = _build_messages(image, caption)
    enc = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    enc.pop("token_type_ids", None)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def _cache_residuals(model, enc: dict) -> list[torch.Tensor]:
    """
    Run forward pass, return residual stream (hidden state) at each layer.
    Returns list of [1, seq_len, hidden_dim] tensors (one per layer, including input embed).
    """
    out = model(**enc, output_hidden_states=True)
    # hidden_states[0] = embedding, [1..n] = after each transformer layer
    return [h.detach().clone() for h in out.hidden_states]


def _patch_layer_hook(cached_state: torch.Tensor, layer_idx: int):
    """
    Returns a forward hook that replaces the residual stream output
    of `layer_idx` with the cached clean state.
    """
    def hook(module, input, output):
        # output is typically a tuple; first element is the hidden state
        if isinstance(output, tuple):
            patched = list(output)
            patched[0] = cached_state.to(output[0].device)
            return tuple(patched)
        else:
            return cached_state.to(output.device)
    return hook


def run_activation_patching(
    model,
    processor,
    image_clean: Image.Image,
    caption_clean: str,
    image_corrupt: Image.Image,
    caption_corrupt: str,
    image_id: str = "img0",
    patch_heads: bool = False,
) -> PatchResult:
    """
    Activation patching (causal tracing) between a clean and corrupt run.

    For Winoground: clean = correct (c0,i0), corrupt = swapped (c1,i0).

    For each layer l:
      1. Run corrupt forward pass
      2. Patch hidden state at layer l with clean cached state
      3. Record P(yes) — if it rises toward clean P(yes), layer l is causally important

    Layer effect[l] = P(yes)_patched_at_l − P(yes)_corrupt
    """
    device = next(model.parameters()).device

    enc_clean   = _encode(processor, image_clean,   caption_clean,   device)
    enc_corrupt = _encode(processor, image_corrupt, caption_corrupt, device)

    p_clean   = _p_yes(model, processor, enc_clean)
    p_corrupt = _p_yes(model, processor, enc_corrupt)
    log.info(f"  P(yes) clean={p_clean:.4f}  corrupt={p_corrupt:.4f}")

    # Cache clean residuals
    clean_residuals = _cache_residuals(model, enc_clean)
    # clean_residuals[0] = embed, [1..n_layers] = after layer 0..n_layers-1

    # Qwen3VLConfig wraps the LM config under text_config
    tc       = model.config.text_config
    n_layers = tc.num_hidden_layers
    layer_effects   = np.zeros(n_layers)
    p_yes_per_layer = np.zeros(n_layers)

    # Access the transformer layers
    # Qwen3-VL architecture: Qwen3VLForConditionalGeneration
    #   └── model  (Qwen3VLModel)
    #       └── language_model  (Qwen3VLTextModel)
    #           └── layers  (transformer decoder blocks)
    # _tied_weights_keys confirms: "model.language_model.embed_tokens.weight"
    lm = model.model.language_model
    layers = lm.layers

    for li in tqdm(range(n_layers), desc="  Patching layers", leave=False):
        # clean_residuals[li+1] = output of layer li on clean run
        cached = clean_residuals[li + 1]
        hook = layers[li].register_forward_hook(
            _patch_layer_hook(cached, li)
        )
        try:
            p_patched = _p_yes(model, processor, enc_corrupt)
        finally:
            hook.remove()
        p_yes_per_layer[li] = p_patched
        layer_effects[li]   = p_patched - p_corrupt

    head_effects = None
    if patch_heads:
        n_heads  = tc.num_attention_heads
        head_effects = np.zeros((n_layers, n_heads))
        head_dim = tc.hidden_size // n_heads

        for li in tqdm(range(n_layers), desc="  Patching heads", leave=False):
            cached = clean_residuals[li + 1]   # [1, seq, hidden]
            for hi in range(n_heads):
                start = hi * head_dim
                end   = start + head_dim

                def head_hook(module, input, output,
                              _cached=cached, _s=start, _e=end):
                    if isinstance(output, tuple):
                        patched = list(output)
                        h = patched[0].clone()
                        h[:, :, _s:_e] = _cached[:, :, _s:_e].to(h.device)
                        patched[0] = h
                        return tuple(patched)
                    else:
                        o = output.clone()
                        o[:, :, _s:_e] = _cached[:, :, _s:_e].to(o.device)
                        return o

                hook = layers[li].register_forward_hook(head_hook)
                try:
                    p_patched = _p_yes(model, processor, enc_corrupt)
                finally:
                    hook.remove()
                head_effects[li, hi] = p_patched - p_corrupt

    return PatchResult(
        idx=0,
        image_id=image_id,
        caption_clean=caption_clean,
        caption_corrupt=caption_corrupt,
        p_yes_clean=p_clean,
        p_yes_corrupt=p_corrupt,
        layer_effects=layer_effects,
        p_yes_per_layer=p_yes_per_layer,
        head_effects=head_effects,
    )


# ── Patching Visualizations ───────────────────────────────────

def plot_layer_patching(result: PatchResult, out_path: Path):
    """
    Bar chart of layer_effects (indirect effect per layer).
    Positive = patching this layer recovers correct answer.
    """
    n = len(result.layer_effects)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.35), 4))

    colors = [
        "#1565c0" if v >= 0 else "#c62828"
        for v in result.layer_effects
    ]
    bars = ax.bar(range(n), result.layer_effects, color=colors, width=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("ΔP(yes)  [patched − corrupt]", fontsize=10)
    ax.set_title(
        f"Activation Patching — Layer Effects\n"
        f"Clean: \"{result.caption_clean[:60]}\"\n"
        f"Corrupt: \"{result.caption_corrupt[:60]}\"\n"
        f"P(yes) clean={result.p_yes_clean:.3f}  corrupt={result.p_yes_corrupt:.3f}",
        fontsize=8,
    )
    ax.set_xticks(range(0, n, max(1, n // 20)))

    # Annotate top-3 layers
    top3 = np.argsort(result.layer_effects)[-3:][::-1]
    for li in top3:
        ax.annotate(
            f"L{li}",
            xy=(li, result.layer_effects[li]),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=7, color="#1565c0",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved layer patching plot → {out_path}")


def plot_head_patching(result: PatchResult, out_path: Path):
    """
    Heatmap of head_effects [n_layers × n_heads].
    """
    if result.head_effects is None:
        log.warning("No head effects to plot.")
        return

    data = result.head_effects
    vmax = np.abs(data).max() + 1e-8

    fig, ax = plt.subplots(figsize=(data.shape[1] * 0.5, data.shape[0] * 0.35))
    im = ax.imshow(
        data, aspect="auto", cmap=PATCH_CMAP, vmin=-vmax, vmax=vmax
    )
    ax.set_xlabel("Head index", fontsize=9)
    ax.set_ylabel("Layer index", fontsize=9)
    ax.set_title(
        f"Activation Patching — Head Effects\n"
        f"Clean: \"{result.caption_clean[:55]}\"\n"
        f"Corrupt: \"{result.caption_corrupt[:55]}\"",
        fontsize=8,
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03,
                 label="ΔP(yes) [patched − corrupt]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved head patching plot → {out_path}")


def plot_patching_summary(all_results: list[PatchResult], out_path: Path):
    """
    Aggregate layer effects over all examples.
    Shows mean ± std of indirect effect across the dataset.
    """
    effects = np.stack([r.layer_effects for r in all_results])   # [N, n_layers]
    mean_e  = effects.mean(0)
    std_e   = effects.std(0)
    n_layers = mean_e.shape[0]

    fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.35), 4))
    xs = np.arange(n_layers)
    ax.bar(xs, mean_e, color="#1565c0", alpha=0.8, width=0.7, label="Mean effect")
    ax.fill_between(xs, mean_e - std_e, mean_e + std_e,
                    alpha=0.3, color="#1565c0", label="±1 std")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("Mean ΔP(yes)", fontsize=10)
    ax.set_title(
        f"Activation Patching Summary  (N={len(all_results)} examples)\n"
        f"Mean indirect effect per layer (clean caption → corrupt input)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 20)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved patching summary → {out_path}")



# ═══════════════════════════════════════════════════════════════
# Swap-specific diagnostics
# ═══════════════════════════════════════════════════════════════

def compute_layerwise_divergence(
    res_a: "AttentionResult", res_b: "AttentionResult"
) -> np.ndarray:
    """
    Per-layer Frobenius norm of attention difference (mean over heads).
    Returns array of shape [n_layers].
    A large value at layer L means the model differentiates the two captions
    visually at that layer; near-zero = captions are treated identically.
    """
    n_layers = res_a.attn_text2img.shape[0]
    min_text = min(res_a.attn_text2img.shape[2], res_b.attn_text2img.shape[2])
    min_img  = min(res_a.attn_text2img.shape[3], res_b.attn_text2img.shape[3])
    divs = []
    for li in range(n_layers):
        # mean over heads → [n_text, n_img]
        a = res_a.attn_text2img[li].mean(0)[:min_text, :min_img]
        b = res_b.attn_text2img[li].mean(0)[:min_text, :min_img]
        divs.append(np.linalg.norm(a - b, "fro"))
    return np.array(divs)


def plot_layerwise_divergence(
    res_a: "AttentionResult",
    res_b: "AttentionResult",
    out_path: "Path",
    title_suffix: str = "",
):
    """
    Line plot of ||A(cap_a) - A(cap_b)||_F per layer.
    The key diagnostic for whether the model EVER differentiates
    subject-swapped captions visually, and at which depth.
    """
    divs = compute_layerwise_divergence(res_a, res_b)
    n_layers = len(divs)
    peak_layer = int(np.argmax(divs))

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.3), 4))
    ax.plot(divs, color="#1565c0", linewidth=1.8, label="||Δ attention||_F")
    ax.axvline(peak_layer, color="#e53935", linewidth=1.2, linestyle="--",
               label=f"Peak: layer {peak_layer}  ({divs[peak_layer]:.4f})")
    ax.fill_between(range(n_layers), divs, alpha=0.15, color="#1565c0")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Frobenius norm of Δ attention", fontsize=10)
    ax.set_title(
        f"Layer-wise Attention Divergence  (caption swap)\n"
        f"  A: \"{res_a.caption[:55]}\"\n"
        f"  B: \"{res_b.caption[:55]}\"\n"
        + (f"  {title_suffix}" if title_suffix else ""),
        fontsize=8,
    )
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 20)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved layer divergence → {out_path}")
    return divs


def plot_token_attention_diff(
    res_a: "AttentionResult",
    res_b: "AttentionResult",
    out_path: "Path",
    layer_idx: int = -1,
    top_n: int = 30,
):
    """
    Per-token attention difference bar chart at a specific layer.
    For each text token: sum_over_img |A_token(cap_a) - A_token(cap_b)|
    Highlights which specific tokens (old/young/kisses) drive the divergence.
    Positive (blue) = token attends MORE to image in cap_a.
    Negative (red)  = token attends MORE to image in cap_b.
    """
    n_layers = res_a.attn_text2img.shape[0]
    li = layer_idx % n_layers

    min_text = min(res_a.attn_text2img.shape[2], res_b.attn_text2img.shape[2])
    min_img  = min(res_a.attn_text2img.shape[3], res_b.attn_text2img.shape[3])

    # mean over heads, then sum over image tokens → [n_text] signed diff
    a_tok = res_a.attn_text2img[li].mean(0)[:min_text, :min_img].sum(axis=1)
    b_tok = res_b.attn_text2img[li].mean(0)[:min_text, :min_img].sum(axis=1)
    diff  = a_tok - b_tok   # positive = cap_a attends more

    tokens_clean = [_clean_token(t) for t in res_a.text_tokens[:min_text]]

    # Sort by absolute difference, keep top_n
    order = np.argsort(np.abs(diff))[-top_n:][::-1]
    diff_sel   = diff[order]
    tokens_sel = [tokens_clean[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, top_n * 0.5), 5))
    colors = ["#1565c0" if v >= 0 else "#c62828" for v in diff_sel]
    ax.barh(range(len(diff_sel)), diff_sel, color=colors, height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(diff_sel)))
    ax.set_yticklabels(tokens_sel, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Σ_img [A(cap_a) − A(cap_b)]   (blue = more in cap_a)", fontsize=8)
    ax.set_title(
        f"Per-token Attention Difference  (layer {li})\n"
        f"  cap_a: \"{res_a.caption[:55]}\"\n"
        f"  cap_b: \"{res_b.caption[:55]}\"",
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved token diff → {out_path}")


def plot_aggregate_divergence(
    swap_pairs: list[tuple["AttentionResult", "AttentionResult"]],
    out_path: "Path",
    label: str = "img1 caption swap",
):
    """
    Aggregate layer-wise divergence over many swap pairs.
    Mean ± std of ||Δ attention||_F per layer across all examples.
    The definitive answer to: does Qwen3-VL EVER differentiate swapped captions?
    """
    all_divs = np.stack([
        compute_layerwise_divergence(a, b) for a, b in swap_pairs
    ])   # [N, n_layers]
    mean_d = all_divs.mean(0)
    std_d  = all_divs.std(0)
    n_layers = mean_d.shape[0]
    peak = int(np.argmax(mean_d))

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.3), 4))
    xs = np.arange(n_layers)
    ax.plot(xs, mean_d, color="#1565c0", linewidth=2.0, label="Mean")
    ax.fill_between(xs, mean_d - std_d, mean_d + std_d,
                    alpha=0.2, color="#1565c0", label="±1 std")
    ax.axvline(peak, color="#e53935", linewidth=1.2, linestyle="--",
               label=f"Peak: layer {peak}  ({mean_d[peak]:.4f})")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean ||Δ attention||_F", fontsize=10)
    ax.set_title(
        f"Aggregate Caption-Swap Attention Divergence  "
        f"(N={len(swap_pairs)}, {label})\n"
        f"Near-zero across all layers = model does not differentiate swapped subjects",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 20)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved aggregate divergence → {out_path}")
    return mean_d, std_d


# ═══════════════════════════════════════════════════════════════
# Aggregate Attention Analysis
# ═══════════════════════════════════════════════════════════════

def plot_mean_attention_by_layer(
    all_results: list[AttentionResult], out_path: Path
):
    """
    For each layer, average text→image attention over all examples and heads.
    Produces a single [n_layers] curve showing how attention flows changes by depth.
    """
    # mean total text→image attention per layer per example
    per_example = []
    for res in all_results:
        # mean over heads, text tokens, img tokens → scalar per layer
        per_example.append(res.attn_text2img.mean(axis=(1, 2, 3)))
    arr = np.stack(per_example)   # [N, n_layers]
    mean_v = arr.mean(0)
    std_v  = arr.std(0)

    n_layers = mean_v.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.3), 4))
    xs = np.arange(n_layers)
    ax.plot(xs, mean_v, color="#00897b", linewidth=1.8, label="Mean")
    ax.fill_between(xs, mean_v - std_v, mean_v + std_v,
                    alpha=0.25, color="#00897b", label="±1 std")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean text→image attention", fontsize=10)
    ax.set_title(
        f"Text→Image Attention by Layer  (N={len(all_results)})\n"
        f"Averaged over all heads and token positions",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved mean attention by layer → {out_path}")


# ═══════════════════════════════════════════════════════════════
# Main Evaluation Loop
# ═══════════════════════════════════════════════════════════════

def run_probes(
    model, processor, dataset, split,
    max_samples, example_indices,
    skip_patching, patch_heads,
    out_dir: Path,
):
    data = dataset[split]
    if example_indices:
        indices = [i for i in example_indices if i < len(data)]
    elif max_samples:
        indices = list(range(min(max_samples, len(data))))
    else:
        indices = list(range(len(data)))

    device = next(model.parameters()).device
    attn_dir  = out_dir / "attention"
    patch_dir = out_dir / "patching"
    attn_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    all_attn_results:  list[AttentionResult] = []
    all_patch_results: list[PatchResult]     = []
    swap_pairs_img0:   list[tuple]           = []  # (r_c0i0, r_c1i0)
    swap_pairs_img1:   list[tuple]           = []  # (r_c1i1, r_c0i1)

    for idx in tqdm(indices, desc="Examples"):
        ex   = data[idx]
        img0 = ex["image_0"].convert("RGB")
        img1 = ex["image_1"].convert("RGB")
        cap0 = ex["caption_0"]
        cap1 = ex["caption_1"]
        tag  = ex.get("tag", "")

        log.info(f"\n── Example {idx} | tag={tag} ──")
        log.info(f"  c0: {cap0}")
        log.info(f"  c1: {cap1}")

        ex_attn_dir  = attn_dir  / f"ex{idx:04d}"
        ex_patch_dir = patch_dir / f"ex{idx:04d}"
        ex_attn_dir.mkdir(exist_ok=True)
        ex_patch_dir.mkdir(exist_ok=True)

        # ── 1. Attention Probes ────────────────────────────────

        # Prepare ProbeInputs to get token spans
        pi_c0i0 = prepare_input(processor, img0, cap0, "c0_i0", device)
        pi_c1i0 = prepare_input(processor, img0, cap1, "c1_i0", device)
        pi_c0i1 = prepare_input(processor, img1, cap0, "c0_i1", device)
        pi_c1i1 = prepare_input(processor, img1, cap1, "c1_i1", device)

        log.info("  Extracting attention...")
        r_c0i0 = extract_attention(model, processor, pi_c0i0)
        r_c1i0 = extract_attention(model, processor, pi_c1i0)
        r_c0i1 = extract_attention(model, processor, pi_c0i1)
        r_c1i1 = extract_attention(model, processor, pi_c1i1)

        all_attn_results.extend([r_c0i0, r_c1i0, r_c0i1, r_c1i1])

        n_layers = r_c0i0.attn_text2img.shape[0]
        layer_sel = list(np.linspace(0, n_layers - 1, 6, dtype=int))

        # Per-pair heatmaps
        for res in [r_c0i0, r_c1i0, r_c0i1, r_c1i1]:
            plot_attention_heatmap(
                res,
                ex_attn_dir / f"attn_heatmap_{res.label}.png",
                layer_indices=layer_sel,
            )

        # Same-image comparison: (c0,i0) vs (c1,i0)
        plot_attention_comparison(
            [r_c0i0, r_c1i0],
            ex_attn_dir / "attn_compare_img0_captions.png",
            layer_idx=layer_sel[-1],
        )
        plot_attention_comparison(
            [r_c0i1, r_c1i1],
            ex_attn_dir / "attn_compare_img1_captions.png",
            layer_idx=layer_sel[-1],
        )

        # Delta attention: correct vs swapped
        plot_attention_delta(
            r_c0i0, r_c1i0,
            ex_attn_dir / "attn_delta_img0.png",
            layer_indices=layer_sel,
        )
        plot_attention_delta(
            r_c1i1, r_c0i1,
            ex_attn_dir / "attn_delta_img1.png",
            layer_indices=layer_sel,
        )

        # Head grid at the last selected layer
        for res in [r_c0i0, r_c1i0]:
            plot_head_attention_grid(
                res,
                ex_attn_dir / f"head_grid_{res.label}_L{layer_sel[-1]}.png",
                layer_idx=layer_sel[-1],
            )

        # ── Swap diagnostics (the key new plots) ──────────────
        swap_pairs_img0.append((r_c0i0, r_c1i0))
        swap_pairs_img1.append((r_c1i1, r_c0i1))

        # 1. Layer-wise divergence curve per example
        plot_layerwise_divergence(
            r_c0i0, r_c1i0,
            ex_attn_dir / "layerdiv_img0_swap.png",
            title_suffix=f"img0  |  tag={tag}",
        )
        plot_layerwise_divergence(
            r_c1i1, r_c0i1,
            ex_attn_dir / "layerdiv_img1_swap.png",
            title_suffix=f"img1  |  tag={tag}",
        )

        # 2. Per-token attention difference at early, mid, late layers
        for probe_li in [n_layers // 5, n_layers // 2, n_layers - 1]:
            plot_token_attention_diff(
                r_c0i1, r_c1i1,
                ex_attn_dir / f"tokendiff_img1_L{probe_li}.png",
                layer_idx=probe_li,
            )

        # ── 2. Activation Patching ─────────────────────────────

        if not skip_patching:
            log.info("  Running activation patching (c0,i0) → corrupt (c1,i0)...")
            pr1 = run_activation_patching(
                model, processor,
                image_clean=img0,   caption_clean=cap0,
                image_corrupt=img0, caption_corrupt=cap1,
                image_id="img0",
                patch_heads=patch_heads,
            )
            pr1.idx = idx
            all_patch_results.append(pr1)

            plot_layer_patching(pr1, ex_patch_dir / "patch_layers_img0.png")
            if patch_heads:
                plot_head_patching(pr1, ex_patch_dir / "patch_heads_img0.png")

            log.info("  Running activation patching (c1,i1) → corrupt (c0,i1)...")
            pr2 = run_activation_patching(
                model, processor,
                image_clean=img1,   caption_clean=cap1,
                image_corrupt=img1, caption_corrupt=cap0,
                image_id="img1",
                patch_heads=patch_heads,
            )
            pr2.idx = idx
            all_patch_results.append(pr2)   # was missing — pr2 never saved to JSON
            plot_layer_patching(pr2, ex_patch_dir / "patch_layers_img1.png")
            if patch_heads:
                plot_head_patching(pr2, ex_patch_dir / "patch_heads_img1.png")

    # ── Aggregate plots ────────────────────────────────────────

    log.info("Generating aggregate plots...")

    if all_attn_results:
        plot_mean_attention_by_layer(
            all_attn_results,
            out_dir / "aggregate_mean_attention_by_layer.png",
        )

    # 3. Aggregate caption-swap divergence — the headline diagnostic
    if swap_pairs_img0:
        plot_aggregate_divergence(
            swap_pairs_img0,
            out_dir / "aggregate_divergence_img0_swap.png",
            label="img0 caption swap",
        )
    if swap_pairs_img1:
        plot_aggregate_divergence(
            swap_pairs_img1,
            out_dir / "aggregate_divergence_img1_swap.png",
            label="img1 caption swap  (hardest: same image, subject-object swap)",
        )

    if all_patch_results:
        plot_patching_summary(
            all_patch_results,
            out_dir / "aggregate_patching_summary.png",
        )

    # Save raw patching data as JSON
    if all_patch_results:
        patch_data = [
            {
                "idx":              r.idx,
                "image_id":         r.image_id,
                "caption_clean":    r.caption_clean,
                "caption_corrupt":  r.caption_corrupt,
                "p_yes_clean":      r.p_yes_clean,
                "p_yes_corrupt":    r.p_yes_corrupt,
                # per-layer raw P(yes) after patching that layer's residual stream
                "p_yes_per_layer":  r.p_yes_per_layer.tolist(),
                # per-layer indirect effect = p_yes_per_layer[l] - p_yes_corrupt
                "layer_effects":    r.layer_effects.tolist(),
                # convenience: index of most causally important layer
                "top_layer":        int(np.argmax(r.layer_effects)),
                "top_layer_effect": float(np.max(r.layer_effects)),
            }
            for r in all_patch_results
        ]
        with open(out_dir / "patching_results.json", "w") as f:
            json.dump(patch_data, f, indent=2)
        log.info(f"Saved patching_results.json → {out_dir}")

    log.info(f"\nAll probe outputs in: {out_dir}")
    _print_probe_summary(all_patch_results)


def _print_probe_summary(patch_results: list[PatchResult]):
    if not patch_results:
        return
    effects = np.stack([r.layer_effects for r in patch_results])
    mean_e  = effects.mean(0)
    top5    = np.argsort(mean_e)[-5:][::-1]
    print("\n" + "═" * 60)
    print(f"  Activation Patching Summary  (N={len(patch_results)})")
    print("  Top-5 causally important layers:")
    for rank, li in enumerate(top5, 1):
        print(f"    {rank}. Layer {li:3d}  —  mean ΔP(yes) = {mean_e[li]:+.4f}")
    print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Qwen3-VL Attention Probes + Activation Patching on Winoground"
    )
    p.add_argument("--model_id",       default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--hf_token",       default=None)
    p.add_argument("--split",          default="test")
    p.add_argument("--max_samples",    type=int, default=None,
                   help="Run on first N examples (ignored if --example_idx set)")
    p.add_argument("--example_idx",    type=int, nargs="+", default=None,
                   help="Specific example indices to probe, e.g. --example_idx 0 5 12")
    p.add_argument("--output_dir",     default="./probe_results")
    p.add_argument("--device",         default="cuda:1")
    p.add_argument("--skip_patching",  action="store_true",
                   help="Skip activation patching (much faster, attention only)")
    p.add_argument("--patch_heads",    action="store_true",
                   help="Also run head-level patching (slow, ~n_layers × n_heads passes)")
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"Qwen3-VL Probing Suite")
    log.info(f"Model: {args.model_id}")
    log.info(f"Modules: attention={'YES'}  patching={'NO' if args.skip_patching else 'YES'}")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    model, processor = load_model(args.model_id, args.device)

    log.info("Loading Winoground...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_probes(
        model, processor, dataset, args.split,
        max_samples=args.max_samples,
        example_indices=args.example_idx,
        skip_patching=args.skip_patching,
        patch_heads=args.patch_heads,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()