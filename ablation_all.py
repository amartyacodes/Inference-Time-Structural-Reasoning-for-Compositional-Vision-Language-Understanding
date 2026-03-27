"""
Winoground Caption Ablation: CLIP, BLIP, LLaVA, Qwen3-VL-8B-Thinking
======================================================================

Tests whether each model relies on compositional structure (subject/object
identity and binding) by masking or corrupting the CAPTION TEXT the model
sees. No scene graphs — purely about what each model attends to in the
caption.

SEQUENTIAL GPU: each model is loaded, evaluated, and freed before the next.

ABLATION CONDITIONS (applied to caption text):
  plain              — original caption (baseline)
  mask_subj          — subject noun phrases → [MASK]
  mask_obj           — object noun phrases → [MASK]
  mask_subj_obj      — both subjects and objects → [MASK]
  mask_verbs         — verbs/predicates → [MASK]
  swap_subj_obj      — swap subject ↔ object phrases in caption
  shuffle_nouns      — randomly reassign noun phrases across slots
  random_nouns       — replace all noun phrases with random nouns
  reverse            — reverse word order of entire caption

MODELS:
  CLIP   → CLS cosine similarity
  BLIP   → ITM P(match)
  LLaVA  → P(yes) next-token logits
  Qwen3  → P(yes) next-token logits (Qwen3-VL-8B-Thinking)

Usage:
    python winoground_caption_ablation_all.py --methods all --max_samples 50
    python winoground_caption_ablation_all.py --methods clip blip
    python winoground_caption_ablation_all.py --methods llava qwen3_gen
"""

import gc
import json
import random
import argparse
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import spacy

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForImageTextRetrieval,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor,
)

try:
    from transformers import Qwen3VLForConditionalGeneration
    HAS_QWEN3VL = True
except ImportError:
    HAS_QWEN3VL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ALL_METHODS = ["clip", "blip", "llava", "qwen3_gen"]

MASK_TOKEN = "[MASK]"

RANDOM_NOUNS = [
    "table", "river", "planet", "shoe", "clock", "mountain", "guitar",
    "window", "candle", "robot", "whale", "forest", "bridge", "lamp",
    "feather", "diamond", "ocean", "castle", "engine", "mirror",
    "cloud", "ladder", "basket", "flame", "crystal", "shadow", "tower",
    "anchor", "blanket", "compass", "dragon", "eagle", "fountain",
]

ABLATION_ORDER = [
    "plain",
    "mask_subj",
    "mask_obj",
    "mask_subj_obj",
    "mask_verbs",
    "swap_subj_obj",
    "shuffle_nouns",
    "random_nouns",
    "reverse",
]

ABLATION_LABELS = {
    "plain":          ("Original caption",       "baseline"),
    "mask_subj":      ("Mask subjects",          "entity ablation"),
    "mask_obj":       ("Mask objects",           "entity ablation"),
    "mask_subj_obj":  ("Mask subj + obj",        "entity ablation"),
    "mask_verbs":     ("Mask verbs",             "relation ablation"),
    "swap_subj_obj":  ("Swap subj ↔ obj",        "compositional"),
    "shuffle_nouns":  ("Shuffle noun phrases",   "binding ablation"),
    "random_nouns":   ("Random noun phrases",    "control"),
    "reverse":        ("Reverse word order",     "word-order"),
}


# ═════════════════════════════════════════════════════════════
# GPU Memory Management
# ═════════════════════════════════════════════════════════════

def free_gpu(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info(f"GPU freed. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "No CUDA")


# ═════════════════════════════════════════════════════════════
# Caption Manipulation via spaCy
# ═════════════════════════════════════════════════════════════

class CaptionManipulator:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.nlp = spacy.load("en_core_web_sm")
        log.info(f"CaptionManipulator spaCy: {self.nlp.meta['name']}")

    def _get_spans(self, caption: str):
        doc = self.nlp(caption)
        subjects, objects, verbs = [], [], []

        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subjects.append(self._get_noun_span(token, doc))
            elif token.dep_ in ("dobj", "pobj", "attr", "oprd"):
                objects.append(self._get_noun_span(token, doc))
            elif token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                verbs.append(self._get_verb_span(token, doc))

        subjects = self._deduplicate_spans(subjects)
        objects = self._deduplicate_spans(objects)
        verbs = self._deduplicate_spans(verbs)
        all_nouns = self._deduplicate_spans(subjects + objects)

        return {"subjects": subjects, "objects": objects, "verbs": verbs, "all_nouns": all_nouns}

    def _get_noun_span(self, token, doc):
        subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
        phrase_tokens = []
        for t in subtree_tokens:
            if t.dep_ in ("det", "amod", "compound", "nummod", "poss", "case",
                          "nsubj", "nsubjpass", "dobj", "pobj", "attr", "oprd",
                          "advmod", "quantmod"):
                phrase_tokens.append(t)
            elif t == token:
                phrase_tokens.append(t)
            elif t.dep_ in ("prep", "relcl", "acl", "advcl", "cc", "conj"):
                break
        if not phrase_tokens:
            phrase_tokens = [token]
        phrase_tokens = sorted(phrase_tokens, key=lambda t: t.i)
        start = phrase_tokens[0].idx
        end = phrase_tokens[-1].idx + len(phrase_tokens[-1].text)
        return (start, end, doc.text[start:end])

    def _get_verb_span(self, token, doc):
        tokens = [token]
        for child in token.children:
            if child.dep_ in ("prt", "neg", "aux", "auxpass"):
                tokens.append(child)
        tokens = sorted(tokens, key=lambda t: t.i)
        start = tokens[0].idx
        end = tokens[-1].idx + len(tokens[-1].text)
        return (start, end, doc.text[start:end])

    def _deduplicate_spans(self, spans):
        if not spans:
            return []
        spans = sorted(set(spans), key=lambda s: (s[0], -s[1]))
        result = [spans[0]]
        for s in spans[1:]:
            prev = result[-1]
            if s[0] >= prev[0] and s[1] <= prev[1]:
                continue
            if s[0] < prev[1]:
                continue
            result.append(s)
        return result

    def _replace_spans(self, text: str, spans: list, replacements: list) -> str:
        if not spans:
            return text
        paired = sorted(zip(spans, replacements), key=lambda x: x[0][0], reverse=True)
        for (start, end, _), repl in paired:
            text = text[:start] + repl + text[end:]
        return text

    def mask_subjects(self, caption):
        spans = self._get_spans(caption)
        s = spans["subjects"]
        return self._replace_spans(caption, s, [MASK_TOKEN] * len(s)) if s else caption

    def mask_objects(self, caption):
        spans = self._get_spans(caption)
        o = spans["objects"]
        return self._replace_spans(caption, o, [MASK_TOKEN] * len(o)) if o else caption

    def mask_subj_obj(self, caption):
        spans = self._get_spans(caption)
        n = spans["all_nouns"]
        return self._replace_spans(caption, n, [MASK_TOKEN] * len(n)) if n else caption

    def mask_verbs(self, caption):
        spans = self._get_spans(caption)
        v = spans["verbs"]
        return self._replace_spans(caption, v, [MASK_TOKEN] * len(v)) if v else caption

    def swap_subj_obj(self, caption):
        spans = self._get_spans(caption)
        subjs, objs = spans["subjects"], spans["objects"]
        if not subjs or not objs:
            return caption
        s, o = subjs[0], objs[0]
        if s[0] < o[0]:
            return caption[:s[0]] + o[2] + caption[s[1]:o[0]] + s[2] + caption[o[1]:]
        else:
            return caption[:o[0]] + s[2] + caption[o[1]:s[0]] + o[2] + caption[s[1]:]

    def shuffle_nouns(self, caption, rng):
        spans = self._get_spans(caption)
        n = spans["all_nouns"]
        if len(n) <= 1:
            return caption
        texts = [s[2] for s in n]
        rng.shuffle(texts)
        return self._replace_spans(caption, n, texts)

    def random_nouns(self, caption, rng):
        spans = self._get_spans(caption)
        n = spans["all_nouns"]
        if not n:
            return caption
        return self._replace_spans(caption, n, [rng.choice(RANDOM_NOUNS) for _ in n])

    def reverse_caption(self, caption):
        return " ".join(reversed(caption.split()))

    def get_parse_info(self, caption):
        spans = self._get_spans(caption)
        return {
            "subjects": [s[2] for s in spans["subjects"]],
            "objects": [s[2] for s in spans["objects"]],
            "verbs": [s[2] for s in spans["verbs"]],
        }

    def ablate_all(self, cap0, cap1, rng):
        """Return dict of ablation_name → (ablated_cap0, ablated_cap1)."""
        result = {}
        for abl in ABLATION_ORDER:
            if abl == "plain":
                result[abl] = (cap0, cap1)
            elif abl == "mask_subj":
                result[abl] = (self.mask_subjects(cap0), self.mask_subjects(cap1))
            elif abl == "mask_obj":
                result[abl] = (self.mask_objects(cap0), self.mask_objects(cap1))
            elif abl == "mask_subj_obj":
                result[abl] = (self.mask_subj_obj(cap0), self.mask_subj_obj(cap1))
            elif abl == "mask_verbs":
                result[abl] = (self.mask_verbs(cap0), self.mask_verbs(cap1))
            elif abl == "swap_subj_obj":
                result[abl] = (self.swap_subj_obj(cap0), self.swap_subj_obj(cap1))
            elif abl == "shuffle_nouns":
                result[abl] = (self.shuffle_nouns(cap0, rng), self.shuffle_nouns(cap1, rng))
            elif abl == "random_nouns":
                result[abl] = (self.random_nouns(cap0, rng), self.random_nouns(cap1, rng))
            elif abl == "reverse":
                result[abl] = (self.reverse_caption(cap0), self.reverse_caption(cap1))
        return result


# ═════════════════════════════════════════════════════════════
# Data Preparation
# ═════════════════════════════════════════════════════════════

def prepare_data(dataset, split, max_samples, manipulator, seed=42):
    """Pre-extract images, captions, parse info, and all ablated captions."""
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    rng = random.Random(seed)
    examples = []

    for idx, example in enumerate(data):
        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        ablated = manipulator.ablate_all(cap0, cap1, rng)

        examples.append({
            "idx": idx,
            "img0": example["image_0"].convert("RGB"),
            "img1": example["image_1"].convert("RGB"),
            "cap0": cap0,
            "cap1": cap1,
            "tag": example.get("tag", ""),
            "parse_cap0": manipulator.get_parse_info(cap0),
            "parse_cap1": manipulator.get_parse_info(cap1),
            "ablated": ablated,  # dict: abl_name → (cap0_abl, cap1_abl)
        })

    return examples


def init_per_example(examples):
    """Create per_example result rows."""
    per_example = []
    for ex in examples:
        per_example.append({
            "idx": ex["idx"],
            "caption_0": ex["cap0"],
            "caption_1": ex["cap1"],
            "tag": ex["tag"],
            "parse_cap0": ex["parse_cap0"],
            "parse_cap1": ex["parse_cap1"],
        })
    return per_example


# ═════════════════════════════════════════════════════════════
# Winoground Metrics
# ═════════════════════════════════════════════════════════════

def winoground_metrics(s00, s10, s01, s11):
    text  = (s00 > s10) and (s11 > s01)
    image = (s00 > s01) and (s11 > s10)
    return text, image, text and image


def _make_row(s00, s10, s01, s11, tc, ic, gc, cap0_used="", cap1_used=""):
    return {
        "scores": {"c0_i0": round(s00, 5), "c1_i0": round(s10, 5),
                    "c0_i1": round(s01, 5), "c1_i1": round(s11, 5)},
        "correct": {"text": tc, "image": ic, "group": gc},
        "caption_0_used": cap0_used,
        "caption_1_used": cap1_used,
    }


def _strat_name(model_name, abl_name):
    """e.g. clip__plain, clip__mask_subj, llava__swap_subj_obj."""
    return f"{model_name}__{abl_name}"


# ═════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════

def load_clip(model_id, device):
    log.info(f"Loading CLIP: {model_id}")
    d = device if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(d).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, d


def load_blip(model_id, device):
    if model_id is None:
        model_id = "Salesforce/blip-itm-base-coco"
    log.info(f"Loading BLIP (ITM): {model_id}")
    d = device if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(d).eval()
    return model, processor, d


def load_llava(model_id, device):
    log.info(f"Loading LLaVA: {model_id}")
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device, low_cpu_mem_usage=True,
    ).eval()
    return model, processor


def load_qwen3_gen(model_id, device):
    if not HAS_QWEN3VL:
        raise ImportError("Qwen3VLForConditionalGeneration not found. pip install transformers>=4.57.0")
    log.info(f"Loading Qwen3-VL-Thinking: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device, low_cpu_mem_usage=True,
    ).eval()
    return model, processor


# ═════════════════════════════════════════════════════════════
# Scoring Primitives
# ═════════════════════════════════════════════════════════════

@torch.no_grad()
def score_clip(model, processor, device, image, caption):
    img_in = processor(images=image, return_tensors="pt")
    txt_in = processor(text=[caption], return_tensors="pt", padding=True)
    combined = {k: v.to(device) for k, v in {**img_in, **txt_in}.items()}
    out = model(**combined)
    img_e = F.normalize(out.image_embeds.float(), dim=-1)[0]
    txt_e = F.normalize(out.text_embeds.float(), dim=-1)[0]
    return ((img_e * txt_e).sum().item() + 1) / 2


@torch.no_grad()
def score_blip(model, processor, device, image, caption):
    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    out = model(**inputs)
    return F.softmax(out.itm_score, dim=1)[0, 1].item()


@torch.no_grad()
def score_llava(model, processor, image, caption):
    prompt = (
        f"USER: <image>\nDoes this image match the caption: '{caption}'?\n"
        f"Answer yes or no.\nASSISTANT:"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    last = logits[0, -1]
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = processor.tokenizer.encode("no",  add_special_tokens=False)[0]
    return torch.softmax(torch.stack([last[yes_id], last[no_id]]), dim=0)[0].item()


def _resolve_yes_no_ids(tokenizer, word):
    candidates = set()
    for variant in [word, word.capitalize(), word.upper()]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            candidates.add(ids[0])
    if not candidates:
        raise ValueError(f"Could not resolve token IDs for '{word}'")
    return torch.tensor(list(candidates), dtype=torch.long)


@torch.no_grad()
def score_qwen3_gen(model, processor, image, caption):
    prompt = (
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer with only 'yes' or 'no'."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    last = logits[0, -1]
    yes_ids = _resolve_yes_no_ids(processor.tokenizer, "yes")
    no_ids = _resolve_yes_no_ids(processor.tokenizer, "no")
    yes_logit = torch.logsumexp(last[yes_ids], dim=0)
    no_logit = torch.logsumexp(last[no_ids], dim=0)
    return torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0].item()


# ═════════════════════════════════════════════════════════════
# Per-Model Evaluation Phases
# ═════════════════════════════════════════════════════════════

def _run_ablation_loop(examples, per_example, model_name, score_fn):
    """
    Run all ablation conditions for one model. score_fn(image, caption) → float.
    Returns counts dict with keys like 'clip__plain', 'clip__mask_subj', etc.
    """
    strats = [_strat_name(model_name, abl) for abl in ABLATION_ORDER]
    counts = {s: {"text": 0, "image": 0, "group": 0} for s in strats}

    for ex in tqdm(examples, desc=f"{model_name} ablation"):
        i = ex["idx"]
        img0, img1 = ex["img0"], ex["img1"]

        for abl in ABLATION_ORDER:
            ac0, ac1 = ex["ablated"][abl]
            sname = _strat_name(model_name, abl)

            s00 = score_fn(img0, ac0)
            s10 = score_fn(img0, ac1)
            s01 = score_fn(img1, ac0)
            s11 = score_fn(img1, ac1)

            tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
            counts[sname]["text"] += int(tc)
            counts[sname]["image"] += int(ic)
            counts[sname]["group"] += int(gc)
            per_example[i][sname] = _make_row(s00, s10, s01, s11, tc, ic, gc, ac0, ac1)

        if (i + 1) % 10 == 0:
            n = i + 1
            for abl in ["plain", "swap_subj_obj", "mask_subj_obj"]:
                sn = _strat_name(model_name, abl)
                c = counts[sn]
                log.info(f"  [{sn}] n={n} | text={c['text']/n:.3f} | "
                         f"image={c['image']/n:.3f} | group={c['group']/n:.3f}")

    return counts


def run_clip_phase(examples, per_example, args):
    log.info("═══ CLIP ablation phase ═══")
    model, processor, device = load_clip(args.clip_model_id, args.device)
    score_fn = lambda img, cap: score_clip(model, processor, device, img, cap)
    counts = _run_ablation_loop(examples, per_example, "clip", score_fn)
    free_gpu(model, processor)
    return counts


def run_blip_phase(examples, per_example, args):
    log.info("═══ BLIP ablation phase ═══")
    model, processor, device = load_blip(args.blip_model_id, args.device)
    score_fn = lambda img, cap: score_blip(model, processor, device, img, cap)
    counts = _run_ablation_loop(examples, per_example, "blip", score_fn)
    free_gpu(model, processor)
    return counts


def run_llava_phase(examples, per_example, args):
    log.info("═══ LLaVA ablation phase ═══")
    model, processor = load_llava(args.llava_model_id, args.device)
    score_fn = lambda img, cap: score_llava(model, processor, img, cap)
    counts = _run_ablation_loop(examples, per_example, "llava", score_fn)
    free_gpu(model, processor)
    return counts


def run_qwen3_gen_phase(examples, per_example, args):
    log.info("═══ Qwen3-VL-Thinking ablation phase ═══")
    model, processor = load_qwen3_gen(args.qwen3_gen_model_id, args.device)
    score_fn = lambda img, cap: score_qwen3_gen(model, processor, img, cap)
    counts = _run_ablation_loop(examples, per_example, "qwen3_gen", score_fn)
    free_gpu(model, processor)
    return counts


# ═════════════════════════════════════════════════════════════
# Tag-level Analysis
# ═════════════════════════════════════════════════════════════

def analyze_by_tag(per_example, strategy_names):
    tag_data = {}
    for ex in per_example:
        tag = ex.get("tag") or "untagged"
        if tag not in tag_data:
            tag_data[tag] = {s: {"text": 0, "image": 0, "group": 0, "n": 0} for s in strategy_names}
        for s in strategy_names:
            if s not in ex: continue
            tag_data[tag][s]["text"]  += int(ex[s]["correct"]["text"])
            tag_data[tag][s]["image"] += int(ex[s]["correct"]["image"])
            tag_data[tag][s]["group"] += int(ex[s]["correct"]["group"])
            tag_data[tag][s]["n"]     += 1
    result = {}
    for tag, data in tag_data.items():
        result[tag] = {}
        for s, c in data.items():
            n = c["n"]
            result[tag][s] = {
                "text": c["text"] / n if n else 0,
                "image": c["image"] / n if n else 0,
                "group": c["group"] / n if n else 0, "n": n,
            }
    return result


# ═════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════

MODEL_DISPLAY = {
    "clip": "CLIP",
    "blip": "BLIP",
    "llava": "LLaVA",
    "qwen3_gen": "Qwen3-VL-Think",
}


def print_summary(summary, active_methods, tag_analysis=None):
    n = summary.get("n_evaluated", "?")
    W, G = 22, 8

    print(f"\n{'═' * 100}")
    print(f"  Winoground Caption Ablation — All Models  (n={n})")
    print(f"  NO scene graphs — caption text manipulation only")
    print(f"{'═' * 100}")

    # Print one table per model
    for method in active_methods:
        model_label = MODEL_DISPLAY.get(method, method)
        base_key = _strat_name(method, "plain")
        base_group = summary.get(base_key, {}).get("group", 0)

        print(f"\n  ── {model_label} ──")
        print(f"  {'Condition':<{W}}  {'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}  {'Δgrp':>{G}}")
        print(f"  {'-' * 52}")
        print(f"  {'Random chance':<{W}}  {'0.250':>{G}}  {'0.250':>{G}}  {'0.063':>{G}}  {'':>{G}}")

        for abl in ABLATION_ORDER:
            sname = _strat_name(method, abl)
            if sname not in summary:
                continue
            label = ABLATION_LABELS[abl][0]
            v = summary[sname]
            delta = v["group"] - base_group if abl != "plain" else ""
            delta_str = f"{delta:+.4f}" if isinstance(delta, float) else ""
            print(f"  {label:<{W}}  "
                  f"{v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}  "
                  f"{delta_str:>{G}}")

    # Cross-model comparison table
    print(f"\n{'═' * 100}")
    print(f"  ── Cross-Model Comparison (Group Score) ──")
    header = f"  {'Condition':<{W}}"
    for method in active_methods:
        header += f"  {MODEL_DISPLAY.get(method, method):>{10}}"
    print(header)
    print(f"  {'-' * (W + 2 + 12 * len(active_methods))}")

    for abl in ABLATION_ORDER:
        label = ABLATION_LABELS[abl][0]
        line = f"  {label:<{W}}"
        for method in active_methods:
            sname = _strat_name(method, abl)
            if sname in summary:
                line += f"  {summary[sname]['group']:>10.4f}"
            else:
                line += f"  {'—':>10}"
        print(line)

    # Delta table
    print(f"\n  ── Cross-Model Δgroup (vs plain) ──")
    header = f"  {'Condition':<{W}}"
    for method in active_methods:
        header += f"  {MODEL_DISPLAY.get(method, method):>{10}}"
    print(header)
    print(f"  {'-' * (W + 2 + 12 * len(active_methods))}")

    for abl in ABLATION_ORDER:
        if abl == "plain":
            continue
        label = ABLATION_LABELS[abl][0]
        line = f"  {label:<{W}}"
        for method in active_methods:
            base_key = _strat_name(method, "plain")
            sname = _strat_name(method, abl)
            if sname in summary and base_key in summary:
                delta = summary[sname]["group"] - summary[base_key]["group"]
                line += f"  {delta:>+10.4f}"
            else:
                line += f"  {'—':>10}"
        print(line)

    print(f"\n{'═' * 100}")

    # Interpretation
    print(f"\n  ── Interpretation Guide ──")
    print(f"  Δgrp = ablated group score − plain group score")
    print(f"  Large negative Δ  → model relies on that caption component")
    print(f"  Δ ≈ 0             → model ignores that component")
    print(f"")
    print(f"  swap_subj_obj is the KEY compositional test:")
    print(f"    plain >> swap?   → model understands who-does-what")
    print(f"    plain ≈ swap?    → model uses bag-of-words, NOT compositional")
    print(f"")
    print(f"  Comparing across models reveals which architectures are more compositional.")
    print()


def print_ablation_examples(examples, n_show=3):
    """Print sample ablated captions for sanity checking."""
    print(f"\n  ── Sample Ablated Captions ──")
    for ex in examples[:n_show]:
        print(f"\n  Example {ex['idx']}: \"{ex['cap0']}\"")
        print(f"    Parse: subj={ex['parse_cap0']['subjects']}  "
              f"obj={ex['parse_cap0']['objects']}  verb={ex['parse_cap0']['verbs']}")
        for abl in ABLATION_ORDER:
            ac0, _ = ex["ablated"][abl]
            label = ABLATION_LABELS[abl][0]
            print(f"    {label:<24} → \"{ac0}\"")
    print()


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Winoground caption ablation: CLIP, BLIP, LLaVA, Qwen3-VL (sequential GPU)")
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS + ["all"], default=["all"])
    p.add_argument("--clip_model_id",      default="openai/clip-vit-base-patch32")
    p.add_argument("--blip_model_id",      default=None)
    p.add_argument("--llava_model_id",     default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--qwen3_gen_model_id", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--spacy_model",        default="en_core_web_sm")
    p.add_argument("--hf_token",           default=None)
    p.add_argument("--max_samples",        type=int, default=None)
    p.add_argument("--split",              default="test")
    p.add_argument("--output_dir",         default="./results_caption_ablation_all")
    p.add_argument("--device",             default="cuda:0")
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--tag_analysis",       action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    methods = ALL_METHODS if "all" in args.methods else args.methods
    log.info(f"Caption ablation study — models: {methods}")
    log.info(f"Sequential GPU: load → evaluate all ablations → free → next model")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # Prepare data + ablated captions (CPU, done once)
    manipulator = CaptionManipulator(args.spacy_model)

    # Sanity check
    test_cap = "a dog is chasing a cat"
    log.info(f"Sanity check on: '{test_cap}'")
    log.info(f"  mask_subj:     '{manipulator.mask_subjects(test_cap)}'")
    log.info(f"  mask_obj:      '{manipulator.mask_objects(test_cap)}'")
    log.info(f"  mask_subj_obj: '{manipulator.mask_subj_obj(test_cap)}'")
    log.info(f"  mask_verbs:    '{manipulator.mask_verbs(test_cap)}'")
    log.info(f"  swap_subj_obj: '{manipulator.swap_subj_obj(test_cap)}'")
    test_rng = random.Random(0)
    log.info(f"  shuffle_nouns: '{manipulator.shuffle_nouns(test_cap, test_rng)}'")
    log.info(f"  random_nouns:  '{manipulator.random_nouns(test_cap, test_rng)}'")
    log.info(f"  reverse:       '{manipulator.reverse_caption(test_cap)}'")

    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    examples = prepare_data(dataset, args.split, args.max_samples, manipulator, args.seed)
    per_example = init_per_example(examples)
    n = len(examples)
    log.info(f"Prepared {n} examples with all ablated captions")

    # Print sample ablations
    print_ablation_examples(examples)

    # Run each model sequentially
    all_counts = {}

    if "clip" in methods:
        all_counts.update(run_clip_phase(examples, per_example, args))

    if "blip" in methods:
        all_counts.update(run_blip_phase(examples, per_example, args))

    if "llava" in methods:
        all_counts.update(run_llava_phase(examples, per_example, args))

    if "qwen3_gen" in methods:
        all_counts.update(run_qwen3_gen_phase(examples, per_example, args))

    # Aggregate
    summary = {s: {k: v / n for k, v in c.items()} for s, c in all_counts.items()}
    summary["n_evaluated"] = n

    all_strats = list(all_counts.keys())
    tag_analysis = analyze_by_tag(per_example, all_strats) if args.tag_analysis else None

    print_summary(summary, methods, tag_analysis)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "methods": methods,
        "clip_model_id": args.clip_model_id,
        "blip_model_id": args.blip_model_id,
        "llava_model_id": args.llava_model_id,
        "qwen3_gen_model_id": args.qwen3_gen_model_id,
        "spacy_model": args.spacy_model,
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "gpu_strategy": "sequential load/unload per model",
        "ablation_conditions": ABLATION_ORDER,
        "ablation_descriptions": {
            "plain": "original caption (baseline)",
            "mask_subj": "subject noun phrases → [MASK]",
            "mask_obj": "object noun phrases → [MASK]",
            "mask_subj_obj": "all noun phrases → [MASK]",
            "mask_verbs": "main verbs → [MASK]",
            "swap_subj_obj": "swap subject ↔ object text in caption",
            "shuffle_nouns": "randomly reassign noun phrase texts across slots",
            "random_nouns": "replace all noun phrases with random words",
            "reverse": "reverse word order of entire caption",
        },
        "scoring": {
            "clip": "CLS cosine similarity",
            "blip": "ITM P(match)",
            "llava": "P(yes) next-token logits",
            "qwen3_gen": "P(yes) next-token logits (Qwen3-VL-8B-Thinking)",
        },
    }
    for name, data in {"summary": summary, "per_example": per_example,
                       "tags": tag_analysis or {}, "config": config}.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"{name:<12} → {path}")

    log.info("Done.")


if __name__ == "__main__":
    main()