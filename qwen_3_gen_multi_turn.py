"""
Winoground Evaluation: Qwen3-VL-8B-Thinking (Generative P(yes))
================================================================

Standalone evaluation script for the Qwen3-VL-8B-Thinking decoder model
on the Winoground benchmark. Scores via P(yes) from next-token logits,
directly comparable to the LLaVA P(yes) approach.

STRATEGIES:
  qwen3_gen          — plain P(yes), no scene graph
  qwen3_gen_sg       — P(yes) with ALL YUKINO-SG triples injected into prompt
  qwen3_gen_sg_mt    — P(yes) with MULTI-TURN self-filtering:
                         Turn 1: model selects which SG triples are relevant
                         Turn 2: model answers yes/no using only those triples
                         (if model says "none", falls back to plain prompt)

The multi-turn strategy lets the model decide *whether* and *which* scene
graph relations matter, rather than blindly injecting all of them.

Usage:
    python winoground_qwen3_gen.py --max_samples 50
    python winoground_qwen3_gen.py --model_id Qwen/Qwen3-VL-8B-Thinking
    python winoground_qwen3_gen.py --no_plain   # skip qwen3_gen baseline
"""

import gc
import json
import argparse
import logging
import re
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import spacy

from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    raise ImportError(
        "Qwen3VLForConditionalGeneration not found. "
        "pip install transformers>=4.57.0"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# GPU helpers
# ═════════════════════════════════════════════════════════════

def log_gpu():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        resrv = torch.cuda.memory_reserved() / 1e9
        log.info(f"GPU mem: {alloc:.2f} GB allocated, {resrv:.2f} GB reserved")


# ═════════════════════════════════════════════════════════════
# Scene Graph Data Structure
# ═════════════════════════════════════════════════════════════

@dataclass
class Triple:
    subject:  str
    relation: str
    obj:      str

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.obj})"


# ═════════════════════════════════════════════════════════════
# Text Scene Graph Parser (YUKINO-SG)
# ═════════════════════════════════════════════════════════════

class TextSceneGraphParser:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
            log.info(f"spaCy loaded: {model}")
        except OSError:
            log.warning(f"{model} not found, falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self._cache: dict[str, list[Triple]] = {}

    def parse(self, caption: str) -> list[Triple]:
        if caption in self._cache:
            return self._cache[caption]
        doc = self.nlp(caption.lower().strip())
        triples = []
        triples += self._extract_svo(doc)
        triples += self._extract_prep(doc)
        triples += self._extract_existential(doc)
        triples += self._extract_copular(doc)
        triples += self._extract_possessive(doc)
        seen, unique = set(), []
        for t in triples:
            key = (t.subject, t.relation, t.obj)
            if key not in seen and t.subject and t.obj:
                seen.add(key)
                unique.append(t)
        self._cache[caption] = unique
        return unique

    # ── internal extraction helpers ──────────────────────────

    def _get_adjectives(self, noun_token) -> list[str]:
        adjs = []
        for child in noun_token.children:
            if child.dep_ in {"amod", "advmod"} and (
                child.pos_ in {"ADJ", "NOUN", "VERB"}
                or child.tag_ in ("JJ", "JJR", "JJS")
            ):
                adjs.extend(c.text for c in child.children if c.dep_ == "compound")
                adjs.append(child.text)
        return adjs

    def _noun_phrase(self, token) -> str:
        noun_token = token
        if token.pos_ not in ("NOUN", "PROPN"):
            for t in token.subtree:
                if t.pos_ in ("NOUN", "PROPN"):
                    noun_token = t
                    break
        head = noun_token.lemma_
        compounds = [c.text for c in noun_token.children if c.dep_ == "compound"]
        adjs = self._get_adjectives(noun_token)
        if not adjs:
            for t in token.subtree:
                if t == noun_token:
                    break
                if t.tag_ in ("JJ", "JJR", "JJS") or (
                    t.dep_ == "amod" and t.pos_ == "ADJ"
                ):
                    adjs.append(t.text)
        parts = adjs + compounds
        return (" ".join(parts) + " " + head).strip() if parts else head

    def _compound_prep(self, prep_token) -> str:
        text = prep_token.text
        for child in prep_token.children:
            if child.dep_ in ("pcomp", "fixed"):
                text = f"{text} {child.text}"
        return text

    def _extract_svo(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            is_root = token.dep_ == "ROOT"
            has_subj = any(
                w.dep_ in ("nsubj", "nsubjpass") for w in token.children
            )
            is_subverb = token.dep_ in {
                "relcl", "acl", "advcl", "xcomp", "ccomp", "conj"
            } and (token.pos_ in ("VERB", "AUX") or has_subj)
            if not is_root and not is_subverb:
                continue
            if is_root and token.lemma_ == "be":
                continue
            subjs = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]
            if not subjs and token.dep_ in {"relcl", "acl"} and token.head.pos_ in (
                "NOUN",
                "PROPN",
            ):
                subjs = [token.head]
            if not subjs and token.dep_ in {
                "advcl", "xcomp", "ccomp", "conj"
            } and token.head.pos_ in ("VERB", "AUX"):
                subjs = [
                    w
                    for w in token.head.children
                    if w.dep_ in ("nsubj", "nsubjpass")
                ]
            objs = [w for w in token.children if w.dep_ in ("dobj", "attr", "oprd")]
            if not objs and token.dep_ in {"relcl", "acl"} and subjs:
                if subjs[0] is not token.head and token.head.pos_ in ("NOUN", "PROPN"):
                    objs = [token.head]
            acomps = [w for w in token.children if w.dep_ == "acomp"]
            negs = [w for w in token.children if w.dep_ == "neg"]
            lemma = ("not " + token.lemma_) if negs else token.lemma_
            for s in subjs:
                for o in objs:
                    triples.append(
                        Triple(self._noun_phrase(s), lemma, self._noun_phrase(o))
                    )
                for a in acomps:
                    triples.append(Triple(self._noun_phrase(s), "is", a.lemma_))
                if not objs and not acomps and token.lemma_ != "be":
                    triples.append(Triple(self._noun_phrase(s), lemma, lemma))
        return triples

    def _extract_prep(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in (
                "NOUN", "PROPN", "VERB", "AUX", "ADJ"
            ):
                for pobj in token.children:
                    if pobj.dep_ == "pobj":
                        triples.append(
                            Triple(
                                self._noun_phrase(token.head),
                                self._compound_prep(token),
                                self._noun_phrase(pobj),
                            )
                        )
        return triples

    def _extract_existential(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.lemma_ in ("be", "have") and token.dep_ == "ROOT":
                if any(w.dep_ == "expl" for w in token.children):
                    for subj in token.children:
                        if subj.dep_ in ("nsubj", "attr") and subj.text != "there":
                            for prep in subj.children:
                                if prep.dep_ == "prep":
                                    for pobj in prep.children:
                                        if pobj.dep_ == "pobj":
                                            triples.append(
                                                Triple(
                                                    self._noun_phrase(subj),
                                                    self._compound_prep(prep),
                                                    self._noun_phrase(pobj),
                                                )
                                            )
        return triples

    def _extract_copular(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.lemma_ != "be" or token.dep_ != "ROOT":
                continue
            if any(w.dep_ == "expl" for w in token.children):
                continue
            subjs = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]
            attrs = [w for w in token.children if w.dep_ in ("attr", "acomp")]
            for s in subjs:
                for a in attrs:
                    obj_str = (
                        self._noun_phrase(a)
                        if a.pos_ in ("NOUN", "PROPN")
                        else a.lemma_
                    )
                    triples.append(Triple(self._noun_phrase(s), "is", obj_str))
        return triples

    def _extract_possessive(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.dep_ == "poss" and token.head.pos_ in ("NOUN", "PROPN"):
                triples.append(
                    Triple(
                        self._noun_phrase(token),
                        "has",
                        self._noun_phrase(token.head),
                    )
                )
        return triples


# ═════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════

def load_model(model_id="Qwen/Qwen3-VL-8B-Thinking", device="cuda:1"):
    log.info(f"Loading Qwen3-VL-Thinking: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    ).eval()
    log_gpu()
    return model, processor


# ═════════════════════════════════════════════════════════════
# Token helpers
# ═════════════════════════════════════════════════════════════

def _resolve_yes_no_ids(tokenizer, word: str) -> torch.Tensor:
    candidates = set()
    for variant in [word, word.capitalize(), word.upper()]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            candidates.add(ids[0])
    if not candidates:
        raise ValueError(f"Could not resolve token IDs for '{word}'")
    return torch.tensor(list(candidates), dtype=torch.long)


def _p_yes_from_logits(model, processor, inputs) -> float:
    """Run one forward pass; return P(yes) from the last-token logits."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    last = logits[0, -1]
    yes_ids = _resolve_yes_no_ids(processor.tokenizer, "yes")
    no_ids = _resolve_yes_no_ids(processor.tokenizer, "no")
    yes_logit = torch.logsumexp(last[yes_ids], dim=0)
    no_logit = torch.logsumexp(last[no_ids], dim=0)
    return torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0].item()


def _generate_text(model, processor, inputs, max_new_tokens=128) -> str:
    """Greedy decode; return decoded string (no special tokens)."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    # only the newly generated tokens
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ═════════════════════════════════════════════════════════════
# Prompt builders
# ═════════════════════════════════════════════════════════════

def _format_sg(triples: list[Triple]) -> str:
    return "\n".join(
        f"  - {t.subject}  [{t.relation}]  {t.obj}" for t in triples
    )


def _build_plain_prompt(caption: str) -> str:
    return (
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer with only 'yes' or 'no'."
    )


def _build_sg_prompt(caption: str, triples: list[Triple]) -> str:
    """Flat SG injection (original strategy)."""
    if not triples:
        return _build_plain_prompt(caption)
    sg = _format_sg(triples)
    return (
        f"Caption: '{caption}'\n\n"
        f"The caption has the following scene graph relations:\n{sg}\n\n"
        f"Using the scene graph as a guide, pay close attention to which "
        f"entity is doing what and any spatial relationships. "
        f"Does this image match the caption?\n"
        f"Answer with only 'yes' or 'no'."
    )


# ─── Multi-turn prompts ───────────────────────────────────────

def _build_mt_turn1(caption: str, triples: list[Triple]) -> str:
    """
    Turn 1: ask the model which SG triples (if any) are actually
    important for verifying this caption against the image.
    """
    sg = _format_sg(triples)
    return (
        f"I want to check whether an image matches this caption:\n"
        f"  \"{caption}\"\n\n"
        f"Here are the scene graph relations extracted from the caption:\n"
        f"{sg}\n\n"
        f"Which of these relations (if any) are the most important to verify "
        f"visually — e.g. who is doing what, spatial positions, or key attributes?\n"
        f"List only the relevant relation lines exactly as shown, one per line. "
        f"If none are important, reply with exactly: none"
    )


def _build_mt_turn2(caption: str, relevant_sg_text: str) -> str:
    """
    Turn 2: final yes/no question.
    relevant_sg_text is the model's Turn-1 reply (already filtered).
    If it contains 'none' (case-insensitive), we use the plain prompt.
    """
    cleaned = relevant_sg_text.strip().lower()
    if cleaned == "none" or not cleaned:
        return _build_plain_prompt(caption)
    return (
        f"Based on the scene graph relations you identified:\n"
        f"{relevant_sg_text.strip()}\n\n"
        f"Pay close attention to those specific relations. "
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer with only 'yes' or 'no'."
    )


def _parse_turn1_reply(reply: str) -> str:
    """
    Robustly extract the relevant-triples text from Turn 1.
    Returns 'none' if the model said nothing useful.
    """
    reply = reply.strip()
    if not reply:
        return "none"
    lower = reply.lower()
    # if the model just said "none" (possibly with punctuation)
    if re.fullmatch(r"none[.!]*", lower):
        return "none"
    # strip any preamble lines (e.g. "Sure! Here are the important ones:")
    lines = reply.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        # keep lines that look like triples: start with "  -" or contain "["
        if stripped.startswith("-") or "[" in stripped:
            kept.append(stripped.lstrip("- ").strip())
    if not kept:
        # if we can't parse structure, just return the whole reply
        return reply
    return "\n".join(kept)


# ═════════════════════════════════════════════════════════════
# Scoring functions
# ═════════════════════════════════════════════════════════════

@torch.no_grad()
def score_plain(model, processor, image: Image.Image, caption: str) -> float:
    """Single-turn, no SG."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _build_plain_prompt(caption)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return _p_yes_from_logits(model, processor, inputs)


@torch.no_grad()
def score_sg(
    model, processor, image: Image.Image, caption: str, triples: list[Triple]
) -> float:
    """Single-turn, all SG triples injected."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _build_sg_prompt(caption, triples)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return _p_yes_from_logits(model, processor, inputs)


@torch.no_grad()
def score_multiturn_sg(
    model,
    processor,
    image: Image.Image,
    caption: str,
    triples: list[Triple],
    max_new_tokens_turn1: int = 128,
) -> tuple[float, str, str]:
    """
    Two-turn SG-aware scoring.

    Turn 1  → model identifies which SG relations are important (text generation)
    Turn 2  → model answers yes/no using only those relations (P(yes) logits)

    Returns:
        p_yes          : float — P(yes) from Turn 2
        turn1_reply    : str   — raw Turn-1 model output
        relevant_sg    : str   — parsed relevant triples (or 'none')
    """
    # ── If no triples, skip Turn 1 entirely ──────────────────
    if not triples:
        p = score_plain(model, processor, image, caption)
        return p, "(no triples)", "none"

    # ── Turn 1: relevance filtering ──────────────────────────
    turn1_text = _build_mt_turn1(caption, triples)
    messages_t1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": turn1_text},
            ],
        }
    ]
    inputs_t1 = processor.apply_chat_template(
        messages_t1, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    turn1_reply = _generate_text(
        model, processor, inputs_t1, max_new_tokens=max_new_tokens_turn1
    )
    relevant_sg = _parse_turn1_reply(turn1_reply)

    # ── Turn 2: yes/no decision ───────────────────────────────
    turn2_text = _build_mt_turn2(caption, relevant_sg)

    # We include the image again because the model has no persistent state;
    # we reconstruct the conversation with the assistant's Turn-1 reply.
    messages_t2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": turn1_text},
            ],
        },
        {
            "role": "assistant",
            # content must be a list-of-typed-dicts, not a bare string.
            # apply_chat_template iterates content looking for {"type": ...}
            # keys; a plain str causes "string indices must be integers".
            "content": [{"type": "text", "text": turn1_reply}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": turn2_text},
            ],
        },
    ]
    inputs_t2 = processor.apply_chat_template(
        messages_t2, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    p_yes = _p_yes_from_logits(model, processor, inputs_t2)

    return p_yes, turn1_reply, relevant_sg


# ═════════════════════════════════════════════════════════════
# Winoground Metrics
# ═════════════════════════════════════════════════════════════

def winoground_metrics(s00, s10, s01, s11):
    text  = (s00 > s10) and (s11 > s01)
    image = (s00 > s01) and (s11 > s10)
    return text, image, text and image


def _make_row(s00, s10, s01, s11, tc, ic, gc):
    return {
        "scores":  {"c0_i0": round(s00, 5), "c1_i0": round(s10, 5),
                    "c0_i1": round(s01, 5), "c1_i1": round(s11, 5)},
        "correct": {"text": tc, "image": ic, "group": gc},
    }


# ═════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════

def evaluate(model, processor, sg_parser, dataset, split, max_samples, run_plain):
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    strategy_names = ["qwen3_gen_sg", "qwen3_gen_sg_mt"]
    if run_plain:
        strategy_names = ["qwen3_gen"] + strategy_names

    counts = {s: {"text": 0, "image": 0, "group": 0} for s in strategy_names}
    per_example = []

    for idx, example in enumerate(tqdm(data, desc="Qwen3-VL-Thinking")):
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        tag  = example.get("tag", "")

        t0 = sg_parser.parse(cap0)
        t1 = sg_parser.parse(cap1)

        row = {
            "idx": idx, "caption_0": cap0, "caption_1": cap1, "tag": tag,
            "sg_cap0": [repr(t) for t in t0],
            "sg_cap1": [repr(t) for t in t1],
        }

        # ── 1. Plain (no SG) ─────────────────────────────────
        if run_plain:
            p00 = score_plain(model, processor, img0, cap0)
            p10 = score_plain(model, processor, img0, cap1)
            p01 = score_plain(model, processor, img1, cap0)
            p11 = score_plain(model, processor, img1, cap1)
            tc, ic, gc = winoground_metrics(p00, p10, p01, p11)
            counts["qwen3_gen"]["text"]  += int(tc)
            counts["qwen3_gen"]["image"] += int(ic)
            counts["qwen3_gen"]["group"] += int(gc)
            row["qwen3_gen"] = _make_row(p00, p10, p01, p11, tc, ic, gc)

        # ── 2. Single-turn SG injection ───────────────────────
        q00 = score_sg(model, processor, img0, cap0, t0)
        q10 = score_sg(model, processor, img0, cap1, t1)
        q01 = score_sg(model, processor, img1, cap0, t0)
        q11 = score_sg(model, processor, img1, cap1, t1)
        tc, ic, gc = winoground_metrics(q00, q10, q01, q11)
        counts["qwen3_gen_sg"]["text"]  += int(tc)
        counts["qwen3_gen_sg"]["image"] += int(ic)
        counts["qwen3_gen_sg"]["group"] += int(gc)
        row["qwen3_gen_sg"] = _make_row(q00, q10, q01, q11, tc, ic, gc)

        # ── 3. Multi-turn SG (self-filtered) ──────────────────
        m00, r00, rel00 = score_multiturn_sg(model, processor, img0, cap0, t0)
        m10, r10, rel10 = score_multiturn_sg(model, processor, img0, cap1, t1)
        m01, r01, rel01 = score_multiturn_sg(model, processor, img1, cap0, t0)
        m11, r11, rel11 = score_multiturn_sg(model, processor, img1, cap1, t1)
        tc, ic, gc = winoground_metrics(m00, m10, m01, m11)
        counts["qwen3_gen_sg_mt"]["text"]  += int(tc)
        counts["qwen3_gen_sg_mt"]["image"] += int(ic)
        counts["qwen3_gen_sg_mt"]["group"] += int(gc)
        row["qwen3_gen_sg_mt"] = {
            **_make_row(m00, m10, m01, m11, tc, ic, gc),
            # store Turn-1 relevance decisions for analysis
            "turn1": {
                "c0_i0": {"reply": r00, "relevant": rel00},
                "c1_i0": {"reply": r10, "relevant": rel10},
                "c0_i1": {"reply": r01, "relevant": rel01},
                "c1_i1": {"reply": r11, "relevant": rel11},
            },
        }

        per_example.append(row)

        if (idx + 1) % 10 == 0:
            n = idx + 1
            for s in strategy_names:
                log.info(
                    f"[{s}] n={n} | text={counts[s]['text']/n:.3f} | "
                    f"image={counts[s]['image']/n:.3f} | "
                    f"group={counts[s]['group']/n:.3f}"
                )

    n = len(data)
    summary = {s: {k: v / n for k, v in c.items()} for s, c in counts.items()}
    summary["n_evaluated"] = n
    return summary, per_example, strategy_names


# ═════════════════════════════════════════════════════════════
# Tag-level Analysis
# ═════════════════════════════════════════════════════════════

def analyze_by_tag(per_example, strategy_names):
    tag_data = {}
    for ex in per_example:
        tag = ex.get("tag") or "untagged"
        if tag not in tag_data:
            tag_data[tag] = {s: {"text": 0, "image": 0, "group": 0, "n": 0}
                             for s in strategy_names}
        for s in strategy_names:
            if s not in ex:
                continue
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
                "text":  c["text"]  / n if n else 0,
                "image": c["image"] / n if n else 0,
                "group": c["group"] / n if n else 0,
                "n": n,
            }
    return result


# ═════════════════════════════════════════════════════════════
# Multi-turn Relevance Analysis
# ═════════════════════════════════════════════════════════════

def analyze_multiturn_relevance(per_example):
    """
    Compute statistics on how often the model found SG useful vs. said 'none'.
    Also shows cases where MT+SG > plain SG (SG actually helped).
    """
    slots = ["c0_i0", "c1_i0", "c0_i1", "c1_i1"]
    total, none_count, kept_count = 0, 0, 0
    mt_wins, mt_losses, ties = 0, 0, 0

    for ex in per_example:
        if "qwen3_gen_sg_mt" not in ex:
            continue
        mt   = ex["qwen3_gen_sg_mt"]
        base = ex.get("qwen3_gen_sg")
        for slot in slots:
            if slot not in mt.get("turn1", {}):
                continue
            total += 1
            rel = mt["turn1"][slot]["relevant"].strip().lower()
            if rel == "none":
                none_count += 1
            else:
                kept_count += 1
        # per-example group score comparison
        if base:
            mt_gc   = int(mt["correct"]["group"])
            base_gc = int(base["correct"]["group"])
            if mt_gc > base_gc:
                mt_wins += 1
            elif mt_gc < base_gc:
                mt_losses += 1
            else:
                ties += 1

    return {
        "total_slots": total,
        "sg_useful":   kept_count,
        "sg_ignored":  none_count,
        "sg_useful_pct": kept_count / total if total else 0,
        "mt_vs_sg_wins":   mt_wins,
        "mt_vs_sg_losses": mt_losses,
        "mt_vs_sg_ties":   ties,
    }


# ═════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════

LABELS = {
    "qwen3_gen":       "Qwen3-VL-Think (plain)",
    "qwen3_gen_sg":    "Qwen3-VL-Think + SG",
    "qwen3_gen_sg_mt": "Qwen3-VL-Think + SG (multi-turn)",
}


def print_summary(summary, strategy_names, tag_analysis=None, mt_rel=None):
    n = summary.get("n_evaluated", "?")
    W, G = 34, 8

    print(f"\n{'═' * 76}")
    print(f"  Winoground — Qwen3-VL-8B-Thinking  (n={n})")
    print(f"  Scoring: P(yes) from next-token logits (generative decoder)")
    print(f"  SG:      YUKINO-SG TextSceneGraphParser")
    print(f"  Multi-turn: Turn-1 = relevance filter, Turn-2 = yes/no decision")
    print(f"{'═' * 76}")
    print(f"  {'Strategy':<{W}}  {'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}")
    print(f"  {'-' * 62}")
    print(f"  {'Random chance':<{W}}  {'0.250':>{G}}  {'0.250':>{G}}  {'0.063':>{G}}")

    for s in strategy_names:
        if s not in summary:
            continue
        v = summary[s]
        label = LABELS.get(s, s)
        print(f"  {label:<{W}}  {v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}")

    print(f"{'═' * 76}")

    # Δ table
    if "qwen3_gen_sg" in summary and "qwen3_gen_sg_mt" in summary:
        base = summary["qwen3_gen_sg"]
        aug  = summary["qwen3_gen_sg_mt"]
        dt = aug["text"]  - base["text"]
        di = aug["image"] - base["image"]
        dg = aug["group"] - base["group"]
        print(f"\n  Multi-turn vs flat SG:  "
              f"text {dt:+.4f}  |  image {di:+.4f}  |  group {dg:+.4f}")

    if "qwen3_gen" in summary and "qwen3_gen_sg_mt" in summary:
        base = summary["qwen3_gen"]
        aug  = summary["qwen3_gen_sg_mt"]
        dt = aug["text"]  - base["text"]
        di = aug["image"] - base["image"]
        dg = aug["group"] - base["group"]
        print(f"  Multi-turn vs plain:    "
              f"text {dt:+.4f}  |  image {di:+.4f}  |  group {dg:+.4f}")

    # Multi-turn relevance stats
    if mt_rel:
        print(f"\n  ── Multi-turn SG relevance decisions ──")
        print(f"  Total (image, caption) slots scored : {mt_rel['total_slots']}")
        pct = mt_rel['sg_useful_pct'] * 100
        print(f"  SG kept as useful                   : "
              f"{mt_rel['sg_useful']}  ({pct:.1f}%)")
        print(f"  SG discarded ('none')                : {mt_rel['sg_ignored']}")
        print(f"  MT wins vs flat-SG (group score)    : {mt_rel['mt_vs_sg_wins']}")
        print(f"  MT losses vs flat-SG                : {mt_rel['mt_vs_sg_losses']}")
        print(f"  Ties                                : {mt_rel['mt_vs_sg_ties']}")

    if tag_analysis:
        for s in strategy_names:
            label = LABELS.get(s, s)
            print(f"\n  ── {label} per tag ──")
            print(f"  {'Tag':<30}  {'n':>4}  {'Text':>8}  {'Image':>8}  {'Group':>8}")
            print(f"  {'-' * 62}")
            for tag, data in sorted(tag_analysis.items()):
                if s not in data:
                    continue
                d = data[s]
                print(
                    f"  {tag:<30}  {d['n']:>4}  "
                    f"{d['text']:>8.3f}  {d['image']:>8.3f}  {d['group']:>8.3f}"
                )
    print()


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Winoground eval: Qwen3-VL-8B-Thinking (multi-turn SG)"
    )
    p.add_argument("--model_id",      default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--spacy_model",   default="en_core_web_sm")
    p.add_argument("--hf_token",      default=None)
    p.add_argument("--max_samples",   type=int, default=None)
    p.add_argument("--split",         default="test")
    p.add_argument("--output_dir",    default="./results_qwen3_gen")
    p.add_argument("--device",        default="cuda:1")
    p.add_argument("--tag_analysis",  action="store_true", default=True)
    p.add_argument(
        "--no_plain",
        action="store_true",
        default=False,
        help="Skip the plain (no-SG) baseline to save ~25%% inference time",
    )
    p.add_argument(
        "--mt_max_tokens",
        type=int,
        default=128,
        help="Max new tokens for Turn-1 relevance generation",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log.info("Qwen3-VL-Thinking multi-turn SG eval")
    log.info(f"Model: {args.model_id}")
    log.info(f"Multi-turn Turn-1 max_new_tokens: {args.mt_max_tokens}")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    sg_parser = TextSceneGraphParser(args.spacy_model)
    model, processor = load_model(args.model_id, args.device)

    yes_ids = _resolve_yes_no_ids(processor.tokenizer, "yes")
    no_ids  = _resolve_yes_no_ids(processor.tokenizer, "no")
    log.info(f"Token IDs — yes: {yes_ids.tolist()}, no: {no_ids.tolist()}")

    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    summary, per_example, strategy_names = evaluate(
        model, processor, sg_parser, dataset,
        args.split, args.max_samples,
        run_plain=not args.no_plain,
    )

    tag_analysis = (
        analyze_by_tag(per_example, strategy_names) if args.tag_analysis else None
    )
    mt_rel = analyze_multiturn_relevance(per_example)
    print_summary(summary, strategy_names, tag_analysis, mt_rel)

    # ── Save ─────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id":      args.model_id,
        "scoring":       "P(yes) from next-token logits (generative decoder)",
        "spacy_model":   args.spacy_model,
        "split":         args.split,
        "max_samples":   args.max_samples,
        "mt_max_tokens": args.mt_max_tokens,
        "strategies": {
            "qwen3_gen":       "Plain prompt, no SG",
            "qwen3_gen_sg":    "Single-turn: all triples injected",
            "qwen3_gen_sg_mt": (
                "Two-turn: Turn-1=relevance filter (greedy decode), "
                "Turn-2=yes/no (P(yes) logits)"
            ),
        },
    }
    for name, data in {
        "summary":     summary,
        "per_example": per_example,
        "tags":        tag_analysis or {},
        "mt_relevance": mt_rel,
        "config":      config,
    }.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"{name:<14} → {path}")

    log.info("Done.")


if __name__ == "__main__":
    main()