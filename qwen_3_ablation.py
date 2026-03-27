"""
Winoground Ablation: Qwen3-VL-8B-Thinking — Caption Masking Experiments
========================================================================

Tests whether Qwen3-VL itself understands compositional structure by
masking/corrupting specific roles (subject, object, relation) IN THE
CAPTION TEXT fed to the model. No scene graph prompt is used — the SG
parser is only used to IDENTIFY which spans in the caption correspond
to subjects, objects, and relations, then those spans are masked.

This directly answers: does the model rely on entity identity, relational
structure, or role binding to perform compositional reasoning?

ABLATION CONDITIONS (all modify the caption, no SG in prompt):
  plain              — original caption, no modification
  mask_subj          — subject spans in caption → [MASK]
  mask_obj           — object spans in caption → [MASK]
  mask_rel           — relation/verb spans in caption → [MASK]
  mask_subj_obj      — both subject and object spans → [MASK]
  mask_all           — all identified spans (subj+rel+obj) → [MASK]
  swap_subj_obj      — swap subject ↔ object text in caption
  shuffle_entities   — randomly reassign entity spans across roles
  replace_subj_rand  — replace subject spans with random nouns
  replace_obj_rand   — replace object spans with random nouns

INTERPRETATION:
  plain > mask_subj?         → model needs subject identity
  plain > mask_obj?          → model needs object identity
  plain > mask_rel?          → model uses relational/verb info
  plain > mask_subj_obj?     → entities are the key signal
  plain > swap_subj_obj?     → model understands who-does-what
    (this is the STRONGEST compositional test — if swap ≈ plain,
     the model treats the caption as a bag of words)
  mask_subj ≈ mask_obj?      → both roles equally important
  mask_all ≈ random chance?  → all compositional info destroyed

Usage:
    python winoground_qwen3_caption_ablation.py --max_samples 50
    python winoground_qwen3_caption_ablation.py
"""

import gc
import json
import random
import re
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

from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    raise ImportError("Qwen3VLForConditionalGeneration not found. pip install transformers>=4.57.0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MASK_TOKEN = "[MASK]"

RANDOM_NOUNS = [
    "table", "river", "planet", "shoe", "clock", "mountain", "guitar",
    "window", "candle", "robot", "whale", "forest", "bridge", "lamp",
    "feather", "diamond", "ocean", "castle", "engine", "mirror",
    "cloud", "ladder", "basket", "flame", "crystal", "shadow", "tower",
    "anchor", "blanket", "compass", "dragon", "eagle", "fountain",
]


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


@dataclass
class SpanInfo:
    """Tracks character-level spans of SG components in the original caption."""
    subjects:  list  # list of (start, end, text)
    objects:   list
    relations: list


# ═════════════════════════════════════════════════════════════
# Text Scene Graph Parser (YUKINO-SG) — extended with span tracking
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
        self._span_cache: dict[str, SpanInfo] = {}

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

    def extract_spans(self, caption: str) -> SpanInfo:
        """
        Parse caption and return character-level spans for subjects,
        objects, and relations as found by the dependency parser.
        Works on the ORIGINAL caption (not lowered) by mapping back.
        """
        if caption in self._span_cache:
            return self._span_cache[caption]

        lower = caption.lower().strip()
        doc = self.nlp(lower)

        subj_spans = []
        obj_spans = []
        rel_spans = []

        seen_subj = set()
        seen_obj = set()
        seen_rel = set()

        for token in doc:
            # Find subjects
            if token.dep_ in ("nsubj", "nsubjpass"):
                span = self._get_constituent_span(token, doc)
                key = (span[0], span[1])
                if key not in seen_subj:
                    seen_subj.add(key)
                    subj_spans.append(span)

            # Find objects
            if token.dep_ in ("dobj", "pobj", "attr", "oprd"):
                span = self._get_constituent_span(token, doc)
                key = (span[0], span[1])
                if key not in seen_obj:
                    seen_obj.add(key)
                    obj_spans.append(span)

            # Find relation verbs and prepositions
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                s, e = token.idx, token.idx + len(token.text)
                key = (s, e)
                if key not in seen_rel:
                    seen_rel.add(key)
                    rel_spans.append((s, e, token.text))
            if token.dep_ == "prep":
                s, e = token.idx, token.idx + len(token.text)
                key = (s, e)
                if key not in seen_rel:
                    seen_rel.add(key)
                    rel_spans.append((s, e, token.text))

        # Map spans back to original caption casing
        info = SpanInfo(
            subjects=self._remap_spans(subj_spans, caption, lower),
            objects=self._remap_spans(obj_spans, caption, lower),
            relations=self._remap_spans(rel_spans, caption, lower),
        )
        self._span_cache[caption] = info
        return info

    def _get_constituent_span(self, token, doc):
        """Get the full noun phrase span for a token (including modifiers, compounds)."""
        subtree_tokens = sorted(token.subtree, key=lambda t: t.i)

        # Filter to relevant tokens (stop at verbs/clauses)
        phrase_tokens = []
        for t in subtree_tokens:
            if t.pos_ in ("VERB", "AUX") and t != token:
                continue
            if t.dep_ in ("relcl", "acl", "advcl"):
                continue
            phrase_tokens.append(t)

        if not phrase_tokens:
            phrase_tokens = [token]

        start = phrase_tokens[0].idx
        last = phrase_tokens[-1]
        end = last.idx + len(last.text)
        text = doc.text[start:end]

        return (start, end, text)

    def _remap_spans(self, spans, original, lower):
        """Remap spans from lowered text to original caption."""
        result = []
        for start, end, text in spans:
            orig_text = original[start:end] if start < len(original) else text
            result.append((start, end, orig_text))
        return result

    # ── standard extraction methods ──

    def _get_adjectives(self, noun_token) -> list[str]:
        adjs = []
        for child in noun_token.children:
            if child.dep_ in {"amod", "advmod"} and (
                child.pos_ in {"ADJ", "NOUN", "VERB"} or child.tag_ in ("JJ", "JJR", "JJS")
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
                if t == noun_token: break
                if t.tag_ in ("JJ", "JJR", "JJS") or (t.dep_ == "amod" and t.pos_ == "ADJ"):
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
            is_root = (token.dep_ == "ROOT")
            has_subj = any(w.dep_ in ("nsubj", "nsubjpass") for w in token.children)
            is_subverb = (token.dep_ in {"relcl", "acl", "advcl", "xcomp", "ccomp", "conj"} and
                          (token.pos_ in ("VERB", "AUX") or has_subj))
            if not is_root and not is_subverb: continue
            if is_root and token.lemma_ == "be": continue
            subjs = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]
            if not subjs and token.dep_ in {"relcl", "acl"} and token.head.pos_ in ("NOUN", "PROPN"):
                subjs = [token.head]
            if not subjs and token.dep_ in {"advcl", "xcomp", "ccomp", "conj"} and token.head.pos_ in ("VERB", "AUX"):
                subjs = [w for w in token.head.children if w.dep_ in ("nsubj", "nsubjpass")]
            objs = [w for w in token.children if w.dep_ in ("dobj", "attr", "oprd")]
            if not objs and token.dep_ in {"relcl", "acl"} and subjs:
                if subjs[0] is not token.head and token.head.pos_ in ("NOUN", "PROPN"):
                    objs = [token.head]
            acomps = [w for w in token.children if w.dep_ == "acomp"]
            negs = [w for w in token.children if w.dep_ == "neg"]
            lemma = ("not " + token.lemma_) if negs else token.lemma_
            for s in subjs:
                for o in objs:
                    triples.append(Triple(self._noun_phrase(s), lemma, self._noun_phrase(o)))
                for a in acomps:
                    triples.append(Triple(self._noun_phrase(s), "is", a.lemma_))
                if not objs and not acomps and token.lemma_ != "be":
                    triples.append(Triple(self._noun_phrase(s), lemma, lemma))
        return triples

    def _extract_prep(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN", "VERB", "AUX", "ADJ"):
                for pobj in token.children:
                    if pobj.dep_ == "pobj":
                        triples.append(Triple(self._noun_phrase(token.head), self._compound_prep(token), self._noun_phrase(pobj)))
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
                                            triples.append(Triple(self._noun_phrase(subj), self._compound_prep(prep), self._noun_phrase(pobj)))
        return triples

    def _extract_copular(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.lemma_ != "be" or token.dep_ != "ROOT": continue
            if any(w.dep_ == "expl" for w in token.children): continue
            subjs = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]
            attrs = [w for w in token.children if w.dep_ in ("attr", "acomp")]
            for s in subjs:
                for a in attrs:
                    obj_str = self._noun_phrase(a) if a.pos_ in ("NOUN", "PROPN") else a.lemma_
                    triples.append(Triple(self._noun_phrase(s), "is", obj_str))
        return triples

    def _extract_possessive(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.dep_ == "poss" and token.head.pos_ in ("NOUN", "PROPN"):
                triples.append(Triple(self._noun_phrase(token), "has", self._noun_phrase(token.head)))
        return triples


# ═════════════════════════════════════════════════════════════
# Caption Ablation Transforms
# ═════════════════════════════════════════════════════════════

def _apply_span_replacement(caption: str, spans: list, replacement: str) -> str:
    """Replace character spans in caption, processing right-to-left to preserve offsets."""
    if not spans:
        return caption
    sorted_spans = sorted(spans, key=lambda s: s[0], reverse=True)
    result = caption
    replaced = set()
    for start, end, text in sorted_spans:
        overlap = False
        for rs, re_ in replaced:
            if start < re_ and end > rs:
                overlap = True
                break
        if overlap:
            continue
        result = result[:start] + replacement + result[end:]
        replaced.add((start, start + len(replacement)))
    return result


def _apply_span_swap(caption: str, spans_a: list, spans_b: list) -> str:
    """Swap text between two sets of spans (subjects ↔ objects)."""
    if not spans_a or not spans_b:
        return caption
    pairs = list(zip(spans_a, spans_b))
    replacements = []
    for (sa_start, sa_end, sa_text), (sb_start, sb_end, sb_text) in pairs:
        replacements.append((sa_start, sa_end, sb_text))
        replacements.append((sb_start, sb_end, sa_text))
    replacements.sort(key=lambda r: r[0], reverse=True)
    result = caption
    last_start = len(caption)
    for start, end, new_text in replacements:
        if end > last_start:
            continue
        result = result[:start] + new_text + result[end:]
        last_start = start
    return result


def _apply_span_shuffle(caption: str, spans: list, rng: random.Random) -> str:
    """Randomly reassign text across a set of spans."""
    if len(spans) <= 1:
        return caption
    texts = [s[2] for s in spans]
    rng.shuffle(texts)
    sorted_spans = sorted(zip(spans, texts), key=lambda x: x[0][0], reverse=True)
    result = caption
    for (start, end, _orig), new_text in sorted_spans:
        result = result[:start] + new_text + result[end:]
    return result


def _apply_span_random(caption: str, spans: list, rng: random.Random) -> str:
    """Replace spans with random nouns."""
    if not spans:
        return caption
    sorted_spans = sorted(spans, key=lambda s: s[0], reverse=True)
    result = caption
    for start, end, _text in sorted_spans:
        result = result[:start] + rng.choice(RANDOM_NOUNS) + result[end:]
    return result


def ablate_caption(caption: str, span_info: SpanInfo, condition: str,
                   rng: random.Random = None) -> str:
    """Apply an ablation condition to the caption text using identified spans."""
    if condition == "plain":
        return caption
    elif condition == "mask_subj":
        return _apply_span_replacement(caption, span_info.subjects, MASK_TOKEN)
    elif condition == "mask_obj":
        return _apply_span_replacement(caption, span_info.objects, MASK_TOKEN)
    elif condition == "mask_rel":
        return _apply_span_replacement(caption, span_info.relations, MASK_TOKEN)
    elif condition == "mask_subj_obj":
        return _apply_span_replacement(caption, span_info.subjects + span_info.objects, MASK_TOKEN)
    elif condition == "mask_all":
        all_spans = span_info.subjects + span_info.objects + span_info.relations
        return _apply_span_replacement(caption, all_spans, MASK_TOKEN)
    elif condition == "swap_subj_obj":
        return _apply_span_swap(caption, span_info.subjects, span_info.objects)
    elif condition == "shuffle_entities":
        all_entity_spans = span_info.subjects + span_info.objects
        return _apply_span_shuffle(caption, all_entity_spans, rng)
    elif condition == "replace_subj_rand":
        return _apply_span_random(caption, span_info.subjects, rng)
    elif condition == "replace_obj_rand":
        return _apply_span_random(caption, span_info.objects, rng)
    else:
        raise ValueError(f"Unknown ablation condition: {condition}")


ABLATION_CONDITIONS = [
    "plain",
    "mask_subj",
    "mask_obj",
    "mask_rel",
    "mask_subj_obj",
    "mask_all",
    "swap_subj_obj",
    "shuffle_entities",
    "replace_subj_rand",
    "replace_obj_rand",
]

NEEDS_RNG = {"shuffle_entities", "replace_subj_rand", "replace_obj_rand"}


# ═════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════

def load_model(model_id="Qwen/Qwen3-VL-8B-Thinking", device="cuda:0"):
    log.info(f"Loading Qwen3-VL-Thinking: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    ).eval()
    if torch.cuda.is_available():
        log.info(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
    return model, processor


# ═════════════════════════════════════════════════════════════
# Scoring
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


@torch.no_grad()
def score(model, processor, image, caption):
    """
    P(yes) for 'does this image match the caption?'
    Caption is passed directly — any masking is already applied.
    No SG prompt — just the (possibly ablated) caption.
    """
    text = (
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer with only 'yes' or 'no'."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
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
# Winoground Metrics
# ═════════════════════════════════════════════════════════════

def winoground_metrics(s00, s10, s01, s11):
    text  = (s00 > s10) and (s11 > s01)
    image = (s00 > s01) and (s11 > s10)
    return text, image, text and image


def _make_row(s00, s10, s01, s11, tc, ic, gc):
    return {
        "scores": {"c0_i0": round(s00, 5), "c1_i0": round(s10, 5),
                    "c0_i1": round(s01, 5), "c1_i1": round(s11, 5)},
        "correct": {"text": tc, "image": ic, "group": gc},
    }


# ═════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════

def evaluate(model, processor, sg_parser, dataset, split, max_samples, seed=42):
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    rng = random.Random(seed)

    counts = {c: {"text": 0, "image": 0, "group": 0} for c in ABLATION_CONDITIONS}
    per_example = []
    n_with_spans = 0

    for idx, example in enumerate(tqdm(data, desc="Caption Ablation")):
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        tag  = example.get("tag", "")

        spans0 = sg_parser.extract_spans(cap0)
        spans1 = sg_parser.extract_spans(cap1)

        has_spans = bool(spans0.subjects or spans0.objects) or bool(spans1.subjects or spans1.objects)
        if has_spans:
            n_with_spans += 1

        row = {
            "idx": idx, "caption_0": cap0, "caption_1": cap1, "tag": tag,
            "has_spans": has_spans,
            "spans_cap0": {
                "subjects": [(s, e, t) for s, e, t in spans0.subjects],
                "objects": [(s, e, t) for s, e, t in spans0.objects],
                "relations": [(s, e, t) for s, e, t in spans0.relations],
            },
            "spans_cap1": {
                "subjects": [(s, e, t) for s, e, t in spans1.subjects],
                "objects": [(s, e, t) for s, e, t in spans1.objects],
                "relations": [(s, e, t) for s, e, t in spans1.relations],
            },
        }

        for cond in ABLATION_CONDITIONS:
            r = rng if cond in NEEDS_RNG else None

            abl_cap0 = ablate_caption(cap0, spans0, cond, r)
            abl_cap1 = ablate_caption(cap1, spans1, cond, r)

            s00 = score(model, processor, img0, abl_cap0)
            s10 = score(model, processor, img0, abl_cap1)
            s01 = score(model, processor, img1, abl_cap0)
            s11 = score(model, processor, img1, abl_cap1)

            tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
            counts[cond]["text"] += int(tc)
            counts[cond]["image"] += int(ic)
            counts[cond]["group"] += int(gc)

            row[cond] = _make_row(s00, s10, s01, s11, tc, ic, gc)
            row[f"{cond}_cap0"] = abl_cap0
            row[f"{cond}_cap1"] = abl_cap1

        per_example.append(row)

        if (idx + 1) % 10 == 0:
            n = idx + 1
            for c in ABLATION_CONDITIONS:
                log.info(f"[{c}] n={n} | text={counts[c]['text']/n:.3f} | "
                         f"image={counts[c]['image']/n:.3f} | group={counts[c]['group']/n:.3f}")

    n = len(data)
    summary = {c: {k: v / n for k, v in cnt.items()} for c, cnt in counts.items()}
    summary["n_evaluated"] = n
    summary["n_with_spans"] = n_with_spans

    # Metrics restricted to examples WITH extracted spans
    if n_with_spans > 0:
        sg_counts = {c: {"text": 0, "image": 0, "group": 0} for c in ABLATION_CONDITIONS}
        for ex in per_example:
            if not ex["has_spans"]:
                continue
            for c in ABLATION_CONDITIONS:
                if c in ex:
                    sg_counts[c]["text"] += int(ex[c]["correct"]["text"])
                    sg_counts[c]["image"] += int(ex[c]["correct"]["image"])
                    sg_counts[c]["group"] += int(ex[c]["correct"]["group"])
        summary["spans_only"] = {
            c: {k: v / n_with_spans for k, v in cnt.items()}
            for c, cnt in sg_counts.items()
        }

    return summary, per_example


# ═════════════════════════════════════════════════════════════
# Tag-level Analysis
# ═════════════════════════════════════════════════════════════

def analyze_by_tag(per_example, conditions):
    tag_data = {}
    for ex in per_example:
        tag = ex.get("tag") or "untagged"
        if tag not in tag_data:
            tag_data[tag] = {c: {"text": 0, "image": 0, "group": 0, "n": 0} for c in conditions}
        for c in conditions:
            if c not in ex: continue
            tag_data[tag][c]["text"]  += int(ex[c]["correct"]["text"])
            tag_data[tag][c]["image"] += int(ex[c]["correct"]["image"])
            tag_data[tag][c]["group"] += int(ex[c]["correct"]["group"])
            tag_data[tag][c]["n"]     += 1
    result = {}
    for tag, data in tag_data.items():
        result[tag] = {}
        for c, cnt in data.items():
            n = cnt["n"]
            result[tag][c] = {
                "text": cnt["text"] / n if n else 0,
                "image": cnt["image"] / n if n else 0,
                "group": cnt["group"] / n if n else 0,
                "n": n,
            }
    return result


# ═════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════

CONDITION_LABELS = {
    "plain":             ("Original caption",     "baseline"),
    "mask_subj":         ("Mask subjects",        "entity"),
    "mask_obj":          ("Mask objects",         "entity"),
    "mask_rel":          ("Mask relations",       "relation"),
    "mask_subj_obj":     ("Mask subj+obj",        "entity"),
    "mask_all":          ("Mask all roles",       "total"),
    "swap_subj_obj":     ("Swap subj↔obj",        "composit."),
    "shuffle_entities":  ("Shuffle entities",     "composit."),
    "replace_subj_rand": ("Random subjects",      "entity"),
    "replace_obj_rand":  ("Random objects",       "entity"),
}


def print_summary(summary, tag_analysis=None):
    n = summary.get("n_evaluated", "?")
    n_sp = summary.get("n_with_spans", "?")
    W, G = 24, 8

    print(f"\n{'═' * 82}")
    print(f"  Winoground Caption Ablation — Qwen3-VL-8B-Thinking  (n={n}, {n_sp} with spans)")
    print(f"  Scoring: P(yes) from next-token logits | NO scene graph prompt")
    print(f"  Ablation target: the CAPTION TEXT itself (spans identified by SG parser)")
    print(f"{'═' * 82}")
    print(f"  {'Condition':<{W}}  {'Tests':<10}  {'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}  {'Δgrp':>{G}}")
    print(f"  {'-' * 76}")
    print(f"  {'Random chance':<{W}}  {'—':<10}  {'0.250':>{G}}  {'0.250':>{G}}  {'0.063':>{G}}  {'':>{G}}")

    base_group = summary.get("plain", {}).get("group", 0)

    for cond in ABLATION_CONDITIONS:
        if cond not in summary or cond in ("n_evaluated", "n_with_spans", "spans_only"):
            continue
        label, test_type = CONDITION_LABELS.get(cond, (cond, "?"))
        v = summary[cond]
        if cond == "plain":
            delta_str = "—"
        else:
            delta = v["group"] - base_group
            delta_str = f"{delta:+.4f}"
        print(f"  {label:<{W}}  {test_type:<10}  "
              f"{v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}  "
              f"{delta_str:>{G}}")

    print(f"{'═' * 82}")

    # Spans-only subset
    if "spans_only" in summary:
        print(f"\n  ── Restricted to examples WITH extracted spans (n={n_sp}) ──")
        print(f"  {'Condition':<{W}}  {'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}  {'Δgrp':>{G}}")
        print(f"  {'-' * 56}")

        sp_base = summary["spans_only"].get("plain", {}).get("group", 0)

        for cond in ABLATION_CONDITIONS:
            if cond not in summary["spans_only"]:
                continue
            label = CONDITION_LABELS.get(cond, (cond,))[0]
            v = summary["spans_only"][cond]
            if cond == "plain":
                delta_str = "—"
            else:
                delta_str = f"{v['group'] - sp_base:+.4f}"
            print(f"  {label:<{W}}  "
                  f"{v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}  "
                  f"{delta_str:>{G}}")

    # Interpretation
    print(f"\n  ── Interpretation Guide ──")
    print(f"  Δgrp = condition group score − plain (original caption) group score")
    print(f"  Negative Δgrp → masking that role HURTS → model relies on it")
    print(f"  Δgrp ≈ 0      → model doesn't use that information")
    print()
    print(f"  Key questions this answers:")
    print(f"    mask_subj < plain?           → Qwen3-VL needs subject identity")
    print(f"    mask_obj < plain?            → Qwen3-VL needs object identity")
    print(f"    mask_subj ≈ mask_obj?        → both roles equally important")
    print(f"    mask_rel < plain?            → Qwen3-VL uses verb/preposition info")
    print(f"    mask_subj_obj << plain?      → entities are the key signal")
    print(f"    mask_all ≈ random chance?    → all compositional info destroyed")
    print(f"    swap_subj_obj < plain?       → Qwen3-VL understands WHO does WHAT")
    print(f"       (strongest compositional test — if swap ≈ plain, model is")
    print(f"        treating caption as bag-of-words, not compositionally)")
    print(f"    shuffle < plain?             → entity-role binding matters")

    if tag_analysis:
        for cond in ["plain", "mask_subj", "mask_obj", "swap_subj_obj"]:
            label = CONDITION_LABELS.get(cond, (cond,))[0]
            print(f"\n  ── {label} per tag ──")
            print(f"  {'Tag':<30}  {'n':>4}  {'Text':>8}  {'Image':>8}  {'Group':>8}")
            print(f"  {'-' * 62}")
            for tag, data in sorted(tag_analysis.items()):
                if cond not in data: continue
                d = data[cond]
                print(f"  {tag:<30}  {d['n']:>4}  {d['text']:>8.3f}  {d['image']:>8.3f}  {d['group']:>8.3f}")
    print()


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Winoground caption ablation: Qwen3-VL-8B-Thinking")
    p.add_argument("--model_id",      default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--spacy_model",   default="en_core_web_sm")
    p.add_argument("--hf_token",      default=None)
    p.add_argument("--max_samples",   type=int, default=None)
    p.add_argument("--split",         default="test")
    p.add_argument("--output_dir",    default="./results_qwen3_caption_ablation")
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--tag_analysis",  action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"Qwen3-VL-Thinking CAPTION ablation study")
    log.info(f"Model: {args.model_id}")
    log.info(f"Ablation conditions: {ABLATION_CONDITIONS}")
    log.info(f"NOTE: SG parser used ONLY to identify spans — no SG in prompt")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    sg_parser = TextSceneGraphParser(args.spacy_model)
    model, processor = load_model(args.model_id, args.device)

    # Verify token resolution
    yes_ids = _resolve_yes_no_ids(processor.tokenizer, "yes")
    no_ids = _resolve_yes_no_ids(processor.tokenizer, "no")
    log.info(f"Token IDs — yes: {yes_ids.tolist()}, no: {no_ids.tolist()}")

    # Demo: show ablation on a sample caption
    demo_cap = "a dog chasing a cat across the yard"
    demo_spans = sg_parser.extract_spans(demo_cap)
    log.info(f"Demo caption: '{demo_cap}'")
    log.info(f"  subjects:  {demo_spans.subjects}")
    log.info(f"  objects:   {demo_spans.objects}")
    log.info(f"  relations: {demo_spans.relations}")
    for cond in ABLATION_CONDITIONS:
        r = random.Random(42) if cond in NEEDS_RNG else None
        abl = ablate_caption(demo_cap, demo_spans, cond, r)
        log.info(f"  {cond:<20} → '{abl}'")

    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    summary, per_example = evaluate(
        model, processor, sg_parser, dataset, args.split,
        args.max_samples, seed=args.seed,
    )

    tag_analysis = analyze_by_tag(per_example, ABLATION_CONDITIONS) if args.tag_analysis else None
    print_summary(summary, tag_analysis)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": args.model_id,
        "scoring": "P(yes) from next-token logits (NO scene graph in prompt)",
        "ablation_target": "caption text itself — spans identified by SG parser",
        "spacy_model": args.spacy_model,
        "split": args.split,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "ablation_conditions": {
            cond: CONDITION_LABELS.get(cond, (cond, "?"))[0]
            for cond in ABLATION_CONDITIONS
        },
        "ablation_descriptions": {
            "plain": "original caption, no modification",
            "mask_subj": "subject noun phrases in caption → [MASK]",
            "mask_obj": "object noun phrases in caption → [MASK]",
            "mask_rel": "relation verbs/prepositions in caption → [MASK]",
            "mask_subj_obj": "both subject and object phrases in caption → [MASK]",
            "mask_all": "all identified spans in caption → [MASK]",
            "swap_subj_obj": "swap subject ↔ object text within caption",
            "shuffle_entities": "randomly reassign entity text across all entity spans in caption",
            "replace_subj_rand": "replace subject phrases in caption with random nouns",
            "replace_obj_rand": "replace object phrases in caption with random nouns",
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