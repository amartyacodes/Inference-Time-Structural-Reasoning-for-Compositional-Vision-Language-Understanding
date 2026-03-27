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

SG SOURCE (new):
  Pass --sg_path to load pre-computed Qwen3-VL scene graph triples from the
  JSON produced by winoground_scene_graph_extract.py.
  Format per item:
    text_scene_graph_0/1.triples: list of {
        subject, predicate, object,
        subject_attributes, object_attributes,
        spatial_detail, subject_count, object_count
    }
  Falls back to spaCy TextSceneGraphParser if --sg_path is not provided.

Usage:
    python winoground_qwen3_gen.py --sg_path winoground_text_sgs.json
    python winoground_qwen3_gen.py --sg_path winoground_text_sgs.json --max_samples 50
    python winoground_qwen3_gen.py --no_plain   # skip plain baseline
    python winoground_qwen3_gen.py              # fallback: use spaCy SG
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
    # rich fields from Qwen3-SG (not used by spaCy fallback)
    subject_attributes: list = None
    object_attributes:  list = None
    spatial_detail:     str  = None
    subject_count:      int  = None
    object_count:       int  = None

    def __repr__(self):
        parts = []
        # include attributes in repr so the model sees them in prompts
        subj = self.subject
        if self.subject_attributes:
            subj = f"{' '.join(self.subject_attributes)} {subj}".strip()
        if self.subject_count and self.subject_count != 1:
            subj = f"{self.subject_count} {subj}"

        obj = self.obj
        if self.object_attributes:
            obj = f"{' '.join(self.object_attributes)} {obj}".strip()
        if self.object_count and self.object_count != 1:
            obj = f"{self.object_count} {obj}"

        rel = self.relation
        if self.spatial_detail:
            rel = f"{rel} ({self.spatial_detail})"

        return f"({subj}, {rel}, {obj})"


# ═════════════════════════════════════════════════════════════
# SG Source 1: Load from Qwen3-SG JSON
# ═════════════════════════════════════════════════════════════

class QwenSGLoader:
    """
    Loads pre-computed Qwen3-VL scene graph triples from
    winoground_scene_graph_extract.py output JSON.

    Provides the same .parse(caption) interface as TextSceneGraphParser
    so the rest of the eval code is unchanged, but also exposes
    .get_by_id(item_id, cap_idx) for direct lookup.
    """

    def __init__(self, sg_path: str):
        sg_path = Path(sg_path)
        if not sg_path.exists():
            raise FileNotFoundError(f"SG file not found: {sg_path}")

        with open(sg_path) as f:
            data = json.load(f)

        # Build two indexes:
        #   by_id[item_id][0 or 1]  -> list[Triple]   (fast lookup during eval loop)
        #   by_caption[caption]     -> list[Triple]   (fallback for .parse())
        self.by_id:      dict[int, dict[int, list[Triple]]] = {}
        self.by_caption: dict[str, list[Triple]]            = {}

        n_ok, n_empty, n_err = 0, 0, 0
        for item in data.get("items", []):
            item_id = item["id"]
            self.by_id[item_id] = {}
            for cap_idx in (0, 1):
                key  = f"text_scene_graph_{cap_idx}"
                cap  = item.get(f"caption_{cap_idx}", "")
                sg   = item.get(key, {})
                triples = self._convert(sg)
                self.by_id[item_id][cap_idx] = triples
                self.by_caption[cap] = triples
                if sg.get("parse_error"):
                    n_err += 1
                elif triples:
                    n_ok += 1
                else:
                    n_empty += 1

        log.info(
            f"QwenSGLoader: loaded {len(data.get('items', []))} items  "
            f"(ok={n_ok}, empty={n_empty}, parse_error={n_err})"
        )

    @staticmethod
    def _convert(sg: dict) -> list[Triple]:
        """Convert one text_scene_graph dict to a list of Triple objects."""
        if sg.get("parse_error") or not sg:
            return []
        triples = []
        for t in sg.get("triples", []):
            triples.append(Triple(
                subject            = t.get("subject", ""),
                relation           = t.get("predicate", ""),
                obj                = t.get("object", ""),
                subject_attributes = t.get("subject_attributes", []) or [],
                object_attributes  = t.get("object_attributes",  []) or [],
                spatial_detail     = t.get("spatial_detail"),
                subject_count      = t.get("subject_count"),
                object_count       = t.get("object_count"),
            ))
        return [t for t in triples if t.subject and t.obj]

    def get_by_id(self, item_id: int, cap_idx: int) -> list[Triple]:
        """Primary lookup — use this in the eval loop."""
        return self.by_id.get(item_id, {}).get(cap_idx, [])

    def parse(self, caption: str) -> list[Triple]:
        """Fallback interface compatible with TextSceneGraphParser."""
        return self.by_caption.get(caption, [])


# ═════════════════════════════════════════════════════════════
# SG Source 2: spaCy fallback (original YUKINO-SG parser)
# ═════════════════════════════════════════════════════════════

class TextSceneGraphParser:
    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
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

    def _get_adjectives(self, noun_token):
        adjs = []
        for child in noun_token.children:
            if child.dep_ in {"amod", "advmod"} and (
                child.pos_ in {"ADJ", "NOUN", "VERB"}
                or child.tag_ in ("JJ", "JJR", "JJS")
            ):
                adjs.extend(c.text for c in child.children if c.dep_ == "compound")
                adjs.append(child.text)
        return adjs

    def _noun_phrase(self, token):
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

    def _compound_prep(self, prep_token):
        text = prep_token.text
        for child in prep_token.children:
            if child.dep_ in ("pcomp", "fixed"):
                text = f"{text} {child.text}"
        return text

    def _extract_svo(self, doc):
        triples = []
        for token in doc:
            is_root = token.dep_ == "ROOT"
            has_subj = any(w.dep_ in ("nsubj", "nsubjpass") for w in token.children)
            is_subverb = token.dep_ in {
                "relcl", "acl", "advcl", "xcomp", "ccomp", "conj"
            } and (token.pos_ in ("VERB", "AUX") or has_subj)
            if not is_root and not is_subverb:
                continue
            if is_root and token.lemma_ == "be":
                continue
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

    def _extract_prep(self, doc):
        triples = []
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN", "VERB", "AUX", "ADJ"):
                for pobj in token.children:
                    if pobj.dep_ == "pobj":
                        triples.append(Triple(self._noun_phrase(token.head), self._compound_prep(token), self._noun_phrase(pobj)))
        return triples

    def _extract_existential(self, doc):
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

    def _extract_copular(self, doc):
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
                    obj_str = self._noun_phrase(a) if a.pos_ in ("NOUN", "PROPN") else a.lemma_
                    triples.append(Triple(self._noun_phrase(s), "is", obj_str))
        return triples

    def _extract_possessive(self, doc):
        triples = []
        for token in doc:
            if token.dep_ == "poss" and token.head.pos_ in ("NOUN", "PROPN"):
                triples.append(Triple(self._noun_phrase(token), "has", self._noun_phrase(token.head)))
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
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    last = logits[0, -1]
    yes_ids = _resolve_yes_no_ids(processor.tokenizer, "yes")
    no_ids  = _resolve_yes_no_ids(processor.tokenizer, "no")
    yes_logit = torch.logsumexp(last[yes_ids], dim=0)
    no_logit  = torch.logsumexp(last[no_ids],  dim=0)
    return torch.softmax(torch.stack([yes_logit, no_logit]), dim=0)[0].item()


def _generate_text(model, processor, inputs, max_new_tokens=128) -> str:
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    new_tokens = out[0, inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ═════════════════════════════════════════════════════════════
# Prompt builders
# ═════════════════════════════════════════════════════════════

def _format_sg(triples: list[Triple]) -> str:
    """
    Format triples for injection into prompts.
    Uses Triple.__repr__ which includes attributes, counts, and spatial_detail.
    """
    return "\n".join(f"  - {t!r}" for t in triples)


def _build_plain_prompt(caption: str) -> str:
    return (
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer with only 'yes' or 'no'."
    )


def _build_sg_prompt(caption: str, triples: list[Triple]) -> str:
    if not triples:
        return _build_plain_prompt(caption)
    sg = _format_sg(triples)
    return (
        f"Caption: '{caption}'\n\n"
        f"The caption has the following scene graph relations:\n{sg}\n\n"
        f"Using the scene graph as a guide, pay close attention to which "
        f"entity is doing what to whom, their attributes (e.g. old/young, "
        f"tall/short), counts, and any spatial relationships.\n"
        f"Does this image match the caption?\n"
        f"Answer with only 'yes' or 'no'."
    )


def _build_mt_turn1(caption: str, triples: list[Triple]) -> str:
    sg = _format_sg(triples)
    return (
        f"I want to check whether an image matches this caption:\n"
        f"  \"{caption}\"\n\n"
        f"Here are the scene graph relations extracted from the caption.\n"
        f"Each triple shows: (subject with attributes, relation, object with attributes)\n"
        f"{sg}\n\n"
        f"Which of these relations (if any) are the most important to verify "
        f"visually — e.g. who is the agent doing the action, who is receiving it, "
        f"key attributes like age/size/color, counts, or spatial positions?\n"
        f"List only the relevant relation lines exactly as shown, one per line. "
        f"If none are important, reply with exactly: none"
    )


def _build_mt_turn2(caption: str, relevant_sg_text: str) -> str:
    cleaned = relevant_sg_text.strip().lower()
    if cleaned == "none" or not cleaned:
        return _build_plain_prompt(caption)
    return (
        f"Based on the scene graph relations you identified:\n"
        f"{relevant_sg_text.strip()}\n\n"
        f"Pay close attention to those specific relations — especially who is "
        f"doing the action (agent) vs. receiving it (patient), their attributes, "
        f"and any counts or spatial details.\n"
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer with only 'yes' or 'no'."
    )


def _parse_turn1_reply(reply: str) -> str:
    reply = reply.strip()
    if not reply:
        return "none"
    if re.fullmatch(r"none[.!]*", reply.lower()):
        return "none"
    lines = reply.splitlines()
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("-") or "[" in stripped or "(" in stripped:
            kept.append(stripped.lstrip("- ").strip())
    return "\n".join(kept) if kept else reply


# ═════════════════════════════════════════════════════════════
# Scoring functions
# ═════════════════════════════════════════════════════════════

@torch.no_grad()
def score_plain(model, processor, image: Image.Image, caption: str) -> float:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": _build_plain_prompt(caption)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return _p_yes_from_logits(model, processor, inputs)


@torch.no_grad()
def score_sg(model, processor, image: Image.Image, caption: str, triples: list[Triple]) -> float:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": _build_sg_prompt(caption, triples)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    return _p_yes_from_logits(model, processor, inputs)


@torch.no_grad()
def score_multiturn_sg(
    model, processor, image: Image.Image,
    caption: str, triples: list[Triple],
    max_new_tokens_turn1: int = 128,
) -> tuple[float, str, str]:
    if not triples:
        p = score_plain(model, processor, image, caption)
        return p, "(no triples)", "none"

    # Turn 1: relevance filtering
    turn1_text = _build_mt_turn1(caption, triples)
    messages_t1 = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": turn1_text},
    ]}]
    inputs_t1 = processor.apply_chat_template(
        messages_t1, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    turn1_reply = _generate_text(model, processor, inputs_t1, max_new_tokens=max_new_tokens_turn1)
    relevant_sg = _parse_turn1_reply(turn1_reply)

    # Turn 2: yes/no decision
    turn2_text = _build_mt_turn2(caption, relevant_sg)
    messages_t2 = [
        {"role": "user",      "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": turn1_text},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": turn1_reply}]},
        {"role": "user",      "content": [{"type": "text", "text": turn2_text}]},
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
# Evaluation loop
# ═════════════════════════════════════════════════════════════

def evaluate(model, processor, sg_source, dataset, split, max_samples, run_plain):
    """
    sg_source: QwenSGLoader or TextSceneGraphParser
    If QwenSGLoader, uses .get_by_id(item_id, cap_idx) for fast lookup.
    Falls back to .parse(caption) for spaCy or cache misses.
    """
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    use_qwen_sg = isinstance(sg_source, QwenSGLoader)
    sg_type = "Qwen3-SG (pre-computed)" if use_qwen_sg else "spaCy YUKINO-SG"
    log.info(f"SG source: {sg_type}")

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
        item_id = example["id"]

        # ── Get triples ──────────────────────────────────────
        if use_qwen_sg:
            t0 = sg_source.get_by_id(item_id, 0)
            t1 = sg_source.get_by_id(item_id, 1)
            # warn if this item wasn't in the SG file
            if not t0 and not t1:
                log.warning(f"id={item_id}: no triples found in SG file, "
                             "falling back to caption lookup")
                t0 = sg_source.parse(cap0)
                t1 = sg_source.parse(cap1)
        else:
            t0 = sg_source.parse(cap0)
            t1 = sg_source.parse(cap1)

        row = {
            "idx": idx, "id": item_id,
            "caption_0": cap0, "caption_1": cap1, "tag": tag,
            "sg_source": sg_type,
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
# Analysis helpers (unchanged)
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


def analyze_multiturn_relevance(per_example):
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
        if base:
            mt_gc   = int(mt["correct"]["group"])
            base_gc = int(base["correct"]["group"])
            if mt_gc > base_gc:   mt_wins   += 1
            elif mt_gc < base_gc: mt_losses += 1
            else:                 ties      += 1
    return {
        "total_slots":       total,
        "sg_useful":         kept_count,
        "sg_ignored":        none_count,
        "sg_useful_pct":     kept_count / total if total else 0,
        "mt_vs_sg_wins":     mt_wins,
        "mt_vs_sg_losses":   mt_losses,
        "mt_vs_sg_ties":     ties,
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
    print(f"  Scoring: P(yes) from next-token logits")
    print(f"{'═' * 76}")
    print(f"  {'Strategy':<{W}}  {'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}")
    print(f"  {'-' * 62}")
    print(f"  {'Random chance':<{W}}  {'0.250':>{G}}  {'0.250':>{G}}  {'0.063':>{G}}")
    for s in strategy_names:
        if s not in summary:
            continue
        v = summary[s]
        print(f"  {LABELS.get(s, s):<{W}}  {v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}")
    print(f"{'═' * 76}")
    if "qwen3_gen_sg" in summary and "qwen3_gen_sg_mt" in summary:
        base, aug = summary["qwen3_gen_sg"], summary["qwen3_gen_sg_mt"]
        print(f"\n  Multi-turn vs flat SG:  "
              f"text {aug['text']-base['text']:+.4f}  |  "
              f"image {aug['image']-base['image']:+.4f}  |  "
              f"group {aug['group']-base['group']:+.4f}")
    if mt_rel:
        print(f"\n  ── Multi-turn SG relevance decisions ──")
        print(f"  Total slots scored   : {mt_rel['total_slots']}")
        print(f"  SG kept as useful    : {mt_rel['sg_useful']} ({mt_rel['sg_useful_pct']*100:.1f}%)")
        print(f"  SG discarded (none)  : {mt_rel['sg_ignored']}")
        print(f"  MT wins vs flat-SG   : {mt_rel['mt_vs_sg_wins']}")
        print(f"  MT losses vs flat-SG : {mt_rel['mt_vs_sg_losses']}")
    if tag_analysis:
        for s in strategy_names:
            print(f"\n  ── {LABELS.get(s, s)} per tag ──")
            print(f"  {'Tag':<30}  {'n':>4}  {'Text':>8}  {'Image':>8}  {'Group':>8}")
            print(f"  {'-' * 62}")
            for tag, data in sorted(tag_analysis.items()):
                if s not in data:
                    continue
                d = data[s]
                print(f"  {tag:<30}  {d['n']:>4}  {d['text']:>8.3f}  {d['image']:>8.3f}  {d['group']:>8.3f}")
    print()


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Winoground eval: Qwen3-VL-8B-Thinking (multi-turn SG)"
    )
    p.add_argument("--model_id",    default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--split",       default="test")
    p.add_argument("--output_dir",  default="./results_qwen3_gen")
    p.add_argument("--device",      default="cuda:1")
    p.add_argument("--tag_analysis", action="store_true", default=True)
    p.add_argument("--no_plain",    action="store_true", default=False,
                   help="Skip plain (no-SG) baseline")
    p.add_argument("--mt_max_tokens", type=int, default=128,
                   help="Max new tokens for Turn-1 relevance generation")
    # ── NEW ──────────────────────────────────────────────────
    p.add_argument(
        "--sg_path", default=None,
        help=(
            "Path to pre-computed Qwen3-SG JSON from winoground_scene_graph_extract.py. "
            "If not provided, falls back to spaCy TextSceneGraphParser."
        ),
    )
    p.add_argument(
        "--spacy_model", default="en_core_web_sm",
        help="spaCy model to use when --sg_path is not provided.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log.info("Qwen3-VL-Thinking multi-turn SG eval")
    log.info(f"Model   : {args.model_id}")
    log.info(f"SG path : {args.sg_path or 'None (using spaCy fallback)'}")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── SG source ────────────────────────────────────────────
    if args.sg_path:
        sg_source = QwenSGLoader(args.sg_path)
    else:
        log.info("No --sg_path provided, using spaCy TextSceneGraphParser")
        sg_source = TextSceneGraphParser(args.spacy_model)

    model, processor = load_model(args.model_id, args.device)

    yes_ids = _resolve_yes_no_ids(processor.tokenizer, "yes")
    no_ids  = _resolve_yes_no_ids(processor.tokenizer, "no")
    log.info(f"Token IDs — yes: {yes_ids.tolist()}, no: {no_ids.tolist()}")

    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    summary, per_example, strategy_names = evaluate(
        model, processor, sg_source, dataset,
        args.split, args.max_samples,
        run_plain=not args.no_plain,
    )

    tag_analysis = analyze_by_tag(per_example, strategy_names) if args.tag_analysis else None
    mt_rel = analyze_multiturn_relevance(per_example)
    print_summary(summary, strategy_names, tag_analysis, mt_rel)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id":      args.model_id,
        "sg_source":     args.sg_path or "spaCy",
        "scoring":       "P(yes) from next-token logits",
        "split":         args.split,
        "max_samples":   args.max_samples,
        "mt_max_tokens": args.mt_max_tokens,
    }
    for name, data in {
        "summary":      summary,
        "per_example":  per_example,
        "tags":         tag_analysis or {},
        "mt_relevance": mt_rel,
        "config":       config,
    }.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"{name:<14} → {path}")

    log.info("Done.")


if __name__ == "__main__":
    main()