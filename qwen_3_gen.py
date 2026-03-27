"""
Winoground Evaluation: Qwen3-VL-8B-Thinking (Generative P(yes))
================================================================

Standalone evaluation script for the Qwen3-VL-8B-Thinking decoder model
on the Winoground benchmark. Scores via P(yes) from next-token logits,
directly comparable to the LLaVA P(yes) approach.

STRATEGIES:
  qwen3_gen       — plain P(yes)
  qwen3_gen_sg    — P(yes) with YUKINO-SG triples injected into prompt

Usage:
    python winoground_qwen3_gen.py --max_samples 50
    python winoground_qwen3_gen.py --model_id Qwen/Qwen3-VL-8B-Thinking
    python winoground_qwen3_gen.py --lam 0.0   # disable SG
"""

import gc
import json
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

    # ── internal extraction helpers ──

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
    log_gpu()
    return model, processor


# ═════════════════════════════════════════════════════════════
# Scoring
# ═════════════════════════════════════════════════════════════

def _format_sg(triples):
    return "\n".join(f"  - {t.subject}  [{t.relation}]  {t.obj}" for t in triples)


def _build_prompt(caption, triples, use_sg):
    """
    Build text content for Qwen3-VL yes/no matching prompt.
    use_sg=True AND triples non-empty → inject SG into prompt.
    Otherwise → plain prompt.
    """
    if use_sg and triples:
        sg = _format_sg(triples)
        return (
            f"Caption: '{caption}'\n\n"
            f"The caption has the following scene graph relations:\n{sg}\n\n"
            f"Using the scene graph as a guide, pay close attention to which "
            f"entity is doing what and any spatial relationships. "
            f"Does this image match the caption?\n"
            f"Answer with only 'yes' or 'no'."
        )
    else:
        return (
            f"Does this image match the caption: '{caption}'?\n"
            f"Answer with only 'yes' or 'no'."
        )


def _resolve_yes_no_ids(tokenizer, word: str) -> torch.Tensor:
    """Resolve token IDs for a word, handling casing variants."""
    candidates = set()
    for variant in [word, word.capitalize(), word.upper()]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            candidates.add(ids[0])
    if not candidates:
        raise ValueError(f"Could not resolve token IDs for '{word}'")
    return torch.tensor(list(candidates), dtype=torch.long)


@torch.no_grad()
def score(model, processor, image, caption, triples, use_sg):
    """
    P(yes) via next-token logits from Qwen3-VL-8B-Thinking.
    When use_sg=True, triples are injected into the prompt if non-empty;
    if triples are empty, falls back to plain prompt automatically.
    """
    text_content = _build_prompt(caption, triples, use_sg)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_content},
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

def evaluate(model, processor, sg_parser, dataset, split, max_samples):
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    counts = {"qwen3_gen": {"text": 0, "image": 0, "group": 0},
              "qwen3_gen_sg": {"text": 0, "image": 0, "group": 0}}
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

        # ── Plain (no SG) ──
        p00 = score(model, processor, img0, cap0, t0, use_sg=False)
        p10 = score(model, processor, img0, cap1, t1, use_sg=False)
        p01 = score(model, processor, img1, cap0, t0, use_sg=False)
        p11 = score(model, processor, img1, cap1, t1, use_sg=False)
        tc, ic, gc = winoground_metrics(p00, p10, p01, p11)
        counts["qwen3_gen"]["text"] += int(tc)
        counts["qwen3_gen"]["image"] += int(ic)
        counts["qwen3_gen"]["group"] += int(gc)
        row["qwen3_gen"] = _make_row(p00, p10, p01, p11, tc, ic, gc)

        # ── SG-augmented prompt ──
        s00 = score(model, processor, img0, cap0, t0, use_sg=True)
        s10 = score(model, processor, img0, cap1, t1, use_sg=True)
        s01 = score(model, processor, img1, cap0, t0, use_sg=True)
        s11 = score(model, processor, img1, cap1, t1, use_sg=True)
        tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
        counts["qwen3_gen_sg"]["text"] += int(tc)
        counts["qwen3_gen_sg"]["image"] += int(ic)
        counts["qwen3_gen_sg"]["group"] += int(gc)
        row["qwen3_gen_sg"] = _make_row(s00, s10, s01, s11, tc, ic, gc)

        per_example.append(row)

        if (idx + 1) % 10 == 0:
            n = idx + 1
            for s in ["qwen3_gen", "qwen3_gen_sg"]:
                log.info(f"[{s}] n={n} | text={counts[s]['text']/n:.3f} | "
                         f"image={counts[s]['image']/n:.3f} | group={counts[s]['group']/n:.3f}")

    n = len(data)
    summary = {s: {k: v / n for k, v in c.items()} for s, c in counts.items()}
    summary["n_evaluated"] = n
    return summary, per_example


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
                "group": c["group"] / n if n else 0,
                "n": n,
            }
    return result


# ═════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════

def print_summary(summary, tag_analysis=None):
    n = summary.get("n_evaluated", "?")
    W, G = 28, 8

    print(f"\n{'═' * 70}")
    print(f"  Winoground — Qwen3-VL-8B-Thinking  (n={n})")
    print(f"  Scoring: P(yes) from next-token logits (generative decoder)")
    print(f"  SG: YUKINO-SG TextSceneGraphParser | empty SG → plain prompt")
    print(f"{'═' * 70}")
    print(f"  {'Strategy':<{W}}  {'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}")
    print(f"  {'-' * 56}")
    print(f"  {'Random chance':<{W}}  {'0.250':>{G}}  {'0.250':>{G}}  {'0.063':>{G}}")

    for s in ["qwen3_gen", "qwen3_gen_sg"]:
        if s not in summary: continue
        v = summary[s]
        label = "Qwen3-VL-Think" if s == "qwen3_gen" else "Qwen3-VL-Think + SG"
        print(f"  {label:<{W}}  {v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}")

    print(f"{'═' * 70}")

    if "qwen3_gen" in summary and "qwen3_gen_sg" in summary:
        base = summary["qwen3_gen"]
        aug  = summary["qwen3_gen_sg"]
        dt = aug["text"] - base["text"]
        di = aug["image"] - base["image"]
        dg = aug["group"] - base["group"]
        print(f"\n  SG delta:  text {dt:+.4f}  |  image {di:+.4f}  |  group {dg:+.4f}")

    if tag_analysis:
        for s in ["qwen3_gen", "qwen3_gen_sg"]:
            label = "Qwen3-VL-Think" if s == "qwen3_gen" else "Qwen3-VL-Think + SG"
            print(f"\n  ── {label} per tag ──")
            print(f"  {'Tag':<30}  {'n':>4}  {'Text':>8}  {'Image':>8}  {'Group':>8}")
            print(f"  {'-' * 62}")
            for tag, data in sorted(tag_analysis.items()):
                if s not in data: continue
                d = data[s]
                print(f"  {tag:<30}  {d['n']:>4}  {d['text']:>8.3f}  {d['image']:>8.3f}  {d['group']:>8.3f}")
    print()


# ═════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Winoground eval: Qwen3-VL-8B-Thinking (generative P(yes))")
    p.add_argument("--model_id",      default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--spacy_model",   default="en_core_web_sm")
    p.add_argument("--hf_token",      default=None)
    p.add_argument("--max_samples",   type=int, default=None)
    p.add_argument("--split",         default="test")
    p.add_argument("--output_dir",    default="./results_qwen3_gen")
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--tag_analysis",  action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    log.info(f"Qwen3-VL-Thinking standalone eval")
    log.info(f"Model: {args.model_id}")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    sg_parser = TextSceneGraphParser(args.spacy_model)
    model, processor = load_model(args.model_id, args.device)

    # Verify yes/no token resolution
    yes_ids = _resolve_yes_no_ids(processor.tokenizer, "yes")
    no_ids = _resolve_yes_no_ids(processor.tokenizer, "no")
    log.info(f"Token IDs — yes: {yes_ids.tolist()}, no: {no_ids.tolist()}")

    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    summary, per_example = evaluate(
        model, processor, sg_parser, dataset, args.split, args.max_samples,
    )

    strats = ["qwen3_gen", "qwen3_gen_sg"]
    tag_analysis = analyze_by_tag(per_example, strats) if args.tag_analysis else None
    print_summary(summary, tag_analysis)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": args.model_id,
        "scoring": "P(yes) from next-token logits (generative decoder)",
        "spacy_model": args.spacy_model,
        "split": args.split,
        "max_samples": args.max_samples,
        "sg_behavior": "triples injected into prompt when non-empty, plain prompt otherwise",
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