"""
qwen3_interpret.py
==================
Interpretability probe for Qwen3-VL-8B-Thinking on Winoground.

For each of N random examples, for each of the 4 (image, caption) slots,
runs a single focused conversation:

  Turn 1 — Yes/No decision
            "Does this image match the caption?"

  Turn 2 — Reasoning trace
            "Explain step by step what you see, which scene graph relations
             you used, and why you answered yes or no."

Scene graph triples (YUKINO-SG) are injected into the Turn 1 prompt.
The model's reasoning in Turn 2 is stored verbatim in the output JSON.

Output: interpretability_results.json
  {
    "summary": { "n": 20, "text": 0.4, "image": 0.35, "group": 0.2 },
    "examples": [
      {
        "idx": 17,
        "caption_0": "...", "caption_1": "...", "tag": "...",
        "slots": {
          "c0_i0": {
            "caption":    "...",
            "sg_triples": ["(person, kiss, dog)", ...],
            "p_yes":      0.823,
            "answer":     "yes",
            "reasoning":  "1. In the image I see ..."
          },
          ...
        },
        "correct": { "text": true, "image": false, "group": false }
      }
    ]
  }

Usage:
    python qwen3_interpret.py
    python qwen3_interpret.py --n_samples 20 --seed 42
    python qwen3_interpret.py --n_samples 5  --output_dir ./debug
"""

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path

import spacy
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    raise ImportError("pip install transformers>=4.57.0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Scene Graph ───────────────────────────────────────────────

@dataclass
class Triple:
    subject: str
    relation: str
    obj: str
    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.obj})"


class SGParser:
    def __init__(self, model="en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            self.nlp = spacy.load("en_core_web_sm")
        self._cache = {}

    def parse(self, caption: str) -> list[Triple]:
        if caption in self._cache:
            return self._cache[caption]
        doc = self.nlp(caption.lower().strip())
        triples = self._svo(doc) + self._prep(doc) + self._copular(doc)
        seen, out = set(), []
        for t in triples:
            k = (t.subject, t.relation, t.obj)
            if k not in seen and t.subject and t.obj:
                seen.add(k); out.append(t)
        self._cache[caption] = out
        return out

    def _np(self, token):
        noun = token
        if token.pos_ not in ("NOUN", "PROPN"):
            for t in token.subtree:
                if t.pos_ in ("NOUN", "PROPN"):
                    noun = t; break
        adjs = [c.text for c in noun.children if c.dep_ == "amod"]
        comps = [c.text for c in noun.children if c.dep_ == "compound"]
        parts = adjs + comps
        return (" ".join(parts) + " " + noun.lemma_).strip() if parts else noun.lemma_

    def _svo(self, doc):
        out = []
        for t in doc:
            if t.dep_ not in {"ROOT", "relcl", "acl", "advcl", "xcomp", "ccomp", "conj"}: continue
            if t.lemma_ == "be": continue
            subjs = [w for w in t.children if w.dep_ in ("nsubj", "nsubjpass")]
            objs  = [w for w in t.children if w.dep_ in ("dobj", "attr", "oprd")]
            neg   = any(w.dep_ == "neg" for w in t.children)
            verb  = ("not " + t.lemma_) if neg else t.lemma_
            for s in subjs:
                for o in objs:
                    out.append(Triple(self._np(s), verb, self._np(o)))
        return out

    def _prep(self, doc):
        out = []
        for t in doc:
            if t.dep_ == "prep" and t.head.pos_ in ("NOUN", "PROPN", "VERB", "AUX"):
                for pobj in t.children:
                    if pobj.dep_ == "pobj":
                        out.append(Triple(self._np(t.head), t.text, self._np(pobj)))
        return out

    def _copular(self, doc):
        out = []
        for t in doc:
            if t.lemma_ != "be" or t.dep_ != "ROOT": continue
            subjs = [w for w in t.children if w.dep_ in ("nsubj", "nsubjpass")]
            attrs = [w for w in t.children if w.dep_ in ("attr", "acomp")]
            for s in subjs:
                for a in attrs:
                    obj = self._np(a) if a.pos_ in ("NOUN", "PROPN") else a.lemma_
                    out.append(Triple(self._np(s), "is", obj))
        return out


# ── Model helpers ─────────────────────────────────────────────

def load_model(model_id, device):
    log.info(f"Loading {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    ).eval()
    return model, processor


def _enc(processor, messages, device):
    enc = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    enc.pop("token_type_ids", None)
    return {k: v.to(device) for k, v in enc.items()}


def _decode(model, processor, messages, device, max_new_tokens) -> str:
    inputs = _enc(processor, messages, device)
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None,
    )
    new = out[0, inputs["input_ids"].shape[-1]:]
    return processor.tokenizer.decode(new, skip_special_tokens=True).strip()


@torch.no_grad()
def _p_yes(model, processor, messages, device) -> tuple[float, str]:
    inputs = _enc(processor, messages, device)
    logits = model(**inputs).logits[0, -1].float()

    def ids(word):
        s = set()
        for v in [word, word.capitalize(), word.upper()]:
            t = processor.tokenizer.encode(v, add_special_tokens=False)
            if t: s.add(t[0])
        return torch.tensor(list(s), dtype=torch.long, device=logits.device)

    y = torch.logsumexp(logits[ids("yes")], dim=0)
    n = torch.logsumexp(logits[ids("no")],  dim=0)
    p = torch.softmax(torch.stack([y, n]), dim=0)[0].item()
    return p, ("yes" if p >= 0.5 else "no")


# ── Core: interpret one (image, caption) slot ─────────────────

def interpret_slot(model, processor, image, caption, triples, device) -> dict:
    sg_lines = "\n".join(f"  - {t}" for t in triples)
    sg_block = (
        f"\nScene graph relations:\n{sg_lines}\n\n"
        "Use these relations to focus on who is doing what and "
        "any spatial or attribute details.\n\n"
    ) if triples else ""

    turn1_prompt = (
        f"{sg_block}"
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer with only 'yes' or 'no'."
    )

    msgs_t1 = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": turn1_prompt},
        ],
    }]
    p_yes, answer = _p_yes(model, processor, msgs_t1, device)

    turn2_prompt = (
        f"You answered '{answer}'. Explain your reasoning step by step:\n\n"
        f"1. What do you see in the image? Describe entities, actions, "
        f"   and spatial relationships.\n"
        f"2. What does the caption claim?\n"
        f"3. Which scene graph relation (if any) was most important for "
        f"   your decision, and why?\n"
        f"4. Where exactly does the image match or contradict the caption?"
    )

    msgs_t2 = [
        *msgs_t1,
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        {"role": "user",      "content": [{"type": "text", "text": turn2_prompt}]},
    ]
    reasoning = _decode(model, processor, msgs_t2, device, max_new_tokens=512)

    return {
        "caption":   caption,
        "sg_triples": [repr(t) for t in triples],
        "p_yes":     round(p_yes, 5),
        "answer":    answer,
        "reasoning": reasoning,
    }


# ── Evaluation loop ───────────────────────────────────────────

def run(model, processor, sg_parser, dataset, split, n_samples, seed, out_dir, device):
    data = dataset[split]
    indices = random.Random(seed).sample(range(len(data)), min(n_samples, len(data)))
    log.info(f"Evaluating {len(indices)} samples (seed={seed})")

    results = []
    counts  = {"text": 0, "image": 0, "group": 0}

    for idx in tqdm(indices, desc="interpret"):
        ex   = data[idx]
        img0 = ex["image_0"].convert("RGB")
        img1 = ex["image_1"].convert("RGB")
        cap0, cap1 = ex["caption_0"], ex["caption_1"]
        tag  = ex.get("tag", "")
        t0, t1 = sg_parser.parse(cap0), sg_parser.parse(cap1)

        slots = {
            "c0_i0": interpret_slot(model, processor, img0, cap0, t0, device),
            "c1_i0": interpret_slot(model, processor, img0, cap1, t1, device),
            "c0_i1": interpret_slot(model, processor, img1, cap0, t0, device),
            "c1_i1": interpret_slot(model, processor, img1, cap1, t1, device),
        }

        s00, s10 = slots["c0_i0"]["p_yes"], slots["c1_i0"]["p_yes"]
        s01, s11 = slots["c0_i1"]["p_yes"], slots["c1_i1"]["p_yes"]
        tc = (s00 > s10) and (s11 > s01)
        ic = (s00 > s01) and (s11 > s10)
        gc = tc and ic
        counts["text"] += int(tc); counts["image"] += int(ic); counts["group"] += int(gc)

        results.append({
            "idx": idx, "caption_0": cap0, "caption_1": cap1, "tag": tag,
            "sg_cap0": [repr(t) for t in t0],
            "sg_cap1": [repr(t) for t in t1],
            "slots":   slots,
            "correct": {"text": tc, "image": ic, "group": gc},
        })

    n = len(results)
    summary = {"n": n, "seed": seed,
               "text":  counts["text"]  / n,
               "image": counts["image"] / n,
               "group": counts["group"] / n}

    print(f"\n{'═'*52}")
    print(f"  Qwen3-VL Interpretability  (n={n}, seed={seed})")
    print(f"{'═'*52}")
    for m in ("text", "image", "group"):
        print(f"  {m:<8} {summary[m]:.4f}")
    print(f"{'═'*52}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "interpretability_results.json"
    with open(path, "w") as f:
        json.dump({"summary": summary, "examples": results}, f, indent=2, default=str)
    log.info(f"Saved → {path}")


# ── CLI ───────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",    default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--hf_token",    default=None)
    p.add_argument("--split",       default="test")
    p.add_argument("--n_samples",   type=int, default=20)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_dir",  default="./interpretability_results")
    p.add_argument("--device",      default="cuda:0")
    args = p.parse_args()

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    model, processor = load_model(args.model_id, args.device)
    sg_parser = SGParser(args.spacy_model)

    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)

    run(model, processor, sg_parser, dataset,
        args.split, args.n_samples, args.seed,
        Path(args.output_dir), args.device)


if __name__ == "__main__":
    main()