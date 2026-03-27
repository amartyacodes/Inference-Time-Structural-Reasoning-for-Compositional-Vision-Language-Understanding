"""
Unified LLaVA Evaluation on Winoground Dataset
===============================================
All scoring methods use the same interface and Winoground metrics,
so results are directly comparable.

Methods:
  1. yesno_plain     — Baseline yes/no, no scene graph
  2. nll_plain       — Baseline NLL,    no scene graph
  3. yesno_sg        — Yes/No with scene graph checklist injected into prompt
  4. nll_sg          — NLL    with scene graph appended to caption
  5. cot_sg          — Chain-of-Thought with scene graph reasoning scaffold (generative)

Every method returns a scalar score per (image, caption) pair.
Higher score = model thinks image matches caption.
The same winoground_metrics() function is applied to all methods.

Usage:
    python llava_winoground_unified.py \\
        --model_id llava-hf/llava-1.5-7b-hf \\
        --methods all \\
        --hf_token YOUR_TOKEN \\
        --max_samples 200 \\
        --output_dir ./results

    # Run only specific methods
    python llava_winoground_unified.py \\
        --methods yesno_plain nll_plain cot_sg

Requirements:
    pip install transformers torch pillow datasets spacy
    python -m spacy download en_core_web_trf
"""

import re
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlavaProcessor, LlavaForConditionalGeneration
import spacy

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_METHODS = ["yesno_plain", "nll_plain", "yesno_sg", "nll_sg", "cot_sg"]


# ═════════════════════════════════════════════════════════════
# SECTION 1: Scene Graph Extraction (shared by all SG methods)
# ═════════════════════════════════════════════════════════════

def load_spacy(model_name: str = "en_core_web_trf"):
    try:
        nlp = spacy.load(model_name)
        log.info(f"spaCy loaded: {model_name}")
    except OSError:
        log.warning(f"{model_name} not found, falling back to en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def _get_noun_chunk(token, doc) -> str:
    """Return full noun chunk for a token, else its lemma."""
    for chunk in doc.noun_chunks:
        if token in chunk:
            return chunk.text.lower()
    return token.lemma_


def extract_triplets(text: str, nlp) -> list[tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triplets via spaCy dependency parsing.
    Handles: direct objects, prepositional objects, attributes, passives.
    """
    doc = nlp(text)
    triplets = []

    for token in doc:
        if token.dep_ not in ("ROOT", "relcl", "acl", "prep", "appos"):
            continue

        subjs = [
            w for w in token.subtree
            if w.dep_ in ("nsubj", "nsubjpass") and w.head == token
        ]
        if not subjs:
            continue

        # Direct / attribute objects
        dobjs = [w for w in token.rights if w.dep_ in ("dobj", "attr", "oprd")]
        for s in subjs:
            for o in dobjs:
                triplets.append((
                    _get_noun_chunk(s, doc),
                    token.lemma_,
                    _get_noun_chunk(o, doc),
                ))

        # Prepositional objects
        for prep in token.rights:
            if prep.dep_ == "prep":
                for o in prep.rights:
                    if o.dep_ == "pobj":
                        triplets.append((
                            _get_noun_chunk(subjs[0], doc),
                            f"{token.lemma_} {prep.lemma_}",
                            _get_noun_chunk(o, doc),
                        ))

    # Deduplicate
    seen, unique = set(), []
    for t in triplets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def triplets_to_str(triplets: list[tuple]) -> str:
    if not triplets:
        return "none detected"
    return "; ".join([f"({s} → {r} → {o})" for s, r, o in triplets])


def triplets_to_checklist(triplets: list[tuple]) -> str:
    """
    Format triplets as a neutral scene graph reference.
    Works for action, spatial, possessive, and attribute relations.
    Does NOT force a rigid 'performing' template — just presents the structure.
    """
    if not triplets:
        return "  - (no relations extracted — check overall caption match)"
    lines = []
    for s, r, o in triplets:
        lines.append(f"  - {s}  [{r}]  {o}")
    return "\n".join(lines)


def build_cot_scaffold(caption: str, triplets: list[tuple]) -> str:
    """
    Build a CoT reasoning scaffold using the scene graph as structured context.
    Presents the graph neutrally — works for action, spatial, and attribute relations.
    LLaVA reasons freely using the graph as a guide rather than following rigid steps.
    """
    lines = [
        f"Caption: '{caption}'",
        f"",
        f"Scene graph relations extracted from the caption:",
    ]

    if triplets:
        for s, r, o in triplets:
            lines.append(f"  - {s}  [{r}]  {o}")
    else:
        lines.append("  - (no relations extracted)")

    lines += [
        f"",
        f"Using the scene graph above as a structured guide, reason about "
        f"whether the image matches the caption. Consider:",
        f"  - Are all the entities present in the image?",
        f"  - Are the relations between entities correct?",
        f"  - Are spatial positions correct if mentioned?",
        f"  - Is the direction of any action correct?",
        f"",
        f"Then answer yes or no.",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# SECTION 2: Shared LLaVA utilities
# ═════════════════════════════════════════════════════════════

def _get_yes_no_prob(model, processor, image: Image.Image, prompt: str) -> float:
    """
    Core yes/no scoring function.
    Runs forward pass, reads P(yes) vs P(no) at the last token position.
    Returns float in [0, 1].
    Used by: yesno_plain, yesno_sg
    """
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # (1, seq_len, vocab)

    last = logits[0, -1]
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = processor.tokenizer.encode("no",  add_special_tokens=False)[0]
    prob_yes = torch.softmax(torch.stack([last[yes_id], last[no_id]]), dim=0)[0]
    return prob_yes.item()


def _get_caption_nll(
    model, processor, image: Image.Image, prefix: str, caption: str
) -> float:
    """
    Core NLL scoring function.
    Computes mean per-token log-likelihood of caption tokens given image + prefix.
    Returns float (higher = more likely).
    Used by: nll_plain, nll_sg
    """
    full_prompt = prefix + caption

    inputs_full   = processor(text=full_prompt, images=image, return_tensors="pt")
    inputs_prefix = processor(text=prefix,      images=image, return_tensors="pt")

    input_ids  = inputs_full["input_ids"].to(model.device)
    prefix_len = inputs_prefix["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:, :prefix_len] = -100  # mask prefix, score only caption tokens

    with torch.no_grad():
        loss = model(
            input_ids=input_ids,
            pixel_values=inputs_full["pixel_values"].to(model.device),
            labels=labels,
        ).loss

    return -loss.item()  # negate NLL → log-likelihood (higher = better)


def _parse_yes_no_from_generated(response: str) -> float:
    """
    Extract yes/no score from a generated CoT response.
    Strategy: last word → last sentence → majority vote → 0.5 fallback.
    Returns float in {0.0, 0.5, 1.0}
    Used by: cot_sg
    """
    text = response.lower().strip()

    last_word = re.sub(r"[^a-z]", "", text.split()[-1]) if text.split() else ""
    if last_word == "yes":
        return 1.0
    if last_word == "no":
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if sentences:
        last = sentences[-1]
        if re.search(r"\byes\b", last):
            return 1.0
        if re.search(r"\bno\b", last):
            return 0.0

    yes_count = len(re.findall(r"\byes\b", text))
    no_count  = len(re.findall(r"\bno\b",  text))
    if yes_count > no_count:
        return 1.0
    if no_count > yes_count:
        return 0.0

    return 0.5  # uncertain


# ═════════════════════════════════════════════════════════════
# SECTION 3: Scoring Methods
# All return a single float: higher = better image-caption match
# ═════════════════════════════════════════════════════════════

def score_yesno_plain(model, processor, nlp, image: Image.Image, caption: str) -> float:
    """
    METHOD 1: Baseline yes/no — no scene graph.
    Prompt: 'Does this image show: <caption>? Answer yes or no.'
    Score:  P(yes) at next-token position.
    """
    prompt = (
        f"USER: <image>\n"
        f"Does this image show: '{caption}'? Answer yes or no.\n"
        f"ASSISTANT:"
    )
    return _get_yes_no_prob(model, processor, image, prompt)


def score_nll_plain(model, processor, nlp, image: Image.Image, caption: str) -> float:
    """
    METHOD 2: Baseline NLL — no scene graph.
    Prefix:  'Describe the image in one sentence. ASSISTANT: '
    Score:   mean per-token log-likelihood of caption tokens.
    """
    prefix = (
        "USER: <image>\n"
        "Describe the image in one sentence.\n"
        "ASSISTANT: "
    )
    return _get_caption_nll(model, processor, image, prefix, caption)


def score_yesno_sg(model, processor, nlp, image: Image.Image, caption: str) -> float:
    """
    METHOD 3: Yes/No with scene graph context injected into prompt.
    Presents scene graph neutrally — works for action, spatial, and attribute relations.
    LLaVA uses the graph as a reference guide before giving yes/no answer.
    Score:  P(yes) at next-token position.
    """
    triplets  = extract_triplets(caption, nlp)
    checklist = triplets_to_checklist(triplets)

    prompt = (
        f"USER: <image>\n"
        f"Caption: '{caption}'\n\n"
        f"The caption has the following scene graph relations:\n"
        f"{checklist}\n\n"
        f"Using the scene graph as a guide, verify whether the image matches "
        f"the caption. Pay close attention to which entity is doing what, "
        f"and spatial positions if mentioned.\n"
        f"Answer yes or no.\n"
        f"ASSISTANT:"
    )
    return _get_yes_no_prob(model, processor, image, prompt)


def score_nll_sg(model, processor, nlp, image: Image.Image, caption: str) -> float:
    """
    METHOD 4: NLL with scene graph appended to caption.
    The model scores the caption + its relational structure as a unit.
    Score:   mean per-token log-likelihood of (caption + graph tag) tokens.
    """
    triplets = extract_triplets(caption, nlp)

    if triplets:
        graph_str = "; ".join([f"{s} {r} {o}" for s, r, o in triplets])
        augmented = f"{caption} [relations: {graph_str}]"
    else:
        augmented = caption

    prefix = (
        "USER: <image>\n"
        "Describe the image and its object relations.\n"
        "ASSISTANT: "
    )
    return _get_caption_nll(model, processor, image, prefix, augmented)


def score_cot_sg(
    model, processor, nlp, image: Image.Image, caption: str,
    max_new_tokens: int = 300,
) -> float:
    """
    METHOD 5: Chain-of-Thought with scene graph reasoning scaffold.
    Builds a step-by-step plan: entity existence → directional relation check.
    LLaVA generates full reasoning, final yes/no is extracted.
    Score: 1.0 (yes) / 0.0 (no) / 0.5 (uncertain)
    """
    triplets = extract_triplets(caption, nlp)
    scaffold = build_cot_scaffold(caption, triplets)

    prompt = (
        f"USER: <image>\n"
        f"{scaffold}\n"
        f"ASSISTANT:"
    )

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    full_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response  = full_text.split("ASSISTANT:")[-1].strip()
    return _parse_yes_no_from_generated(response)


# Map method names to functions
SCORE_FN_MAP = {
    "yesno_plain": score_yesno_plain,
    "nll_plain":   score_nll_plain,
    "yesno_sg":    score_yesno_sg,
    "nll_sg":      score_nll_sg,
    "cot_sg":      score_cot_sg,
}


# ═════════════════════════════════════════════════════════════
# SECTION 4: Winoground Metrics (shared, applied to all methods)
# ═════════════════════════════════════════════════════════════

def winoground_metrics(
    s_c0_i0: float,
    s_c1_i0: float,
    s_c0_i1: float,
    s_c1_i1: float,
) -> tuple[bool, bool, bool]:
    """
    Standard Winoground three-way accuracy.

    Notation: s_cX_iY = score(caption X, image Y)
      caption 0 is correct for image 0
      caption 1 is correct for image 1

    Text  : correct caption scores higher for BOTH images
    Image : correct image  scores higher for BOTH captions
    Group : text AND image both correct (strictest)
    """
    text  = (s_c0_i0 > s_c1_i0) and (s_c1_i1 > s_c0_i1)
    image = (s_c0_i0 > s_c0_i1) and (s_c1_i1 > s_c1_i0)
    group = text and image
    return text, image, group


# ═════════════════════════════════════════════════════════════
# SECTION 5: Evaluation Loop
# ═════════════════════════════════════════════════════════════

def evaluate(
    model,
    processor,
    nlp,
    dataset,
    methods: list[str],
    split: str = "test",
    max_samples: int | None = None,
    max_new_tokens: int = 300,
):
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    score_fns = {m: SCORE_FN_MAP[m] for m in methods}
    counts     = {m: {"text": 0, "image": 0, "group": 0} for m in methods}
    per_example = []

    for idx, example in enumerate(tqdm(data, desc="Evaluating")):
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        tag  = example.get("tag", "")

        row = {
            "idx": idx,
            "caption_0": cap0,
            "caption_1": cap1,
            "tag": tag,
        }

        for method, fn in score_fns.items():
            # kwargs for cot_sg need max_new_tokens
            kwargs = {"max_new_tokens": max_new_tokens} if method == "cot_sg" else {}

            s_c0_i0 = fn(model, processor, nlp, img0, cap0, **kwargs)
            s_c1_i0 = fn(model, processor, nlp, img0, cap1, **kwargs)
            s_c0_i1 = fn(model, processor, nlp, img1, cap0, **kwargs)
            s_c1_i1 = fn(model, processor, nlp, img1, cap1, **kwargs)

            text_c, image_c, group_c = winoground_metrics(
                s_c0_i0, s_c1_i0, s_c0_i1, s_c1_i1
            )

            counts[method]["text"]  += int(text_c)
            counts[method]["image"] += int(image_c)
            counts[method]["group"] += int(group_c)

            row[method] = {
                "scores": {
                    "c0_i0": round(s_c0_i0, 5),
                    "c1_i0": round(s_c1_i0, 5),
                    "c0_i1": round(s_c0_i1, 5),
                    "c1_i1": round(s_c1_i1, 5),
                },
                "correct": {
                    "text":  text_c,
                    "image": image_c,
                    "group": group_c,
                },
            }

        per_example.append(row)

        # Live logging every 10 examples
        if (idx + 1) % 10 == 0:
            n = idx + 1
            for m in methods:
                log.info(
                    f"[{m}] n={n} | "
                    f"text={counts[m]['text']/n:.3f} | "
                    f"image={counts[m]['image']/n:.3f} | "
                    f"group={counts[m]['group']/n:.3f}"
                )

    n = len(data)
    summary = {
        m: {metric: v / n for metric, v in c.items()}
        for m, c in counts.items()
    }
    summary["n_evaluated"] = n

    return summary, per_example


# ═════════════════════════════════════════════════════════════
# SECTION 6: Tag-level Subset Analysis
# ═════════════════════════════════════════════════════════════

def analyze_by_tag(per_example: list[dict], methods: list[str]) -> dict:
    """Break down accuracy by Winoground linguistic tag per method."""
    tag_data = {}

    for ex in per_example:
        tag = ex.get("tag") or "untagged"
        if tag not in tag_data:
            tag_data[tag] = {m: {"text": 0, "image": 0, "group": 0, "n": 0}
                             for m in methods}
        for m in methods:
            if m not in ex:
                continue
            tag_data[tag][m]["text"]  += int(ex[m]["correct"]["text"])
            tag_data[tag][m]["image"] += int(ex[m]["correct"]["image"])
            tag_data[tag][m]["group"] += int(ex[m]["correct"]["group"])
            tag_data[tag][m]["n"]     += 1

    # Normalize
    result = {}
    for tag, data in tag_data.items():
        result[tag] = {}
        for m, c in data.items():
            n = c["n"]
            result[tag][m] = {
                "text":  c["text"]  / n if n else 0,
                "image": c["image"] / n if n else 0,
                "group": c["group"] / n if n else 0,
                "n": n,
            }
    return result


# ═════════════════════════════════════════════════════════════
# SECTION 7: Reporting
# ═════════════════════════════════════════════════════════════

METHOD_LABELS = {
    "yesno_plain": "YesNo (plain)",
    "nll_plain":   "NLL   (plain)",
    "yesno_sg":    "YesNo + SG",
    "nll_sg":      "NLL   + SG",
    "cot_sg":      "CoT   + SG",
}


def print_summary(summary: dict, tag_analysis: dict | None = None):
    n       = summary.get("n_evaluated", "?")
    methods = [k for k in summary if k != "n_evaluated"]

    W = 20
    print(f"\n{'═' * 62}")
    print(f"  Winoground Results  (n={n})")
    print(f"{'═' * 62}")
    print(f"  {'Method':<{W}}  {'Text':>8}  {'Image':>8}  {'Group':>8}")
    print(f"  {'-' * 56}")
    print(f"  {'Random chance':<{W}}  {'0.2500':>8}  {'0.2500':>8}  {'0.0625':>8}")

    for m in methods:
        label = METHOD_LABELS.get(m, m)
        s = summary[m]
        print(
            f"  {label:<{W}}  "
            f"{s['text']:>8.4f}  "
            f"{s['image']:>8.4f}  "
            f"{s['group']:>8.4f}"
        )
    print(f"{'═' * 62}")

    if tag_analysis:
        print(f"\n  Per-tag breakdown:")
        for m in methods:
            label = METHOD_LABELS.get(m, m)
            print(f"\n  ── {label} ──")
            print(f"  {'Tag':<30}  {'n':>4}  {'Text':>7}  {'Image':>7}  {'Group':>7}")
            print(f"  {'-' * 56}")
            for tag, data in sorted(tag_analysis.items()):
                if m not in data:
                    continue
                d = data[m]
                print(
                    f"  {tag:<30}  {d['n']:>4}  "
                    f"{d['text']:>7.3f}  {d['image']:>7.3f}  {d['group']:>7.3f}"
                )
    print()


# ═════════════════════════════════════════════════════════════
# SECTION 8: Model Loading
# ═════════════════════════════════════════════════════════════

def load_llava(model_id: str, device: str = "auto"):
    log.info(f"Loading processor: {model_id}")
    processor = LlavaProcessor.from_pretrained(model_id)

    log.info(f"Loading model: {model_id}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("Model ready.")
    return model, processor


# ═════════════════════════════════════════════════════════════
# SECTION 9: CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LLaVA × Winoground evaluation — all methods comparable"
    )
    parser.add_argument(
        "--model_id", type=str,
        default="llava-hf/llava-1.5-7b-hf",
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=ALL_METHODS + ["all"],
        default=["all"],
        help="Methods to run. Use 'all' or list e.g. yesno_plain cot_sg",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token (required for Winoground)",
    )
    parser.add_argument(
        "--spacy_model", type=str, default="en_core_web_trf",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to N examples (None = full dataset ~400)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=300,
        help="Max tokens for CoT generation (cot_sg only)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve methods
    methods = ALL_METHODS if "all" in args.methods else args.methods
    log.info(f"Running methods: {methods}")

    # Auth
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        log.info("HuggingFace login successful.")

    # Load components
    nlp              = load_spacy(args.spacy_model)
    model, processor = load_llava(args.model_id, device=args.device)

    # Dataset
    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    # Evaluate
    summary, per_example = evaluate(
        model=model,
        processor=processor,
        nlp=nlp,
        dataset=dataset,
        methods=methods,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    # Tag analysis
    tag_analysis = analyze_by_tag(per_example, methods)

    # Print
    print_summary(summary, tag_analysis)

    # Save
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug     = args.model_id.replace("/", "_")

    paths = {
        "summary":     out_dir / f"{slug}_summary.json",
        "per_example": out_dir / f"{slug}_per_example.json",
        "tags":        out_dir / f"{slug}_tags.json",
    }

    with open(paths["summary"],     "w") as f: json.dump(summary,      f, indent=2)
    with open(paths["per_example"], "w") as f: json.dump(per_example,  f, indent=2)
    with open(paths["tags"],        "w") as f: json.dump(tag_analysis, f, indent=2)

    for name, path in paths.items():
        log.info(f"{name:<12} → {path}")


if __name__ == "__main__":
    main()