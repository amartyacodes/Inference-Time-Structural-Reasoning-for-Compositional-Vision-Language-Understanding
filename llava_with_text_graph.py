"""
LLaVA + Scene Graph Chain-of-Thought Evaluation on Winoground
=============================================================
Pipeline:
  1. Parse caption into (subject, relation, object) triplets using spaCy
  2. Build a CoT reasoning scaffold from the scene graph
  3. Inject scaffold into LLaVA prompt
  4. LLaVA generates step-by-step reasoning, then gives final yes/no
  5. Score is extracted from the generated response
  6. Winoground text / image / group accuracy is computed

Additionally includes:
  - Baseline yes/no scoring (no scene graph) for comparison
  - Per-example logging with raw scores and CoT responses
  - Subset breakdown by Winoground tag (relational, swapped, etc.)

Usage:
    python llava_sg_cot_winoground.py \\
        --model_id llava-hf/llava-1.5-7b-hf \\
        --hf_token YOUR_HF_TOKEN \\
        --max_samples 200 \\
        --output_dir ./results

Requirements:
    pip install transformers torch pillow datasets spacy sentence-transformers
    python -m spacy download en_core_web_trf
"""

import os
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


# ─────────────────────────────────────────────────────────────
# spaCy scene graph extraction
# ─────────────────────────────────────────────────────────────
def load_spacy(model_name: str = "en_core_web_trf"):
    """Load spaCy model. Falls back to en_core_web_sm if trf not available."""
    try:
        nlp = spacy.load(model_name)
        log.info(f"spaCy model loaded: {model_name}")
    except OSError:
        log.warning(f"{model_name} not found, falling back to en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def extract_triplets(text: str, nlp) -> list[tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triplets from text using spaCy
    dependency parsing.

    Handles:
      - Direct objects:     nsubj → ROOT → dobj
      - Prepositional objs: nsubj → ROOT → prep → pobj
      - Passive subjects:   nsubjpass → ROOT → agent → pobj
      - Attribute:          nsubj → ROOT → attr
    """
    doc = nlp(text)
    triplets = []

    for token in doc:
        if token.pos_ not in ("VERB", "ADP", "NOUN"):
            continue
        if token.dep_ not in ("ROOT", "relcl", "acl", "prep", "appos"):
            continue

        subjs = [
            w for w in token.subtree
            if w.dep_ in ("nsubj", "nsubjpass") and w.head == token
        ]
        if not subjs:
            continue

        # Direct objects
        dobjs = [
            w for w in token.rights
            if w.dep_ in ("dobj", "attr", "oprd")
        ]
        for s in subjs:
            for o in dobjs:
                triplets.append((
                    _get_noun_chunk(s, doc),
                    token.lemma_,
                    _get_noun_chunk(o, doc)
                ))

        # Prepositional objects
        for prep in token.rights:
            if prep.dep_ == "prep":
                pobjs = [w for w in prep.rights if w.dep_ == "pobj"]
                for s in subjs:
                    for o in pobjs:
                        rel = f"{token.lemma_} {prep.lemma_}"
                        triplets.append((
                            _get_noun_chunk(s, doc),
                            rel,
                            _get_noun_chunk(o, doc)
                        ))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in triplets:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _get_noun_chunk(token, doc) -> str:
    """Return the full noun chunk for a token if available, else lemma."""
    for chunk in doc.noun_chunks:
        if token in chunk:
            return chunk.text.lower()
    return token.lemma_


def triplets_to_str(triplets: list[tuple]) -> str:
    """Human-readable triplet string for prompts."""
    if not triplets:
        return "none detected"
    return "; ".join([f"({s} → {r} → {o})" for s, r, o in triplets])


# ─────────────────────────────────────────────────────────────
# LLaVA model loading
# ─────────────────────────────────────────────────────────────
def load_llava(model_id: str, device: str = "auto"):
    log.info(f"Loading LLaVA processor: {model_id}")
    processor = LlavaProcessor.from_pretrained(model_id)

    log.info(f"Loading LLaVA model: {model_id}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("LLaVA model ready.")
    return model, processor


# ─────────────────────────────────────────────────────────────
# Scoring Method 1: Baseline Yes/No (no scene graph)
# ─────────────────────────────────────────────────────────────
def score_baseline_yesno(
    model,
    processor,
    image: Image.Image,
    caption: str,
) -> float:
    """
    Baseline: plain yes/no prompt with no scene graph.
    Returns P(yes) in [0, 1].
    """
    prompt = (
        f"USER: <image>\n"
        f"Does this image match the caption: '{caption}'?\n"
        f"Answer yes or no.\n"
        f"ASSISTANT:"
    )

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    last = logits[0, -1]
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = processor.tokenizer.encode("no",  add_special_tokens=False)[0]
    prob_yes = torch.softmax(torch.stack([last[yes_id], last[no_id]]), dim=0)[0]
    return prob_yes.item()


# ─────────────────────────────────────────────────────────────
# Scoring Method 2: Scene Graph Chain-of-Thought
# ─────────────────────────────────────────────────────────────
def build_cot_prompt(caption: str, triplets: list[tuple]) -> str:
    """
    Build a CoT reasoning scaffold from scene graph triplets.

    Example output:
      Caption: 'a dog biting a man'
      Scene graph relations:
        (dog → bite → man)

      Reasoning plan:
        Step 1: Is there a 'dog' in the image?
        Step 2: Is there a 'man' in the image?
        Step 3: Is the 'dog' performing the action 'bite' on the 'man'?
               (Not the reverse: man biting dog)
        Final:  Does the full caption match the image?

      Work through each step, then answer yes or no.
    """
    lines = [
        f"Caption: '{caption}'",
        f"Scene graph relations extracted from the caption:",
        f"  {triplets_to_str(triplets)}",
        "",
        "Reasoning plan:",
    ]

    step = 1
    seen_entities = set()

    for s, r, o in triplets:
        # Entity existence checks
        if s not in seen_entities:
            lines.append(f"  Step {step}: Is there a '{s}' visible in the image?")
            step += 1
            seen_entities.add(s)
        if o not in seen_entities:
            lines.append(f"  Step {step}: Is there a '{o}' visible in the image?")
            step += 1
            seen_entities.add(o)

        # Relation directional check — critical for Winoground
        lines.append(
            f"  Step {step}: Is '{s}' performing '{r}' on/to '{o}'? "
            f"(NOT the reverse: '{o}' {r} '{s}')"
        )
        step += 1

    lines.append(
        f"  Final: Considering all steps, does the caption '{caption}' correctly "
        f"describe the image? Answer yes or no."
    )

    return "\n".join(lines)


def score_sg_cot(
    model,
    processor,
    nlp,
    image: Image.Image,
    caption: str,
    max_new_tokens: int = 300,
) -> tuple[float, str, list[tuple]]:
    """
    Scene Graph Chain-of-Thought scoring.

    Returns:
        score      : float in [0, 1]  — 1.0 = model says yes
        response   : str              — full CoT generated text
        triplets   : list of (s,r,o)  — extracted scene graph
    """
    triplets = extract_triplets(caption, nlp)
    cot_body  = build_cot_prompt(caption, triplets)

    prompt = (
        f"USER: <image>\n"
        f"{cot_body}\n"
        f"ASSISTANT:"
    )

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )

    full_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response  = full_text.split("ASSISTANT:")[-1].strip()

    score = _extract_yes_no_score(response)
    return score, response, triplets


def _extract_yes_no_score(response: str) -> float:
    """
    Parse yes/no from generated CoT response.
    Checks (in order):
      1. Last word of response
      2. Last sentence
      3. Overall yes/no count
      4. Default to 0.5 (uncertain)
    """
    text = response.lower().strip()

    # 1. Last word
    last_word = re.sub(r"[^a-z]", "", text.split()[-1]) if text.split() else ""
    if last_word == "yes":
        return 1.0
    if last_word == "no":
        return 0.0

    # 2. Last sentence
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if sentences:
        last = sentences[-1]
        if re.search(r"\byes\b", last):
            return 1.0
        if re.search(r"\bno\b", last):
            return 0.0

    # 3. Count yes/no in full response
    yes_count = len(re.findall(r"\byes\b", text))
    no_count  = len(re.findall(r"\bno\b",  text))
    if yes_count > no_count:
        return 1.0
    if no_count > yes_count:
        return 0.0

    # 4. Uncertain
    return 0.5


# ─────────────────────────────────────────────────────────────
# Winoground scoring criteria
# ─────────────────────────────────────────────────────────────
def winoground_metrics(s_c0_i0, s_c1_i0, s_c0_i1, s_c1_i1):
    """
    Standard Winoground three-way accuracy.

    Text  : correct caption scores higher for BOTH images
    Image : correct image  scores higher for BOTH captions
    Group : text AND image both correct
    """
    text  = (s_c0_i0 > s_c1_i0) and (s_c1_i1 > s_c0_i1)
    image = (s_c0_i0 > s_c0_i1) and (s_c1_i1 > s_c1_i0)
    group = text and image
    return text, image, group


# ─────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────
def evaluate(
    model,
    processor,
    nlp,
    dataset,
    split: str = "test",
    max_samples: int | None = None,
    max_new_tokens: int = 300,
    run_baseline: bool = True,
):
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    # Accumulators
    methods = ["sg_cot"]
    if run_baseline:
        methods = ["baseline", "sg_cot"]

    counts = {m: {"text": 0, "image": 0, "group": 0} for m in methods}
    per_example = []

    for idx, example in enumerate(tqdm(data, desc="Evaluating")):
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        tag  = example.get("tag", "")

        row = {
            "idx":       idx,
            "caption_0": cap0,
            "caption_1": cap1,
            "tag":       tag,
        }

        # ── Baseline ──────────────────────────────────────────
        if run_baseline:
            b_c0_i0 = score_baseline_yesno(model, processor, img0, cap0)
            b_c1_i0 = score_baseline_yesno(model, processor, img0, cap1)
            b_c0_i1 = score_baseline_yesno(model, processor, img1, cap0)
            b_c1_i1 = score_baseline_yesno(model, processor, img1, cap1)

            bt, bi, bg = winoground_metrics(b_c0_i0, b_c1_i0, b_c0_i1, b_c1_i1)
            counts["baseline"]["text"]  += int(bt)
            counts["baseline"]["image"] += int(bi)
            counts["baseline"]["group"] += int(bg)

            row["baseline"] = {
                "scores":  {"c0_i0": b_c0_i0, "c1_i0": b_c1_i0,
                             "c0_i1": b_c0_i1, "c1_i1": b_c1_i1},
                "correct": {"text": bt, "image": bi, "group": bg},
            }

        # ── Scene Graph CoT ───────────────────────────────────
        sg_c0_i0, resp_c0_i0, triplets_c0 = score_sg_cot(
            model, processor, nlp, img0, cap0, max_new_tokens)
        sg_c1_i0, resp_c1_i0, _           = score_sg_cot(
            model, processor, nlp, img0, cap1, max_new_tokens)
        sg_c0_i1, resp_c0_i1, _           = score_sg_cot(
            model, processor, nlp, img1, cap0, max_new_tokens)
        sg_c1_i1, resp_c1_i1, triplets_c1 = score_sg_cot(
            model, processor, nlp, img1, cap1, max_new_tokens)

        st, si, sg = winoground_metrics(sg_c0_i0, sg_c1_i0, sg_c0_i1, sg_c1_i1)
        counts["sg_cot"]["text"]  += int(st)
        counts["sg_cot"]["image"] += int(si)
        counts["sg_cot"]["group"] += int(sg)

        row["sg_cot"] = {
            "scores":   {"c0_i0": sg_c0_i0, "c1_i0": sg_c1_i0,
                          "c0_i1": sg_c0_i1, "c1_i1": sg_c1_i1},
            "correct":  {"text": st, "image": si, "group": sg},
            "triplets": {"caption_0": triplets_c0, "caption_1": triplets_c1},
            "responses": {
                "c0_i0": resp_c0_i0, "c1_i0": resp_c1_i0,
                "c0_i1": resp_c0_i1, "c1_i1": resp_c1_i1,
            },
        }

        per_example.append(row)

        # Live progress log every 10 examples
        if (idx + 1) % 10 == 0:
            for m in methods:
                n = idx + 1
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


# ─────────────────────────────────────────────────────────────
# Tag-level subset analysis
# ─────────────────────────────────────────────────────────────
def analyze_by_tag(per_example: list[dict], methods: list[str]) -> dict:
    """Break down accuracy by Winoground linguistic tag."""
    tag_counts = {}

    for ex in per_example:
        tag = ex.get("tag") or "untagged"
        if tag not in tag_counts:
            tag_counts[tag] = {m: {"text": 0, "image": 0, "group": 0, "n": 0}
                               for m in methods}

        for m in methods:
            if m not in ex:
                continue
            tag_counts[tag][m]["text"]  += int(ex[m]["correct"]["text"])
            tag_counts[tag][m]["image"] += int(ex[m]["correct"]["image"])
            tag_counts[tag][m]["group"] += int(ex[m]["correct"]["group"])
            tag_counts[tag][m]["n"]     += 1

    # Normalize
    tag_acc = {}
    for tag, data in tag_counts.items():
        tag_acc[tag] = {}
        for m, c in data.items():
            n = c["n"]
            tag_acc[tag][m] = {
                "text":  c["text"]  / n if n else 0,
                "image": c["image"] / n if n else 0,
                "group": c["group"] / n if n else 0,
                "n": n,
            }

    return tag_acc


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────
def print_summary(summary: dict, tag_analysis: dict | None = None):
    n = summary.get("n_evaluated", "?")
    methods = [k for k in summary if k != "n_evaluated"]

    print(f"\n{'═'*60}")
    print(f"  Winoground Results  (n={n})")
    print(f"{'═'*60}")
    print(f"  {'Method':<16}  {'Text':>8}  {'Image':>8}  {'Group':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Random chance':<16}  {'0.2500':>8}  {'0.2500':>8}  {'0.0625':>8}")
    for m in methods:
        s = summary[m]
        print(f"  {m:<16}  {s['text']:>8.4f}  {s['image']:>8.4f}  {s['group']:>8.4f}")
    print(f"{'═'*60}")

    if tag_analysis:
        print(f"\n  Per-tag breakdown (sg_cot):")
        print(f"  {'Tag':<28}  {'n':>4}  {'Text':>7}  {'Image':>7}  {'Group':>7}")
        print(f"  {'-'*58}")
        for tag, data in sorted(tag_analysis.items()):
            if "sg_cot" in data:
                d = data["sg_cot"]
                print(
                    f"  {tag:<28}  {d['n']:>4}  "
                    f"{d['text']:>7.3f}  {d['image']:>7.3f}  {d['group']:>7.3f}"
                )
    print()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLaVA + Scene Graph CoT evaluation on Winoground"
    )
    parser.add_argument(
        "--model_id", type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token (required for Winoground)",
    )
    parser.add_argument(
        "--spacy_model", type=str,
        default="en_core_web_trf",
        help="spaCy model for dependency parsing",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to N examples (None = full dataset)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=300,
        help="Max tokens for CoT generation",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device map (auto | cuda | cpu)",
    )
    parser.add_argument(
        "--no_baseline", action="store_true",
        help="Skip baseline yes/no scoring",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Auth
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
        log.info("Logged in to HuggingFace.")

    # Load components
    nlp              = load_spacy(args.spacy_model)
    model, processor = load_llava(args.model_id, device=args.device)

    # Load dataset
    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}' → {len(dataset[args.split])} examples")

    # Evaluate
    summary, per_example = evaluate(
        model=model,
        processor=processor,
        nlp=nlp,
        dataset=dataset,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        run_baseline=not args.no_baseline,
    )

    # Tag analysis
    methods = ["sg_cot"] + ([] if args.no_baseline else ["baseline"])
    tag_analysis = analyze_by_tag(per_example, methods)

    # Print
    print_summary(summary, tag_analysis)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = args.model_id.replace("/", "_")
    summary_path  = out_dir / f"{slug}_sg_cot_summary.json"
    details_path  = out_dir / f"{slug}_sg_cot_per_example.json"
    tags_path     = out_dir / f"{slug}_sg_cot_tags.json"

    with open(summary_path, "w")  as f: json.dump(summary,      f, indent=2)
    with open(details_path, "w")  as f: json.dump(per_example,  f, indent=2)
    with open(tags_path, "w")     as f: json.dump(tag_analysis, f, indent=2)

    log.info(f"Summary  → {summary_path}")
    log.info(f"Details  → {details_path}")
    log.info(f"Tags     → {tags_path}")


if __name__ == "__main__":
    main()