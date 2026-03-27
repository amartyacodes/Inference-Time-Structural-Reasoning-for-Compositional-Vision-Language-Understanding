"""
winoground_scene_graph_extract.py
──────────────────────────────────────────────────────────────────────────────
Extracts TEXT scene graphs from the Winoground dataset using
Qwen3-VL-8B-Thinking and saves structured triples to JSON.

Model: Qwen/Qwen3-VL-8B-Thinking

Output JSON schema per Winoground item:
{
  "id": int,
  "caption_0": str,
  "caption_1": str,
  "text_scene_graph_0": {
      "entities": [...],
      "relations": [...],
      "triples": [
          {
              "subject":            str,       # entity label
              "predicate":          str,       # action / relation
              "object":             str,       # entity label
              "subject_attributes": [str],     # adjectives for subject
              "object_attributes":  [str],     # adjectives for object
              "spatial_detail":     str|null,  # explicit spatial info
              "subject_count":      int|null,
              "object_count":       int|null,
          },
          ...
      ]
  },
  "text_scene_graph_1": { ... },
  "thinking": { "text_sg_0": str, "text_sg_1": str },
  "meta": { ... }
}
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

TEXT_PROMPT_VERSION = "v3.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════════════════════

TEXT_SG_PROMPT = """\
You are a precise linguistic scene graph parser. Given a short image caption, \
extract a structured scene graph that captures ALL entities, their counts, their \
visual attributes, and every spatial/relational/action predicate between them.

OUTPUT — respond with ONLY a valid JSON object. No markdown fences, no prose, no preamble.

Schema:
{{
  "entities": [
    {{
      "id": "e0",
      "label": "<noun>",
      "count": <integer if stated, else null>,
      "attributes": ["<adj_or_state>", ...]
    }},
    ...
  ],
  "relations": [
    {{
      "subject": "<entity_id>",
      "predicate": "<verb_or_preposition>",
      "object": "<entity_id>",
      "spatial_detail": "<spatial string or null>"
    }},
    ...
  ]
}}

Rules:
1. Every distinct noun phrase referring to a real-world entity is its own entry in "entities".
2. count: set to integer when explicitly stated ("two dogs" → 2, "a cat" → 1); null if ambiguous.
3. attributes: color, size, material, texture, pose, age, emotional state.
   Do NOT put count inside attributes.
4. predicate: the core action or relation — concise and grounded.
   Examples: "holds", "kisses", "hugs", "bites", "chases", "feeds", "rides",
             "sits on", "stands next to", "wears", "carries", "looks at".
5. spatial_detail: fill ONLY when the caption explicitly states position:
   "to the left of", "to the right of", "above", "below", "in front of",
   "behind", "between", "on top of". Use null otherwise.
6. Agent/patient direction is critical:
   subject = AGENT (who does the action), object = PATIENT (receives it).
   "a dog biting a man" → subject=dog, predicate="bites", object=man.
7. If no explicit relation is stated, use predicate "in scene", spatial_detail null.
8. Do NOT invent entities absent from the caption.
9. Entity IDs are zero-indexed strings: "e0", "e1", "e2", ...

Caption: "{caption}"

Return ONLY the JSON object.
"""


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"

_GENERATE_KWARGS = dict(
    max_new_tokens=4096,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
)


def load_model(device: str, flash_attn: bool = False):
    log.info(f"Loading {MODEL_NAME} ...")
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    log.info("Model ready.")
    return model, processor


def _run(model, processor, caption: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": TEXT_SG_PROMPT.format(caption=caption)},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **_GENERATE_KWARGS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# ══════════════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════════════

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE     = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_all_json_objects(text: str) -> list[dict]:
    """Extract every valid top-level JSON object from text."""
    results, i = [], 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        results.append(json.loads(text[i:j + 1]))
                    except json.JSONDecodeError:
                        pass
                    break
            j += 1
        i = j + 1
    return results


def _score_schema(obj: dict) -> int:
    """Score how well a dict matches the SG schema. Higher = better match."""
    score = 0
    if "entities" in obj:
        score += 10
        ents = obj["entities"]
        if isinstance(ents, list) and ents:
            score += 5
            if all("id" in e and "label" in e for e in ents):
                score += 5
    if "relations" in obj:
        score += 10
        rels = obj["relations"]
        if isinstance(rels, list):
            score += 3
            if all("predicate" in r for r in rels):
                score += 5
    return score


def _parse_raw(raw: str, tag: str) -> dict:
    """
    Parse raw model output into a scene graph dict.
    Handles: tagged <think>, untagged CoT, multiple JSON objects in output.
    Always picks the JSON object that best matches the SG schema.
    """
    cleaned = _FENCE_RE.sub("", raw).strip()

    # Extract tagged think block if present
    think_block = ""
    m = _THINK_TAG_RE.search(cleaned)
    if m:
        think_block = m.group(1).strip()
        cleaned = _THINK_TAG_RE.sub("", cleaned).strip()
    
    # Scan ALL JSON objects in the full output and pick the best one
    candidates = _extract_all_json_objects(raw)  # use raw to catch everything
    if not candidates:
        log.warning(f"[{tag}] No JSON found in output.")
        return {"parse_error": True, "raw_output": raw, "_think": think_block}

    best = max(candidates, key=_score_schema)
    best_score = _score_schema(best)

    if best_score < 10:
        log.warning(f"[{tag}] Best candidate score={best_score} (too low). "
                    f"Keys: {list(best.keys())}")
        return {"parse_error": True, "raw_output": raw, "_think": think_block}

    # If think block wasn't tagged, everything before the first '{' in cleaned is CoT
    if not think_block:
        first_brace = cleaned.find("{")
        if first_brace > 0:
            think_block = cleaned[:first_brace].strip()

    best["_think"] = think_block
    return best


# ══════════════════════════════════════════════════════════════════════════════
# TRIPLE BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def _build_triples(sg: dict) -> list[dict]:
    """
    Convert the parsed scene graph into structured triples:
    {
        subject, predicate, object,
        subject_attributes, object_attributes,
        spatial_detail,
        subject_count, object_count
    }
    """
    if sg.get("parse_error"):
        return []

    entity_map = {e["id"]: e for e in sg.get("entities", [])}
    triples = []

    for rel in sg.get("relations", []):
        subj_id = rel.get("subject", "")
        obj_id  = rel.get("object",  "")
        subj_e  = entity_map.get(subj_id, {})
        obj_e   = entity_map.get(obj_id,  {})

        triples.append({
            "subject":            subj_e.get("label", subj_id),
            "predicate":          rel.get("predicate", "?"),
            "object":             obj_e.get("label", obj_id),
            "subject_attributes": subj_e.get("attributes", []),
            "object_attributes":  obj_e.get("attributes",  []),
            "spatial_detail":     rel.get("spatial_detail", None),
            "subject_count":      subj_e.get("count", None),
            "object_count":       obj_e.get("count",  None),
        })

    return triples


def query_text_sg(model, processor, caption: str) -> dict:
    raw = _run(model, processor, caption)
    sg  = _parse_raw(raw, f"text_sg|{caption[:50]}")
    sg["triples"] = _build_triples(sg)
    return sg


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT I/O
# ══════════════════════════════════════════════════════════════════════════════

def _save(path: Path, items: list, meta: dict):
    with open(path, "w") as f:
        json.dump({"meta": meta, "count": len(items), "items": items}, f, indent=2)


def _load_checkpoint(path: Path) -> tuple[list, set]:
    if not path.exists():
        return [], set()
    with open(path) as f:
        data = json.load(f)
    items = data.get("items", [])
    log.info(f"Resuming — {len(items)} items already in checkpoint.")
    return items, {r["id"] for r in items}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def extract_scene_graphs(
    output_path: str = "winoground_text_sgs.json",
    device: str = "cuda",
    flash_attn: bool = False,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    checkpoint_every: int = 10,
    hf_token: Optional[str] = None,
):
    log.info("Loading Winoground ...")
    load_kw = {"split": "test"}
    if hf_token:
        load_kw["token"] = hf_token
    dataset = load_dataset("facebook/winoground", **load_kw)

    if end_idx is None:
        end_idx = len(dataset)
    subset = dataset.select(range(start_idx, end_idx))
    log.info(f"Items {start_idx}–{end_idx - 1}  ({len(subset)} total)")

    model, processor = load_model(device, flash_attn)

    out_path = Path(output_path)
    results, processed_ids = _load_checkpoint(out_path)

    meta = {
        "model":               MODEL_NAME,
        "timestamp":           datetime.utcnow().isoformat() + "Z",
        "text_prompt_version": TEXT_PROMPT_VERSION,
        "generate_kwargs":     _GENERATE_KWARGS,
    }

    for i, item in enumerate(subset):
        item_id = item["id"]

        if item_id in processed_ids:
            log.info(f"[{i + 1}/{len(subset)}] Skip id={item_id}")
            continue

        log.info(f"[{i + 1}/{len(subset)}] id={item_id}")
        t0 = time.time()

        cap0, cap1 = item["caption_0"], item["caption_1"]

        log.info(f"  text_sg_0 ← '{cap0[:70]}'")
        tsg0 = query_text_sg(model, processor, cap0)

        log.info(f"  text_sg_1 ← '{cap1[:70]}'")
        tsg1 = query_text_sg(model, processor, cap1)

        thinking = {
            "text_sg_0": tsg0.pop("_think", ""),
            "text_sg_1": tsg1.pop("_think", ""),
        }

        results.append({
            "id":                 item_id,
            "caption_0":          cap0,
            "caption_1":          cap1,
            "text_scene_graph_0": tsg0,
            "text_scene_graph_1": tsg1,
            "thinking":           thinking,
        })
        processed_ids.add(item_id)
        log.info(f"  ✓ {time.time() - t0:.1f}s  "
                 f"triples: {len(tsg0.get('triples', []))} / {len(tsg1.get('triples', []))}")

        if (i + 1) % checkpoint_every == 0:
            _save(out_path, results, meta)
            log.info(f"  💾 checkpoint ({len(results)} items)")

    _save(out_path, results, meta)
    log.info(f"\n✅ Done — {len(results)} items → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Extract text scene graphs from Winoground using Qwen3-VL-8B-Thinking"
    )
    p.add_argument("--output",     default="winoground_text_sgs.json")
    p.add_argument("--device",     default="cuda",
                   help="cuda / cpu / cuda:0 / cuda:1 ...")
    p.add_argument("--flash-attn", action="store_true",
                   help="Enable flash_attention_2 (requires flash-attn installed)")
    p.add_argument("--start",      type=int, default=0)
    p.add_argument("--end",        type=int, default=None)
    p.add_argument("--checkpoint", type=int, default=1)
    p.add_argument("--hf-token",   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_scene_graphs(
        output_path=args.output,
        device=args.device,
        flash_attn=args.flash_attn,
        start_idx=args.start,
        end_idx=args.end,
        checkpoint_every=args.checkpoint,
        hf_token=args.hf_token,
    )