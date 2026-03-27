"""
Unified Winoground Evaluation: CLIP, BLIP, LLaVA, Qwen3-VL-Embedding, Qwen3-VL-Thinking
=========================================================================================

SEQUENTIAL GPU MANAGEMENT:
  Models are loaded, evaluated, and unloaded one group at a time to avoid
  CUDA OOM. Order: CLIP → BLIP → LLaVA → Qwen3-Emb → Qwen3-Gen (Thinking).
  Each model is deleted + torch.cuda.empty_cache() before the next loads.

SCORING DESIGN:
  CLIP       → CLS token cosine similarity (dual encoder)
  BLIP       → ITM head P(match) from BlipForImageTextRetrieval (cross-attention)
  LLaVA      → P(yes) from next-token logits (instruction-tuned generative)
  Qwen3 Emb  → EOS-token cosine similarity (contrastive InfoNCE trained decoder)
               Tested with and without task instruction.
  Qwen3 Gen  → P(yes) from next-token logits (Qwen3-VL-8B-Thinking, generative)
               Comparable to LLaVA scoring — both are decoder-only P(yes).

SCENE GRAPH (YUKINO-SG TextSceneGraphParser):
  For CLIP/BLIP/Qwen3-Emb: additive λ × graph_prior on base score.
  For LLaVA/Qwen3-Gen:     triples injected into prompt when non-empty.
  Empty SG:                 base score unchanged (graph_prior = 0 / plain prompt).

STRATEGIES REPORTED (plain vs SG-augmented for all):
  clip, clip_sg
  blip, blip_sg
  llava, llava_sg
  qwen3, qwen3_sg                  (embedding, no instruction)
  qwen3_inst, qwen3_inst_sg        (embedding, with instruction)
  qwen3_gen, qwen3_gen_sg          (generative P(yes), Thinking model)

Usage:
    python winoground_eval_v2.py --methods all
    python winoground_eval_v2.py --methods clip blip --max_samples 50
    python winoground_eval_v2.py --methods qwen3 qwen3_gen --max_samples 50
    python winoground_eval_v2.py --methods llava qwen3_gen --max_samples 50
"""

import gc
import json
import argparse
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
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

ALL_METHODS = ["clip", "blip", "llava", "qwen3", "qwen3_gen"]


# ═════════════════════════════════════════════════════════════
# GPU Memory Management
# ═════════════════════════════════════════════════════════════

def free_gpu(*objects):
    """Delete objects and aggressively free CUDA memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info(f"GPU freed. Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB" if torch.cuda.is_available() else "No CUDA")


# ═════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════

def _to_tensor(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            val = getattr(out, attr)
            if attr == "last_hidden_state":
                return val[:, 0, :]
            return val
    if hasattr(out, "__getitem__"):
        return out[0]
    raise TypeError(f"Cannot extract tensor from {type(out)}")


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), seq_lens]


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
# Graph Asymmetry Scorer
# ═════════════════════════════════════════════════════════════

class GraphAsymmetryScorer:
    def __init__(self, alpha=1.0, beta=1.5, gamma=1.0, lam=0.3):
        self.alpha, self.beta, self.gamma, self.lam = alpha, beta, gamma, lam
        self._text_sim_cache: dict[tuple[str, str], float] = {}

    def _text_sim(self, a, b, encoder):
        key = (a, b)
        if key not in self._text_sim_cache:
            ea, eb = encoder.embed_text(a), encoder.embed_text(b)
            self._text_sim_cache[key] = float(np.dot(ea, eb))
            self._text_sim_cache[(b, a)] = self._text_sim_cache[key]
        return self._text_sim_cache[key]

    def _pair_asymmetry(self, ta, tb, encoder):
        fwd = (self.alpha * self._text_sim(ta.subject, tb.subject, encoder) +
               self.beta  * self._text_sim(ta.relation, tb.relation, encoder) +
               self.gamma * self._text_sim(ta.obj, tb.obj, encoder))
        flp = (self.alpha * self._text_sim(ta.subject, tb.obj, encoder) +
               self.beta  * self._text_sim(ta.relation, tb.relation, encoder) +
               self.gamma * self._text_sim(ta.obj, tb.subject, encoder))
        return fwd - flp

    def graph_prior(self, triples_a, triples_b, encoder):
        """Returns 0.0 when either set is empty — base score unchanged."""
        if not triples_a or not triples_b:
            return 0.0
        n, m = len(triples_a), len(triples_b)
        cost = np.zeros((n, m))
        for i, ta in enumerate(triples_a):
            for j, tb in enumerate(triples_b):
                cost[i, j] = self._pair_asymmetry(ta, tb, encoder)
        row_ind, col_ind = linear_sum_assignment(-np.abs(cost))
        return float(cost[row_ind, col_ind].mean())

    def augment(self, base_score, triples_cap, triples_other, encoder):
        """Add λ × graph_prior. Returns base_score unchanged when SG is empty."""
        return base_score + self.lam * self.graph_prior(triples_cap, triples_other, encoder)

    def clear_cache(self):
        """Clear text similarity cache between model phases to avoid stale encoder refs."""
        self._text_sim_cache.clear()


# ═════════════════════════════════════════════════════════════
# Text Encoders for Graph Scoring
# ═════════════════════════════════════════════════════════════

class CLIPTextEncoder:
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        if text in self._cache: return self._cache[text]
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        out = self.model.get_text_features(**inputs)
        emb = _to_tensor(out)
        emb = F.normalize(emb.float(), dim=-1).cpu().numpy()
        if emb.ndim == 2: emb = emb[0]
        self._cache[text] = emb
        return emb


class BLIPTextEncoder:
    """Mean-pooled text encoder hidden states for graph word-level similarity."""
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        if text in self._cache: return self._cache[text]
        dummy = Image.new("RGB", (1, 1), color=(255, 255, 255))
        inputs = self.processor(images=dummy, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        if hasattr(outputs, "question_embeds") and outputs.question_embeds is not None:
            hidden = outputs.question_embeds
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs.itm_score.new_zeros(1, 1, 1)
        if hidden.ndim == 3:
            mask = inputs.get("attention_mask", torch.ones(hidden.shape[:2], device=self.device))
            mask = mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden
        emb = F.normalize(pooled.float(), dim=-1).cpu().numpy()
        if emb.ndim == 2: emb = emb[0]
        self._cache[text] = emb
        return emb


class Qwen3TextEncoder:
    """EOS-pooled text encoder for graph word-level similarity."""
    def __init__(self, model, processor, device):
        self.model, self.processor, self.device = model, processor, device
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        if text in self._cache: return self._cache[text]
        inputs = self.processor.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(self.device)
        outputs = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        last_hs = outputs.hidden_states[-1]
        pooled = _last_token_pool(last_hs, inputs["attention_mask"])
        emb = F.normalize(pooled.float(), dim=-1).cpu().numpy()
        if emb.ndim == 2: emb = emb[0]
        self._cache[text] = emb
        return emb


# ═════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════

def load_clip(model_id="openai/clip-vit-base-patch32", device="cuda:0"):
    log.info(f"Loading CLIP: {model_id}")
    d = device if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(d).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, d

def load_blip(model_id=None, device="cuda:0"):
    if model_id is None: model_id = "Salesforce/blip-itm-base-coco"
    log.info(f"Loading BLIP (ITM): {model_id}")
    processor = BlipProcessor.from_pretrained(model_id)
    d = device if torch.cuda.is_available() else "cpu"
    model = BlipForImageTextRetrieval.from_pretrained(model_id).to(d).eval()
    return model, processor, d

def load_llava(model_id="llava-hf/llava-1.5-7b-hf", device="cuda:0"):
    log.info(f"Loading LLaVA: {model_id}")
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device, low_cpu_mem_usage=True,
    ).eval()
    return model, processor

def load_qwen3(model_id="Qwen/Qwen3-VL-Embedding-8B", device="cuda:0"):
    """Load Qwen3-VL-Embedding for contrastive EOS-pooled scoring."""
    if not HAS_QWEN3VL:
        raise ImportError("Qwen3VLForConditionalGeneration not found. pip install transformers>=4.57.0")
    log.info(f"Loading Qwen3-VL-Embedding: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device, low_cpu_mem_usage=True,
    ).eval()
    return model, processor

def load_qwen3_gen(model_id="Qwen/Qwen3-VL-8B-Thinking", device="cuda:0"):
    """Load Qwen3-VL-8B-Thinking for generative P(yes) scoring."""
    if not HAS_QWEN3VL:
        raise ImportError("Qwen3VLForConditionalGeneration not found. pip install transformers>=4.57.0")
    log.info(f"Loading Qwen3-VL-Thinking (generative): {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device, low_cpu_mem_usage=True,
    ).eval()
    return model, processor


# ═════════════════════════════════════════════════════════════
# Scoring Primitives
# ═════════════════════════════════════════════════════════════

# ── CLIP ────────────────────────────────────────────────────

@torch.no_grad()
def score_clip(model, processor, device, image, caption):
    img_in = processor(images=image, return_tensors="pt")
    txt_in = processor(text=[caption], return_tensors="pt", padding=True)
    combined = {k: v.to(device) for k, v in {**img_in, **txt_in}.items()}
    out = model(**combined)
    img_e = F.normalize(out.image_embeds.float(), dim=-1)[0]
    txt_e = F.normalize(out.text_embeds.float(), dim=-1)[0]
    return ((img_e * txt_e).sum().item() + 1) / 2

# ── BLIP ────────────────────────────────────────────────────

@torch.no_grad()
def score_blip(model, processor, device, image, caption):
    inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
    out = model(**inputs)
    return F.softmax(out.itm_score, dim=1)[0, 1].item()

# ── Qwen3-VL-Embedding (contrastive) ───────────────────────

QWEN3_INSTRUCTION = "Determine whether the image and text match each other."

@torch.no_grad()
def _qwen3_embed(qwen3_model, qwen3_processor, content_list, instruction=None):
    content = list(content_list)
    if instruction:
        content = [{"type": "text", "text": f"Instruct: {instruction}"}] + content
    messages = [{"role": "user", "content": content}]
    inputs = qwen3_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    device = next(qwen3_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = qwen3_model(**inputs, output_hidden_states=True)
    last_hs = outputs.hidden_states[-1]
    attn = inputs.get("attention_mask", torch.ones(last_hs.shape[:2], device=device))
    pooled = _last_token_pool(last_hs, attn)
    return F.normalize(pooled[0].float(), dim=0)


def score_qwen3(model, processor, image, caption, instruction=None):
    img_emb = _qwen3_embed(model, processor, [{"type": "image", "image": image}], instruction)
    txt_emb = _qwen3_embed(model, processor, [{"type": "text", "text": caption}], instruction)
    return ((img_emb * txt_emb).sum().item() + 1) / 2


# ── Qwen3-VL-Thinking (generative P(yes)) ──────────────────

def _format_sg(triples):
    return "\n".join(f"  - {t.subject}  [{t.relation}]  {t.obj}" for t in triples)


def _build_qwen3_gen_prompt(caption, triples, use_sg):
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


@torch.no_grad()
def score_qwen3_gen(model, processor, image, caption, triples, use_sg):
    text_content = _build_qwen3_gen_prompt(caption, triples, use_sg)
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


def _resolve_yes_no_ids(tokenizer, word: str) -> torch.Tensor:
    candidates = set()
    for variant in [word, word.capitalize(), word.upper()]:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            candidates.add(ids[0])
    if not candidates:
        raise ValueError(f"Could not resolve token IDs for '{word}'")
    return torch.tensor(list(candidates), dtype=torch.long)


# ── LLaVA ───────────────────────────────────────────────────

def _build_llava_prompt(caption, triples, use_sg):
    if use_sg and triples:
        sg = _format_sg(triples)
        return (
            f"USER: <image>\nCaption: '{caption}'\n\n"
            f"The caption has the following scene graph relations:\n{sg}\n\n"
            f"Using the scene graph as a guide, pay close attention to which "
            f"entity is doing what and any spatial relationships. "
            f"Does this image match the caption?\nAnswer yes or no.\nASSISTANT:"
        )
    else:
        return (
            f"USER: <image>\nDoes this image match the caption: '{caption}'?\n"
            f"Answer yes or no.\nASSISTANT:"
        )

@torch.no_grad()
def score_llava(model, processor, image, caption, triples, use_sg):
    prompt = _build_llava_prompt(caption, triples, use_sg)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits
    last = logits[0, -1]
    yes_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = processor.tokenizer.encode("no",  add_special_tokens=False)[0]
    return torch.softmax(torch.stack([last[yes_id], last[no_id]]), dim=0)[0].item()


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
# Per-Method Evaluation Phases (sequential, GPU-safe)
# ═════════════════════════════════════════════════════════════

def _prepare_data(dataset, split, max_samples):
    """Pre-extract images and captions to avoid re-reading per phase."""
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    examples = []
    for idx, example in enumerate(data):
        examples.append({
            "idx": idx,
            "img0": example["image_0"].convert("RGB"),
            "img1": example["image_1"].convert("RGB"),
            "cap0": example["caption_0"],
            "cap1": example["caption_1"],
            "tag":  example.get("tag", ""),
        })
    return examples


def _init_per_example(examples, sg_parser):
    """Initialize per_example rows with scene graph parses (done once, CPU only)."""
    per_example = []
    sg_cache = {}  # idx → (t0, t1)
    for ex in examples:
        t0 = sg_parser.parse(ex["cap0"])
        t1 = sg_parser.parse(ex["cap1"])
        sg_cache[ex["idx"]] = (t0, t1)
        per_example.append({
            "idx": ex["idx"],
            "caption_0": ex["cap0"],
            "caption_1": ex["cap1"],
            "tag": ex["tag"],
            "sg_cap0": [repr(t) for t in t0],
            "sg_cap1": [repr(t) for t in t1],
        })
    return per_example, sg_cache


# ── CLIP phase ──────────────────────────────────────────────

def run_clip_phase(examples, sg_cache, per_example, sg_scorer, args):
    log.info("═══ CLIP phase ═══")
    model, processor, device = load_clip(args.clip_model_id, args.device)
    encoder = CLIPTextEncoder(model, processor, device)

    counts = {"clip": {"text": 0, "image": 0, "group": 0},
              "clip_sg": {"text": 0, "image": 0, "group": 0}}

    for ex in tqdm(examples, desc="CLIP"):
        i = ex["idx"]
        t0, t1 = sg_cache[i]

        s00 = score_clip(model, processor, device, ex["img0"], ex["cap0"])
        s10 = score_clip(model, processor, device, ex["img0"], ex["cap1"])
        s01 = score_clip(model, processor, device, ex["img1"], ex["cap0"])
        s11 = score_clip(model, processor, device, ex["img1"], ex["cap1"])

        tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
        counts["clip"]["text"] += int(tc); counts["clip"]["image"] += int(ic); counts["clip"]["group"] += int(gc)
        per_example[i]["clip"] = _make_row(s00, s10, s01, s11, tc, ic, gc)

        a00 = sg_scorer.augment(s00, t0, t1, encoder)
        a10 = sg_scorer.augment(s10, t1, t0, encoder)
        a01 = sg_scorer.augment(s01, t0, t1, encoder)
        a11 = sg_scorer.augment(s11, t1, t0, encoder)
        tc, ic, gc = winoground_metrics(a00, a10, a01, a11)
        counts["clip_sg"]["text"] += int(tc); counts["clip_sg"]["image"] += int(ic); counts["clip_sg"]["group"] += int(gc)
        per_example[i]["clip_sg"] = _make_row(a00, a10, a01, a11, tc, ic, gc)

    sg_scorer.clear_cache()
    free_gpu(model, processor, encoder)
    return counts


# ── BLIP phase ──────────────────────────────────────────────

def run_blip_phase(examples, sg_cache, per_example, sg_scorer, args):
    log.info("═══ BLIP phase ═══")
    model, processor, device = load_blip(args.blip_model_id, args.device)
    encoder = BLIPTextEncoder(model, processor, device)

    counts = {"blip": {"text": 0, "image": 0, "group": 0},
              "blip_sg": {"text": 0, "image": 0, "group": 0}}

    for ex in tqdm(examples, desc="BLIP"):
        i = ex["idx"]
        t0, t1 = sg_cache[i]

        s00 = score_blip(model, processor, device, ex["img0"], ex["cap0"])
        s10 = score_blip(model, processor, device, ex["img0"], ex["cap1"])
        s01 = score_blip(model, processor, device, ex["img1"], ex["cap0"])
        s11 = score_blip(model, processor, device, ex["img1"], ex["cap1"])

        tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
        counts["blip"]["text"] += int(tc); counts["blip"]["image"] += int(ic); counts["blip"]["group"] += int(gc)
        per_example[i]["blip"] = _make_row(s00, s10, s01, s11, tc, ic, gc)

        a00 = sg_scorer.augment(s00, t0, t1, encoder)
        a10 = sg_scorer.augment(s10, t1, t0, encoder)
        a01 = sg_scorer.augment(s01, t0, t1, encoder)
        a11 = sg_scorer.augment(s11, t1, t0, encoder)
        tc, ic, gc = winoground_metrics(a00, a10, a01, a11)
        counts["blip_sg"]["text"] += int(tc); counts["blip_sg"]["image"] += int(ic); counts["blip_sg"]["group"] += int(gc)
        per_example[i]["blip_sg"] = _make_row(a00, a10, a01, a11, tc, ic, gc)

    sg_scorer.clear_cache()
    free_gpu(model, processor, encoder)
    return counts


# ── LLaVA phase ─────────────────────────────────────────────

def run_llava_phase(examples, sg_cache, per_example, args):
    log.info("═══ LLaVA phase ═══")
    model, processor = load_llava(args.llava_model_id, args.device)

    counts = {"llava": {"text": 0, "image": 0, "group": 0},
              "llava_sg": {"text": 0, "image": 0, "group": 0}}

    for ex in tqdm(examples, desc="LLaVA"):
        i = ex["idx"]
        t0, t1 = sg_cache[i]

        # Plain
        p00 = score_llava(model, processor, ex["img0"], ex["cap0"], t0, use_sg=False)
        p10 = score_llava(model, processor, ex["img0"], ex["cap1"], t1, use_sg=False)
        p01 = score_llava(model, processor, ex["img1"], ex["cap0"], t0, use_sg=False)
        p11 = score_llava(model, processor, ex["img1"], ex["cap1"], t1, use_sg=False)
        tc, ic, gc = winoground_metrics(p00, p10, p01, p11)
        counts["llava"]["text"] += int(tc); counts["llava"]["image"] += int(ic); counts["llava"]["group"] += int(gc)
        per_example[i]["llava"] = _make_row(p00, p10, p01, p11, tc, ic, gc)

        # SG prompt
        s00 = score_llava(model, processor, ex["img0"], ex["cap0"], t0, use_sg=True)
        s10 = score_llava(model, processor, ex["img0"], ex["cap1"], t1, use_sg=True)
        s01 = score_llava(model, processor, ex["img1"], ex["cap0"], t0, use_sg=True)
        s11 = score_llava(model, processor, ex["img1"], ex["cap1"], t1, use_sg=True)
        tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
        counts["llava_sg"]["text"] += int(tc); counts["llava_sg"]["image"] += int(ic); counts["llava_sg"]["group"] += int(gc)
        per_example[i]["llava_sg"] = _make_row(s00, s10, s01, s11, tc, ic, gc)

    free_gpu(model, processor)
    return counts


# ── Qwen3-Embedding phase ──────────────────────────────────

def run_qwen3_emb_phase(examples, sg_cache, per_example, sg_scorer, args):
    log.info("═══ Qwen3-VL-Embedding phase ═══")
    model, processor = load_qwen3(args.qwen3_model_id, args.device)
    device = next(model.parameters()).device
    encoder = Qwen3TextEncoder(model, processor, device)

    counts = {k: {"text": 0, "image": 0, "group": 0}
              for k in ["qwen3", "qwen3_sg", "qwen3_inst", "qwen3_inst_sg"]}

    for ex in tqdm(examples, desc="Qwen3-Emb"):
        i = ex["idx"]
        t0, t1 = sg_cache[i]

        # No instruction
        s00 = score_qwen3(model, processor, ex["img0"], ex["cap0"])
        s10 = score_qwen3(model, processor, ex["img0"], ex["cap1"])
        s01 = score_qwen3(model, processor, ex["img1"], ex["cap0"])
        s11 = score_qwen3(model, processor, ex["img1"], ex["cap1"])

        tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
        counts["qwen3"]["text"] += int(tc); counts["qwen3"]["image"] += int(ic); counts["qwen3"]["group"] += int(gc)
        per_example[i]["qwen3"] = _make_row(s00, s10, s01, s11, tc, ic, gc)

        a00 = sg_scorer.augment(s00, t0, t1, encoder)
        a10 = sg_scorer.augment(s10, t1, t0, encoder)
        a01 = sg_scorer.augment(s01, t0, t1, encoder)
        a11 = sg_scorer.augment(s11, t1, t0, encoder)
        tc, ic, gc = winoground_metrics(a00, a10, a01, a11)
        counts["qwen3_sg"]["text"] += int(tc); counts["qwen3_sg"]["image"] += int(ic); counts["qwen3_sg"]["group"] += int(gc)
        per_example[i]["qwen3_sg"] = _make_row(a00, a10, a01, a11, tc, ic, gc)

        # With instruction
        s00 = score_qwen3(model, processor, ex["img0"], ex["cap0"], instruction=QWEN3_INSTRUCTION)
        s10 = score_qwen3(model, processor, ex["img0"], ex["cap1"], instruction=QWEN3_INSTRUCTION)
        s01 = score_qwen3(model, processor, ex["img1"], ex["cap0"], instruction=QWEN3_INSTRUCTION)
        s11 = score_qwen3(model, processor, ex["img1"], ex["cap1"], instruction=QWEN3_INSTRUCTION)

        tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
        counts["qwen3_inst"]["text"] += int(tc); counts["qwen3_inst"]["image"] += int(ic); counts["qwen3_inst"]["group"] += int(gc)
        per_example[i]["qwen3_inst"] = _make_row(s00, s10, s01, s11, tc, ic, gc)

        a00 = sg_scorer.augment(s00, t0, t1, encoder)
        a10 = sg_scorer.augment(s10, t1, t0, encoder)
        a01 = sg_scorer.augment(s01, t0, t1, encoder)
        a11 = sg_scorer.augment(s11, t1, t0, encoder)
        tc, ic, gc = winoground_metrics(a00, a10, a01, a11)
        counts["qwen3_inst_sg"]["text"] += int(tc); counts["qwen3_inst_sg"]["image"] += int(ic); counts["qwen3_inst_sg"]["group"] += int(gc)
        per_example[i]["qwen3_inst_sg"] = _make_row(a00, a10, a01, a11, tc, ic, gc)

    sg_scorer.clear_cache()
    free_gpu(model, processor, encoder)
    return counts


# ── Qwen3-VL-Thinking (generative) phase ───────────────────

def run_qwen3_gen_phase(examples, sg_cache, per_example, args):
    log.info("═══ Qwen3-VL-Thinking (generative) phase ═══")
    model, processor = load_qwen3_gen(args.qwen3_gen_model_id, args.device)

    counts = {"qwen3_gen": {"text": 0, "image": 0, "group": 0},
              "qwen3_gen_sg": {"text": 0, "image": 0, "group": 0}}

    for ex in tqdm(examples, desc="Qwen3-Gen"):
        i = ex["idx"]
        t0, t1 = sg_cache[i]

        # Plain
        p00 = score_qwen3_gen(model, processor, ex["img0"], ex["cap0"], t0, use_sg=False)
        p10 = score_qwen3_gen(model, processor, ex["img0"], ex["cap1"], t1, use_sg=False)
        p01 = score_qwen3_gen(model, processor, ex["img1"], ex["cap0"], t0, use_sg=False)
        p11 = score_qwen3_gen(model, processor, ex["img1"], ex["cap1"], t1, use_sg=False)
        tc, ic, gc = winoground_metrics(p00, p10, p01, p11)
        counts["qwen3_gen"]["text"] += int(tc); counts["qwen3_gen"]["image"] += int(ic); counts["qwen3_gen"]["group"] += int(gc)
        per_example[i]["qwen3_gen"] = _make_row(p00, p10, p01, p11, tc, ic, gc)

        # SG prompt
        s00 = score_qwen3_gen(model, processor, ex["img0"], ex["cap0"], t0, use_sg=True)
        s10 = score_qwen3_gen(model, processor, ex["img0"], ex["cap1"], t1, use_sg=True)
        s01 = score_qwen3_gen(model, processor, ex["img1"], ex["cap0"], t0, use_sg=True)
        s11 = score_qwen3_gen(model, processor, ex["img1"], ex["cap1"], t1, use_sg=True)
        tc, ic, gc = winoground_metrics(s00, s10, s01, s11)
        counts["qwen3_gen_sg"]["text"] += int(tc); counts["qwen3_gen_sg"]["image"] += int(ic); counts["qwen3_gen_sg"]["group"] += int(gc)
        per_example[i]["qwen3_gen_sg"] = _make_row(s00, s10, s01, s11, tc, ic, gc)

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
            result[tag][s] = {"text": c["text"]/n if n else 0, "image": c["image"]/n if n else 0,
                              "group": c["group"]/n if n else 0, "n": n}
    return result


# ═════════════════════════════════════════════════════════════
# Reporting
# ═════════════════════════════════════════════════════════════

STRATEGY_LABELS = {
    "clip":            ("CLIP",                        "CLS cosine",          "—"),
    "clip_sg":         ("CLIP + SG",                   "CLS cosine + SG",     "additive"),
    "blip":            ("BLIP ITM",                    "P(match)",            "—"),
    "blip_sg":         ("BLIP ITM + SG",               "P(match) + SG",       "additive"),
    "llava":           ("LLaVA",                       "P(yes)",              "—"),
    "llava_sg":        ("LLaVA + SG",                  "P(yes) + SG prompt",  "prompt"),
    "qwen3":           ("Qwen3-VL-Emb",               "EOS cosine",          "—"),
    "qwen3_sg":        ("Qwen3-VL-Emb + SG",          "EOS cosine + SG",     "additive"),
    "qwen3_inst":      ("Qwen3-VL-Emb (inst)",        "EOS cosine",          "—"),
    "qwen3_inst_sg":   ("Qwen3-VL-Emb (inst) + SG",   "EOS cosine + SG",     "additive"),
    "qwen3_gen":       ("Qwen3-VL-Think",             "P(yes)",              "—"),
    "qwen3_gen_sg":    ("Qwen3-VL-Think + SG",        "P(yes) + SG prompt",  "prompt"),
}

def print_summary(summary, tag_analysis=None):
    n = summary.get("n_evaluated", "?")
    strats = [k for k in summary if k != "n_evaluated"]
    W, M, S, G = 30, 18, 10, 8

    print(f"\n{'═' * 90}")
    print(f"  Winoground Results  (n={n})")
    print(f"  SG: YUKINO-SG TextSceneGraphParser  |  empty SG → base score unchanged")
    print(f"  GPU: sequential load/unload per model (no concurrent loading)")
    print(f"{'═' * 90}")
    print(f"  {'Strategy':<{W}}  {'Scoring':<{M}}  {'SG mode':<{S}}  {'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}")
    print(f"  {'-' * 86}")
    print(f"  {'Random chance':<{W}}  {'—':<{M}}  {'—':<{S}}  {'0.250':>{G}}  {'0.250':>{G}}  {'0.063':>{G}}")

    for s in strats:
        if s not in STRATEGY_LABELS: continue
        label, mech, sg_mode = STRATEGY_LABELS[s]
        v = summary[s]
        print(f"  {label:<{W}}  {mech:<{M}}  {sg_mode:<{S}}  "
              f"{v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}")

    print(f"{'═' * 90}")
    print(f"\n  Notes:")
    print(f"   CLIP/Qwen3-Emb: embedding cosine similarity")
    print(f"   BLIP: ITM head P(match) via cross-attention")
    print(f"   LLaVA: P(yes) logit; SG triples injected into prompt when non-empty")
    print(f"   Qwen3-VL-Think: P(yes) logit via Qwen3-VL-8B-Thinking; SG via prompt")
    print(f"   Qwen3-Emb (inst): adds 'Instruct: {QWEN3_INSTRUCTION}'")
    print(f"   SG additive: base_score + λ × graph_asymmetry (0 when SG empty)")
    print(f"   SG prompt: triples injected into prompt text (no effect when SG empty)\n")

    if tag_analysis:
        print(f"  Per-tag breakdown:")
        for s in strats:
            if s not in STRATEGY_LABELS: continue
            label = STRATEGY_LABELS[s][0]
            print(f"\n  ── {label} ──")
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
    p = argparse.ArgumentParser(description="Winoground eval: CLIP, BLIP, LLaVA, Qwen3-VL-Embedding, Qwen3-VL-Thinking (sequential GPU)")
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS + ["all"], default=["all"])
    p.add_argument("--clip_model_id",      default="openai/clip-vit-base-patch32")
    p.add_argument("--blip_model_id",      default=None)
    p.add_argument("--llava_model_id",     default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--qwen3_model_id",     default="Qwen/Qwen3-VL-Embedding-8B",
                   help="Qwen3-VL-Embedding model for contrastive scoring")
    p.add_argument("--qwen3_gen_model_id", default="Qwen/Qwen3-VL-8B-Thinking",
                   help="Qwen3-VL generative model for P(yes) scoring")
    p.add_argument("--spacy_model",        default="en_core_web_sm")
    p.add_argument("--hf_token",           default=None)
    p.add_argument("--max_samples",        type=int, default=None)
    p.add_argument("--split",              default="test")
    p.add_argument("--output_dir",         default="./results_v2")
    p.add_argument("--device",             default="cuda:0",
                   help="GPU device (all models share this sequentially)")
    p.add_argument("--lam",                type=float, default=0.3,
                   help="Weight λ for additive TextSG prior (0 = no SG effect)")
    p.add_argument("--tag_analysis",       action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    methods = ALL_METHODS if "all" in args.methods else args.methods
    log.info(f"Running methods (sequential GPU): {methods}")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── Load dataset + parse scene graphs (CPU, persists across all phases) ──
    sg_parser = TextSceneGraphParser(args.spacy_model)
    sg_scorer = GraphAsymmetryScorer(lam=args.lam)

    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    examples = _prepare_data(dataset, args.split, args.max_samples)
    per_example, sg_cache = _init_per_example(examples, sg_parser)
    n = len(examples)
    log.info(f"Prepared {n} examples with scene graphs")

    # ── Run each method sequentially ──
    all_counts = {}

    if "clip" in methods:
        all_counts.update(run_clip_phase(examples, sg_cache, per_example, sg_scorer, args))
        _log_phase_results(all_counts, n, ["clip", "clip_sg"])

    if "blip" in methods:
        all_counts.update(run_blip_phase(examples, sg_cache, per_example, sg_scorer, args))
        _log_phase_results(all_counts, n, ["blip", "blip_sg"])

    if "llava" in methods:
        all_counts.update(run_llava_phase(examples, sg_cache, per_example, args))
        _log_phase_results(all_counts, n, ["llava", "llava_sg"])

    if "qwen3" in methods:
        all_counts.update(run_qwen3_emb_phase(examples, sg_cache, per_example, sg_scorer, args))
        _log_phase_results(all_counts, n, ["qwen3", "qwen3_sg", "qwen3_inst", "qwen3_inst_sg"])

    if "qwen3_gen" in methods:
        all_counts.update(run_qwen3_gen_phase(examples, sg_cache, per_example, args))
        _log_phase_results(all_counts, n, ["qwen3_gen", "qwen3_gen_sg"])

    # ── Aggregate and report ──
    summary = {s: {k: v / n for k, v in c.items()} for s, c in all_counts.items()}
    summary["n_evaluated"] = n

    strats = list(all_counts.keys())
    tag_analysis = analyze_by_tag(per_example, strats) if args.tag_analysis else None
    print_summary(summary, tag_analysis)

    # ── Save results ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "methods": methods,
        "clip_model_id": args.clip_model_id, "blip_model_id": args.blip_model_id,
        "llava_model_id": args.llava_model_id,
        "qwen3_model_id": args.qwen3_model_id,
        "qwen3_gen_model_id": args.qwen3_gen_model_id,
        "spacy_model": args.spacy_model, "split": args.split,
        "max_samples": args.max_samples, "lam": args.lam,
        "qwen3_instruction": QWEN3_INSTRUCTION,
        "gpu_strategy": "sequential load/unload — one model on GPU at a time",
        "sg_behavior": "additive for CLIP/BLIP/Qwen3-Emb, prompt injection for LLaVA/Qwen3-Gen, "
                        "no effect when SG is empty (graph_prior=0 / plain prompt)",
    }
    for name, data in {"summary": summary, "per_example": per_example,
                       "tags": tag_analysis or {}, "config": config}.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"{name:<12} → {path}")
    log.info("Done.")


def _log_phase_results(all_counts, n, strats):
    for s in strats:
        if s in all_counts:
            c = all_counts[s]
            log.info(f"  {s}: text={c['text']/n:.3f}  image={c['image']/n:.3f}  group={c['group']/n:.3f}")


if __name__ == "__main__":
    main()