"""
Unified Winoground Evaluation: CLIP, BLIP, LLaVA — Embedding + Scene Graph
============================================================================

SCORING DESIGN:
  CLIP   → CLS token from get_image_features() / get_text_features()
           (contrastively trained dual encoder — most principled)
  BLIP   → CLS token (position 0) from image_embeds / last_hidden_state
           (ITC/ITM trained cross-attention encoder)
  LLaVA  → P(yes) from next-token logits after ASSISTANT:
           (most faithful to LLaVA's instruction-tuning objective)

Scene Graph (TextSceneGraphParser from YUKINO-SG):
  Rich spaCy-based parser extracting (subject, relation, object) triples.
  Handles: SVO, prepositional phrases, existential constructions,
           copular verbs, possessives, adjective-aware noun phrases,
           relative clauses, intransitive verbs.

  For CLIP/BLIP: TextSG prior augments cosine score with graph asymmetry.
  For LLaVA:     Scene graph is injected into the prompt ONLY when:
                   (a) the parsed graph is non-empty, AND
                   (b) the asymmetry between caption_0's and caption_1's
                       graphs exceeds a threshold (i.e. the SG is actually
                       capturing the compositional difference, not noise).
                 When the SG is not useful, plain P(yes) prompt is used.

Winoground Metrics (same for all methods):
  Text  : caption_0 scores higher than caption_1 for image_0,
          AND caption_1 scores higher than caption_0 for image_1
  Image : image_0 scores higher than image_1 for caption_0,
          AND image_1 scores higher than image_0 for caption_1
  Group : Text AND Image both correct (strictest)
  Random chance baselines: Text=0.25, Image=0.25, Group=0.0625

Usage:
    # All methods, full eval
    python winoground_eval_v2.py --methods all

    # Quick test on 50 examples, CLIP and BLIP only
    python winoground_eval_v2.py --methods clip blip --max_samples 50

    # LLaVA with scene graph, adjust threshold
    python winoground_eval_v2.py --methods llava --sg_threshold 0.05

    # Tune scene graph prior weight for CLIP/BLIP
    python winoground_eval_v2.py --methods clip blip --lam 0.4

Requirements:
    pip install transformers torch pillow datasets spacy scipy
    python -m spacy download en_core_web_sm
    (or: python -m spacy download en_core_web_trf  for better accuracy)
"""

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
    LlavaProcessor,
    LlavaForConditionalGeneration,
)

try:
    from transformers import BlipForImageTextMatching
    BLIP_HAS_ITM = True
except ImportError:
    from transformers import BlipForConditionalGeneration as BlipForImageTextMatching
    BLIP_HAS_ITM = False

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ALL_METHODS = ["clip", "blip", "llava"]


# ═════════════════════════════════════════════════════════════
# Helper: safely extract a tensor from model outputs
# ═════════════════════════════════════════════════════════════

def _to_tensor(out) -> torch.Tensor:
    """
    Many HuggingFace model methods return either a raw Tensor OR a
    dataclass (BaseModelOutputWithPooling, etc.) depending on the
    transformers version and model config.  This helper guarantees
    we always get a Tensor back.
    """
    if isinstance(out, torch.Tensor):
        return out
    # Try common attribute names in order of preference
    for attr in ("pooler_output", "text_embeds", "image_embeds",
                 "last_hidden_state"):
        if hasattr(out, attr) and getattr(out, attr) is not None:
            val = getattr(out, attr)
            if attr == "last_hidden_state":
                return val[:, 0, :]          # CLS token
            return val
    # Last resort: first element
    if hasattr(out, "__getitem__"):
        return out[0]
    raise TypeError(f"Cannot extract tensor from {type(out)}")


# ═════════════════════════════════════════════════════════════
# SECTION 1: Scene Graph Data Structure
# ═════════════════════════════════════════════════════════════

@dataclass
class Triple:
    subject:  str
    relation: str
    obj:      str

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.obj})"


# ═════════════════════════════════════════════════════════════
# SECTION 2: Rich Text Scene Graph Parser (from YUKINO-SG)
# ═════════════════════════════════════════════════════════════

class TextSceneGraphParser:
    """
    spaCy-based parser extracting (subject, relation, object) triples.
    Handles:
      - SVO (subject-verb-object), including subordinate/relative clauses
      - Prepositional phrases (including participial "sitting on a chair")
      - Existential constructions ("there is a mug in some grass")
      - Copular verbs ("the dog is a puppy", "the dog is brown")
      - Possessives ("the man's hat" → man has hat)
      - Adjective-aware noun phrases ("taller person", "red mug")
      - Compound modifiers ("fire truck")
      - Intransitive verbs ("person sits" → (person, sit, sit))
    """

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
        ADJ_DEPS = {"amod", "advmod"}
        ADJ_POS  = {"ADJ", "NOUN", "VERB"}
        adjs = []
        for child in noun_token.children:
            if child.dep_ in ADJ_DEPS and (
                child.pos_ in ADJ_POS or child.tag_ in ("JJ", "JJR", "JJS")
            ):
                pre = [c.text for c in child.children if c.dep_ == "compound"]
                adjs.extend(pre)
                adjs.append(child.text)
        return adjs

    def _noun_phrase(self, token) -> str:
        noun_token = token
        if token.pos_ not in ("NOUN", "PROPN"):
            for t in token.subtree:
                if t.pos_ in ("NOUN", "PROPN"):
                    noun_token = t
                    break
        head      = noun_token.lemma_
        compounds = [c.text for c in noun_token.children if c.dep_ == "compound"]
        adjs      = self._get_adjectives(noun_token)
        if not adjs:
            for t in token.subtree:
                if t == noun_token:
                    break
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
        SUB_VERB_DEPS = {"relcl", "acl", "advcl", "xcomp", "ccomp", "conj"}
        for token in doc:
            is_root       = (token.dep_ == "ROOT")
            has_subj_child = any(w.dep_ in ("nsubj", "nsubjpass") for w in token.children)
            is_subverb    = (token.dep_ in SUB_VERB_DEPS and
                             (token.pos_ in ("VERB", "AUX") or has_subj_child))
            if not is_root and not is_subverb:
                continue
            if is_root and token.lemma_ == "be":
                continue

            subjs = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]

            # Subject relative clause: head noun is the subject
            if not subjs and token.dep_ in {"relcl", "acl"}:
                if token.head.pos_ in ("NOUN", "PROPN"):
                    subjs = [token.head]

            # Inherit subject from governing verb for advcl/xcomp/ccomp/conj
            if not subjs and token.dep_ in {"advcl", "xcomp", "ccomp", "conj"}:
                if token.head.pos_ in ("VERB", "AUX"):
                    subjs = [w for w in token.head.children if w.dep_ in ("nsubj", "nsubjpass")]

            objs = [w for w in token.children if w.dep_ in ("dobj", "attr", "oprd")]

            # Object relative clause
            if not objs and token.dep_ in {"relcl", "acl"} and subjs:
                if (subjs[0] is not token.head) and token.head.pos_ in ("NOUN", "PROPN"):
                    objs = [token.head]

            acomps = [w for w in token.children if w.dep_ == "acomp"]
            negs   = [w for w in token.children if w.dep_ == "neg"]
            lemma  = ("not " + token.lemma_) if negs else token.lemma_

            for s in subjs:
                for o in objs:
                    triples.append(Triple(self._noun_phrase(s), lemma, self._noun_phrase(o)))
                for a in acomps:
                    triples.append(Triple(self._noun_phrase(s), "is", a.lemma_))
                # Intransitive: encode (subject, verb, verb) for who-does-what signal
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
                        triples.append(Triple(
                            self._noun_phrase(token.head),
                            self._compound_prep(token),
                            self._noun_phrase(pobj),
                        ))
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
                                            triples.append(Triple(
                                                self._noun_phrase(subj),
                                                self._compound_prep(prep),
                                                self._noun_phrase(pobj),
                                            ))
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
                        self._noun_phrase(a) if a.pos_ in ("NOUN", "PROPN") else a.lemma_
                    )
                    triples.append(Triple(self._noun_phrase(s), "is", obj_str))
        return triples

    def _extract_possessive(self, doc) -> list[Triple]:
        triples = []
        for token in doc:
            if token.dep_ == "poss" and token.head.pos_ in ("NOUN", "PROPN"):
                triples.append(Triple(
                    self._noun_phrase(token),
                    "has",
                    self._noun_phrase(token.head),
                ))
        return triples


# ═════════════════════════════════════════════════════════════
# SECTION 3: Graph Asymmetry Scorer
# ═════════════════════════════════════════════════════════════

class GraphAsymmetryScorer:
    """
    Computes how asymmetric two scene graphs are using Hungarian matching.

    Used for two purposes:
      1. CLIP/BLIP TextSG prior: augment base score with graph asymmetry
         between caption and the other caption.
      2. LLaVA gating: decide whether the SG is informative enough to inject
         into the prompt. If asymmetry < threshold, the SG is not capturing
         the compositional difference and injection would be noise.

    Asymmetry formula for a matched triple pair (ta, tb):
        forward = α·sim(ta.subj, tb.subj) + β·sim(ta.rel, tb.rel) + γ·sim(ta.obj, tb.obj)
        flipped = α·sim(ta.subj, tb.obj)  + β·sim(ta.rel, tb.rel) + γ·sim(ta.obj, tb.subj)
        asymmetry = forward - flipped

    A high positive asymmetry means the captions have the same entities in
    different roles — exactly the Winoground pattern.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta:  float = 1.5,   # relation upweighted
        gamma: float = 1.0,
        lam:   float = 0.3,   # weight for CLIP/BLIP augmentation
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.lam   = lam
        self._text_sim_cache: dict[tuple[str, str], float] = {}

    def _text_sim(self, a: str, b: str, encoder) -> float:
        key = (a, b)
        if key not in self._text_sim_cache:
            ea = encoder.embed_text(a)
            eb = encoder.embed_text(b)
            self._text_sim_cache[key] = float(np.dot(ea, eb))
            self._text_sim_cache[(b, a)] = self._text_sim_cache[key]
        return self._text_sim_cache[key]

    def _pair_asymmetry(self, ta: Triple, tb: Triple, encoder) -> float:
        forward = (
            self.alpha * self._text_sim(ta.subject,  tb.subject,  encoder) +
            self.beta  * self._text_sim(ta.relation, tb.relation, encoder) +
            self.gamma * self._text_sim(ta.obj,      tb.obj,      encoder)
        )
        flipped = (
            self.alpha * self._text_sim(ta.subject,  tb.obj,      encoder) +
            self.beta  * self._text_sim(ta.relation, tb.relation, encoder) +
            self.gamma * self._text_sim(ta.obj,      tb.subject,  encoder)
        )
        return forward - flipped

    def graph_prior(
        self, triples_a: list[Triple], triples_b: list[Triple], encoder
    ) -> float:
        """
        Hungarian-matched asymmetry between two triple sets.
        Returns 0.0 if either set is empty.
        """
        if not triples_a or not triples_b:
            return 0.0
        n, m = len(triples_a), len(triples_b)
        cost = np.zeros((n, m))
        for i, ta in enumerate(triples_a):
            for j, tb in enumerate(triples_b):
                cost[i, j] = self._pair_asymmetry(ta, tb, encoder)
        row_ind, col_ind = linear_sum_assignment(-np.abs(cost))
        return float(cost[row_ind, col_ind].mean())

    def augment(
        self,
        base_score: float,
        triples_cap: list[Triple],
        triples_other: list[Triple],
        encoder,
    ) -> float:
        """Add λ · graph_prior to the base score."""
        return base_score + self.lam * self.graph_prior(triples_cap, triples_other, encoder)

    def is_useful(
        self,
        triples_cap0: list[Triple],
        triples_cap1: list[Triple],
        encoder,
        threshold: float,
    ) -> bool:
        """
        Returns True if the scene graphs for cap0 and cap1 are asymmetric enough
        to be worth injecting into the LLaVA prompt.

        Logic:
          - If either triple set is empty → not useful (no structure to inject)
          - If |graph_prior(cap0, cap1)| >= threshold → useful
          - Otherwise → not useful (SG doesn't distinguish the two captions)
        """
        if not triples_cap0 or not triples_cap1:
            return False
        asymmetry = abs(self.graph_prior(triples_cap0, triples_cap1, encoder))
        return asymmetry >= threshold


# ═════════════════════════════════════════════════════════════
# SECTION 4: Simple CLIP text encoder for graph scoring
# ═════════════════════════════════════════════════════════════

class CLIPTextEncoder:
    """
    Lightweight wrapper exposing only embed_text() for use by
    GraphAsymmetryScorer. Caches all embeddings.
    """

    def __init__(self, clip_model, clip_processor, clip_device):
        self.model     = clip_model
        self.processor = clip_processor
        self.device    = clip_device
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True
        ).to(self.device)
        out = self.model.get_text_features(**inputs)
        emb = _to_tensor(out)
        emb = F.normalize(emb.float(), dim=-1).cpu().numpy()
        # Handle both (D,) and (1, D) shapes
        if emb.ndim == 2:
            emb = emb[0]
        self._cache[text] = emb
        return emb


class BLIPTextEncoder:
    """
    Lightweight wrapper exposing only embed_text() via BLIP's text encoder CLS token.
    Used when BLIP is available and we want graph scoring in BLIP's text space.
    """

    def __init__(self, blip_model, blip_processor, blip_device):
        self.model     = blip_model
        self.processor = blip_processor
        self.device    = blip_device
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        if text in self._cache:
            return self._cache[text]
        dummy = Image.new("RGB", (1, 1), color=(255, 255, 255))
        inputs = self.processor(
            images=dummy, text=text, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        # Extract CLS token from last_hidden_state
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            cls_emb = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            cls_emb = outputs.text_embeds
        else:
            cls_emb = _to_tensor(outputs)
        emb = F.normalize(cls_emb.float(), dim=-1).cpu().numpy()
        if emb.ndim == 2:
            emb = emb[0]
        self._cache[text] = emb
        return emb


# ═════════════════════════════════════════════════════════════
# SECTION 5: Model Loading
# ═════════════════════════════════════════════════════════════

def load_clip(model_id="openai/clip-vit-large-patch14"):
    log.info(f"Loading CLIP: {model_id}")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    log.info("CLIP ready.")
    return model, processor, device


def load_blip(model_id=None):
    if model_id is None:
        model_id = ("Salesforce/blip-itm-base-coco" if BLIP_HAS_ITM
                    else "Salesforce/blip-image-captioning-base")
    log.info(f"Loading BLIP: {model_id}  (ITM head available: {BLIP_HAS_ITM})")
    processor = BlipProcessor.from_pretrained(model_id)
    model     = BlipForImageTextMatching.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device).eval()
    log.info("BLIP ready.")
    return model, processor, device


def load_llava(model_id="llava-hf/llava-1.5-7b-hf", device="cuda:0"):
    log.info(f"Loading LLaVA: {model_id}")
    processor = LlavaProcessor.from_pretrained(model_id)
    model     = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("LLaVA ready.")
    return model, processor


def load_models(methods: list[str], args) -> dict:
    models = {}
    if "clip" in methods:
        m, p, d = load_clip(args.clip_model_id)
        models.update({"clip": m, "clip_processor": p, "clip_device": d})
    if "blip" in methods:
        m, p, d = load_blip(args.blip_model_id)
        models.update({"blip": m, "blip_processor": p, "blip_device": d})
    if "llava" in methods:
        m, p = load_llava(args.llava_model_id, device=args.device)
        models.update({"llava": m, "llava_processor": p})
    return models


# ═════════════════════════════════════════════════════════════
# SECTION 6: Scoring Primitives
# ═════════════════════════════════════════════════════════════

# ── CLIP: joint forward pass for guaranteed same-space embeddings ──

@torch.no_grad()
def score_clip_base(models, image: Image.Image, caption: str) -> float:
    """
    Cosine similarity between CLIP image and text embeddings → [0, 1].
    Uses a single joint forward pass so both embeddings are guaranteed
    to go through their respective projection heads into the same space.
    """
    clip_model = models["clip"]
    clip_proc  = models["clip_processor"]
    clip_dev   = models["clip_device"]

    # Process image and text separately, then merge inputs
    img_inputs = clip_proc(images=image, return_tensors="pt")
    txt_inputs = clip_proc(text=[caption], return_tensors="pt", padding=True)

    # Combine into a single input dict
    combined = {}
    for k, v in img_inputs.items():
        combined[k] = v.to(clip_dev)
    for k, v in txt_inputs.items():
        combined[k] = v.to(clip_dev)

    outputs = clip_model(**combined)

    # image_embeds and text_embeds are the projected embeddings
    img_emb = F.normalize(outputs.image_embeds.float(), dim=-1)[0]
    txt_emb = F.normalize(outputs.text_embeds.float(), dim=-1)[0]

    return ((img_emb * txt_emb).sum().item() + 1) / 2


# ── BLIP: CLS token cosine similarity ───────────────────────

@torch.no_grad()
def _blip_image_embedding(blip_model, blip_processor, blip_device,
                           image: Image.Image) -> torch.Tensor:
    """
    CLS token (position 0) from BLIP's image encoder output.
    """
    # BLIP ITM models need text too; use empty string for image-only embedding
    inputs  = blip_processor(images=image, text="", return_tensors="pt").to(blip_device)
    outputs = blip_model(**inputs)
    # Try image_embeds first, then fall back to vision_model output
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        cls_emb = outputs.image_embeds
        if cls_emb.ndim == 3:
            cls_emb = cls_emb[:, 0, :]
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        cls_emb = outputs.last_hidden_state[:, 0, :]
    else:
        cls_emb = _to_tensor(outputs)
        if cls_emb.ndim == 3:
            cls_emb = cls_emb[:, 0, :]
    return F.normalize(cls_emb[0].float(), dim=0)


@torch.no_grad()
def _blip_text_embedding(blip_model, blip_processor, blip_device,
                          caption: str) -> torch.Tensor:
    """
    CLS token (position 0) from BLIP's text encoder output.
    A dummy 1x1 white image is passed to satisfy BLIP's forward signature.
    """
    dummy   = Image.new("RGB", (1, 1), color=(255, 255, 255))
    inputs  = blip_processor(images=dummy, text=caption,
                              return_tensors="pt").to(blip_device)
    outputs = blip_model(**inputs)
    # Try text-specific outputs first
    if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        cls_emb = outputs.text_embeds
        if cls_emb.ndim == 3:
            cls_emb = cls_emb[:, 0, :]
    elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        cls_emb = outputs.last_hidden_state[:, 0, :]
    else:
        cls_emb = _to_tensor(outputs)
        if cls_emb.ndim == 3:
            cls_emb = cls_emb[:, 0, :]
    return F.normalize(cls_emb[0].float(), dim=0)


def score_blip_base(models, image: Image.Image, caption: str) -> float:
    """Cosine similarity between BLIP CLS image and text embeddings → [0, 1]."""
    img_emb = _blip_image_embedding(
        models["blip"], models["blip_processor"], models["blip_device"], image
    )
    txt_emb = _blip_text_embedding(
        models["blip"], models["blip_processor"], models["blip_device"], caption
    )
    return ((img_emb * txt_emb).sum().item() + 1) / 2


# ── LLaVA: P(yes) from next-token logits ────────────────────

def _get_first_device(model) -> torch.device:
    """Safely get device of first parameter — correct with device_map='auto'."""
    return next(model.parameters()).device


def _format_scene_graph(triples: list[Triple]) -> str:
    return "\n".join([f"  - {t.subject}  [{t.relation}]  {t.obj}" for t in triples])


def _build_llava_prompt(caption: str, triples: list[Triple], use_sg: bool) -> str:
    """
    Build LLaVA yes/no prompt. Injects scene graph only when use_sg=True.
    use_sg=True is only set when the SG passes the informativeness gate.
    """
    if use_sg and triples:
        sg = _format_scene_graph(triples)
        return (
            f"USER: <image>\n"
            f"Caption: '{caption}'\n\n"
            f"The caption has the following scene graph relations:\n{sg}\n\n"
            f"Using the scene graph as a guide, pay close attention to which "
            f"entity is doing what and any spatial relationships. "
            f"Does this image match the caption?\n"
            f"Answer yes or no.\nASSISTANT:"
        )
    else:
        return (
            f"USER: <image>\n"
            f"Does this image match the caption: '{caption}'?\n"
            f"Answer yes or no.\nASSISTANT:"
        )


def _get_yes_prob(llava_model, llava_processor,
                  image: Image.Image, prompt: str) -> float:
    """
    P(yes) from next-token logits after ASSISTANT:.
    Single forward pass — no generation. Returns float in [0, 1].
    """
    inputs       = llava_processor(text=prompt, images=image, return_tensors="pt")
    first_device = _get_first_device(llava_model)
    inputs       = {k: v.to(first_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = llava_model(**inputs).logits

    last   = logits[0, -1]
    yes_id = llava_processor.tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id  = llava_processor.tokenizer.encode("no",  add_special_tokens=False)[0]
    return torch.softmax(
        torch.stack([last[yes_id], last[no_id]]), dim=0
    )[0].item()


def score_llava_base(models, image: Image.Image, caption: str,
                     triples: list[Triple], use_sg: bool) -> float:
    """
    LLaVA P(yes) score with conditional scene graph injection.

    use_sg: pre-determined by the informativeness gate in the evaluation loop.
    If True and triples is non-empty, the scene graph is injected into the prompt.
    If False or triples is empty, plain yes/no prompt is used.
    """
    prompt = _build_llava_prompt(caption, triples, use_sg)
    return _get_yes_prob(
        models["llava"], models["llava_processor"], image, prompt
    )


# ═════════════════════════════════════════════════════════════
# SECTION 7: Winoground Metrics
# ═════════════════════════════════════════════════════════════

def winoground_metrics(s_c0_i0, s_c1_i0, s_c0_i1, s_c1_i1):
    """
    Standard Winoground three-way accuracy.
    s_cX_iY = score(caption_X, image_Y).
    caption_0 is correct for image_0; caption_1 is correct for image_1.

    Text  : correct caption ranks higher for BOTH images
    Image : correct image  ranks higher for BOTH captions
    Group : Text AND Image both correct (strictest)

    Random chance: Text=0.25, Image=0.25, Group=0.0625
    """
    text  = (s_c0_i0 > s_c1_i0) and (s_c1_i1 > s_c0_i1)
    image = (s_c0_i0 > s_c0_i1) and (s_c1_i1 > s_c1_i0)
    group = text and image
    return text, image, group


# ═════════════════════════════════════════════════════════════
# SECTION 8: Evaluation Loop
# ═════════════════════════════════════════════════════════════

def evaluate(
    models:       dict,
    sg_parser:    TextSceneGraphParser,
    sg_scorer:    GraphAsymmetryScorer,
    dataset,
    methods:      list[str],
    split:        str   = "test",
    max_samples:  Optional[int] = None,
    lam:          float = 0.3,
    sg_threshold: float = 0.05,
):
    data = dataset[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    # Build text encoders for graph scoring
    clip_text_enc = (
        CLIPTextEncoder(models["clip"], models["clip_processor"], models["clip_device"])
        if "clip" in methods else None
    )
    blip_text_enc = (
        BLIPTextEncoder(models["blip"], models["blip_processor"], models["blip_device"])
        if "blip" in methods else None
    )

    # Strategy names per method
    strategy_names = []
    if "clip" in methods:
        strategy_names += ["clip", "clip_sg"]
    if "blip" in methods:
        strategy_names += ["blip", "blip_sg"]
    if "llava" in methods:
        strategy_names += ["llava_plain", "llava_sg"]

    counts      = {s: {"text": 0, "image": 0, "group": 0} for s in strategy_names}
    per_example = []

    for idx, example in enumerate(tqdm(data, desc="Evaluating")):
        img0 = example["image_0"].convert("RGB")
        img1 = example["image_1"].convert("RGB")
        cap0 = example["caption_0"]
        cap1 = example["caption_1"]
        tag  = example.get("tag", "")

        # ── Parse scene graphs once per example ─────────────────────────
        t0 = sg_parser.parse(cap0)
        t1 = sg_parser.parse(cap1)

        row = {
            "idx": idx, "caption_0": cap0, "caption_1": cap1, "tag": tag,
            "sg_cap0": [repr(t) for t in t0],
            "sg_cap1": [repr(t) for t in t1],
        }

        # ── CLIP ────────────────────────────────────────────────────────
        if "clip" in methods:
            s_c0_i0 = score_clip_base(models, img0, cap0)
            s_c1_i0 = score_clip_base(models, img0, cap1)
            s_c0_i1 = score_clip_base(models, img1, cap0)
            s_c1_i1 = score_clip_base(models, img1, cap1)

            # Plain CLIP
            text_c, image_c, group_c = winoground_metrics(s_c0_i0, s_c1_i0, s_c0_i1, s_c1_i1)
            counts["clip"]["text"]  += int(text_c)
            counts["clip"]["image"] += int(image_c)
            counts["clip"]["group"] += int(group_c)
            row["clip"] = _make_row(s_c0_i0, s_c1_i0, s_c0_i1, s_c1_i1, text_c, image_c, group_c)

            # CLIP + TextSG prior (augment with graph asymmetry in CLIP text space)
            sg_scorer.lam = lam
            sg_c0_i0 = sg_scorer.augment(s_c0_i0, t0, t1, clip_text_enc)
            sg_c1_i0 = sg_scorer.augment(s_c1_i0, t1, t0, clip_text_enc)
            sg_c0_i1 = sg_scorer.augment(s_c0_i1, t0, t1, clip_text_enc)
            sg_c1_i1 = sg_scorer.augment(s_c1_i1, t1, t0, clip_text_enc)

            text_c, image_c, group_c = winoground_metrics(sg_c0_i0, sg_c1_i0, sg_c0_i1, sg_c1_i1)
            counts["clip_sg"]["text"]  += int(text_c)
            counts["clip_sg"]["image"] += int(image_c)
            counts["clip_sg"]["group"] += int(group_c)
            row["clip_sg"] = _make_row(sg_c0_i0, sg_c1_i0, sg_c0_i1, sg_c1_i1, text_c, image_c, group_c)

        # ── BLIP ────────────────────────────────────────────────────────
        if "blip" in methods:
            b_c0_i0 = score_blip_base(models, img0, cap0)
            b_c1_i0 = score_blip_base(models, img0, cap1)
            b_c0_i1 = score_blip_base(models, img1, cap0)
            b_c1_i1 = score_blip_base(models, img1, cap1)

            # Plain BLIP
            text_c, image_c, group_c = winoground_metrics(b_c0_i0, b_c1_i0, b_c0_i1, b_c1_i1)
            counts["blip"]["text"]  += int(text_c)
            counts["blip"]["image"] += int(image_c)
            counts["blip"]["group"] += int(group_c)
            row["blip"] = _make_row(b_c0_i0, b_c1_i0, b_c0_i1, b_c1_i1, text_c, image_c, group_c)

            # BLIP + TextSG prior (graph asymmetry in BLIP text space)
            sg_scorer.lam = lam
            enc = blip_text_enc if blip_text_enc is not None else clip_text_enc
            sg_b_c0_i0 = sg_scorer.augment(b_c0_i0, t0, t1, enc)
            sg_b_c1_i0 = sg_scorer.augment(b_c1_i0, t1, t0, enc)
            sg_b_c0_i1 = sg_scorer.augment(b_c0_i1, t0, t1, enc)
            sg_b_c1_i1 = sg_scorer.augment(b_c1_i1, t1, t0, enc)

            text_c, image_c, group_c = winoground_metrics(sg_b_c0_i0, sg_b_c1_i0, sg_b_c0_i1, sg_b_c1_i1)
            counts["blip_sg"]["text"]  += int(text_c)
            counts["blip_sg"]["image"] += int(image_c)
            counts["blip_sg"]["group"] += int(group_c)
            row["blip_sg"] = _make_row(sg_b_c0_i0, sg_b_c1_i0, sg_b_c0_i1, sg_b_c1_i1, text_c, image_c, group_c)

        # ── LLaVA ───────────────────────────────────────────────────────
        if "llava" in methods:
            # Gate: is the SG informative enough to inject?
            # Uses CLIP text encoder for asymmetry computation.
            # Falls back to plain prompt if CLIP encoder not loaded.
            gate_enc = clip_text_enc if clip_text_enc is not None else None
            sg_useful = (
                sg_scorer.is_useful(t0, t1, gate_enc, threshold=sg_threshold)
                if gate_enc is not None
                else False
            )
            row["llava_sg_injected"] = sg_useful
            row["llava_sg_cap0"]     = [repr(t) for t in t0] if sg_useful else []
            row["llava_sg_cap1"]     = [repr(t) for t in t1] if sg_useful else []

            # LLaVA plain (never inject SG)
            lp_c0_i0 = score_llava_base(models, img0, cap0, t0, use_sg=False)
            lp_c1_i0 = score_llava_base(models, img0, cap1, t1, use_sg=False)
            lp_c0_i1 = score_llava_base(models, img1, cap0, t0, use_sg=False)
            lp_c1_i1 = score_llava_base(models, img1, cap1, t1, use_sg=False)

            text_c, image_c, group_c = winoground_metrics(lp_c0_i0, lp_c1_i0, lp_c0_i1, lp_c1_i1)
            counts["llava_plain"]["text"]  += int(text_c)
            counts["llava_plain"]["image"] += int(image_c)
            counts["llava_plain"]["group"] += int(group_c)
            row["llava_plain"] = _make_row(lp_c0_i0, lp_c1_i0, lp_c0_i1, lp_c1_i1, text_c, image_c, group_c)

            # LLaVA + SG (inject only when gate passes)
            ls_c0_i0 = score_llava_base(models, img0, cap0, t0, use_sg=sg_useful)
            ls_c1_i0 = score_llava_base(models, img0, cap1, t1, use_sg=sg_useful)
            ls_c0_i1 = score_llava_base(models, img1, cap0, t0, use_sg=sg_useful)
            ls_c1_i1 = score_llava_base(models, img1, cap1, t1, use_sg=sg_useful)

            text_c, image_c, group_c = winoground_metrics(ls_c0_i0, ls_c1_i0, ls_c0_i1, ls_c1_i1)
            counts["llava_sg"]["text"]  += int(text_c)
            counts["llava_sg"]["image"] += int(image_c)
            counts["llava_sg"]["group"] += int(group_c)
            row["llava_sg"] = _make_row(ls_c0_i0, ls_c1_i0, ls_c0_i1, ls_c1_i1, text_c, image_c, group_c)

        per_example.append(row)

        if (idx + 1) % 10 == 0:
            n = idx + 1
            for s in strategy_names:
                log.info(
                    f"[{s}] n={n} | "
                    f"text={counts[s]['text']/n:.3f} | "
                    f"image={counts[s]['image']/n:.3f} | "
                    f"group={counts[s]['group']/n:.3f}"
                )

    n       = len(data)
    summary = {s: {k: v / n for k, v in c.items()} for s, c in counts.items()}
    summary["n_evaluated"] = n
    return summary, per_example


def _make_row(s_c0_i0, s_c1_i0, s_c0_i1, s_c1_i1, text_c, image_c, group_c) -> dict:
    return {
        "scores":  {
            "c0_i0": round(s_c0_i0, 5), "c1_i0": round(s_c1_i0, 5),
            "c0_i1": round(s_c0_i1, 5), "c1_i1": round(s_c1_i1, 5),
        },
        "correct": {"text": text_c, "image": image_c, "group": group_c},
    }


# ═════════════════════════════════════════════════════════════
# SECTION 9: Tag-level Analysis
# ═════════════════════════════════════════════════════════════

def analyze_by_tag(per_example: list[dict], strategy_names: list[str]) -> dict:
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
# SECTION 10: Reporting
# ═════════════════════════════════════════════════════════════

STRATEGY_LABELS = {
    "clip":        ("CLIP ViT-L/14",              "CLS cosine sim",       "no SG"),
    "clip_sg":     ("CLIP + TextSG prior",         "CLS cosine + SG aug",  "always"),
    "blip":        ("BLIP (CLS cosine)",           "CLS cosine sim",       "no SG"),
    "blip_sg":     ("BLIP + TextSG prior",         "CLS cosine + SG aug",  "always"),
    "llava_plain": ("LLaVA 1.5-7b (plain)",        "P(yes) logit",         "never"),
    "llava_sg":    ("LLaVA 1.5-7b + SG (gated)",  "P(yes) logit + SG",    "gated"),
}


def print_summary(summary: dict, tag_analysis: Optional[dict] = None,
                  sg_threshold: float = 0.05):
    n         = summary.get("n_evaluated", "?")
    strats    = [k for k in summary if k != "n_evaluated"]
    W, M, G   = 32, 18, 8

    print(f"\n{'═' * 82}")
    print(f"  Winoground Results  (n={n})")
    print(f"  Scene graph: YUKINO-SG TextSceneGraphParser")
    print(f"  LLaVA SG gate threshold: {sg_threshold}")
    print(f"{'═' * 82}")
    print(f"  {'Strategy':<{W}}  {'Mechanism':<{M}}  {'SG':>5}  "
          f"{'Text':>{G}}  {'Image':>{G}}  {'Group':>{G}}")
    print(f"  {'-' * 78}")
    print(f"  {'Random chance':<{W}}  {'—':<{M}}  {'—':>5}  "
          f"{'0.250':>{G}}  {'0.250':>{G}}  {'0.063':>{G}}")

    for s in strats:
        if s not in STRATEGY_LABELS:
            continue
        label, mech, sg_col = STRATEGY_LABELS[s]
        v = summary[s]
        print(f"  {label:<{W}}  {mech:<{M}}  {sg_col:>5}  "
              f"{v['text']:>{G}.4f}  {v['image']:>{G}.4f}  {v['group']:>{G}.4f}")

    print(f"{'═' * 82}")
    print(f"\n  Notes:")
    print(f"   - CLIP/BLIP: directly comparable (both use CLS embedding cosine similarity)")
    print(f"   - LLaVA: P(yes) logit — not directly comparable to CLIP/BLIP cosine scores")
    print(f"   - LLaVA SG gate: SG injected only when |graph_asymmetry| >= {sg_threshold}")
    print(f"   - SG parser: YUKINO-SG TextSceneGraphParser (spaCy, rich extraction)\n")

    if tag_analysis:
        print(f"  Per-tag breakdown:")
        for s in strats:
            if s not in STRATEGY_LABELS:
                continue
            label = STRATEGY_LABELS[s][0]
            print(f"\n  ── {label} ──")
            print(f"  {'Tag':<30}  {'n':>4}  {'Text':>8}  {'Image':>8}  {'Group':>8}")
            print(f"  {'-' * 62}")
            for tag, data in sorted(tag_analysis.items()):
                if s not in data:
                    continue
                d = data[s]
                print(f"  {tag:<30}  {d['n']:>4}  "
                      f"{d['text']:>8.3f}  {d['image']:>8.3f}  {d['group']:>8.3f}")
    print()


# ═════════════════════════════════════════════════════════════
# SECTION 11: CLI
# ═════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Winoground evaluation: CLIP (CLS), BLIP (CLS), LLaVA (P(yes)).\n"
            "Scene graphs via YUKINO-SG TextSceneGraphParser.\n"
            "LLaVA SG injection is gated by graph asymmetry informativeness."
        )
    )
    parser.add_argument(
        "--methods", nargs="+",
        choices=ALL_METHODS + ["all"], default=["all"],
    )
    parser.add_argument("--clip_model_id",  type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--blip_model_id",  type=str, default=None)
    parser.add_argument("--llava_model_id", type=str,
                        default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--spacy_model",    type=str, default="en_core_web_sm",
                        help="spaCy model name (en_core_web_sm or en_core_web_trf)")
    parser.add_argument("--hf_token",       type=str, default=None)
    parser.add_argument("--max_samples",    type=int, default=None)
    parser.add_argument("--split",          type=str, default="test")
    parser.add_argument("--output_dir",     type=str, default="./results_v2")
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument(
        "--lam", type=float, default=0.3,
        help="Weight λ for TextSG prior augmentation on CLIP/BLIP scores"
    )
    parser.add_argument(
        "--sg_threshold", type=float, default=0.05,
        help=(
            "Graph asymmetry threshold for LLaVA SG injection gate. "
            "If |asymmetry(cap0_SG, cap1_SG)| < threshold, SG is not injected. "
            "Lower = inject more often; higher = inject only when clearly useful."
        )
    )
    parser.add_argument("--tag_analysis", action="store_true", default=True)
    return parser.parse_args()


def main():
    args    = parse_args()
    methods = ALL_METHODS if "all" in args.methods else args.methods
    log.info(f"Running methods: {methods}")

    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)

    # ── Scene graph components ───────────────────────────────
    sg_parser = TextSceneGraphParser(args.spacy_model)
    sg_scorer = GraphAsymmetryScorer(lam=args.lam)

    # ── Models ──────────────────────────────────────────────
    models = load_models(methods, args)

    # ── Dataset ─────────────────────────────────────────────
    log.info("Loading Winoground ...")
    dataset = load_dataset("facebook/winoground", trust_remote_code=True)
    log.info(f"Split '{args.split}': {len(dataset[args.split])} examples")

    # ── Evaluate ────────────────────────────────────────────
    summary, per_example = evaluate(
        models=models,
        sg_parser=sg_parser,
        sg_scorer=sg_scorer,
        dataset=dataset,
        methods=methods,
        split=args.split,
        max_samples=args.max_samples,
        lam=args.lam,
        sg_threshold=args.sg_threshold,
    )

    # ── Strategy names for tag analysis ─────────────────────
    strategy_names = []
    if "clip"  in methods: strategy_names += ["clip", "clip_sg"]
    if "blip"  in methods: strategy_names += ["blip", "blip_sg"]
    if "llava" in methods: strategy_names += ["llava_plain", "llava_sg"]

    tag_analysis = analyze_by_tag(per_example, strategy_names) if args.tag_analysis else None
    print_summary(summary, tag_analysis, sg_threshold=args.sg_threshold)

    # ── Save outputs ─────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "methods":        methods,
        "clip_model_id":  args.clip_model_id,
        "blip_model_id":  args.blip_model_id,
        "llava_model_id": args.llava_model_id,
        "spacy_model":    args.spacy_model,
        "split":          args.split,
        "max_samples":    args.max_samples,
        "lam":            args.lam,
        "sg_threshold":   args.sg_threshold,
        "scoring": {
            "clip":        "CLS token cosine similarity (get_image_features / get_text_features)",
            "blip":        "CLS token cosine similarity (image_embeds[:,0] / last_hidden_state[:,0])",
            "llava_plain": "P(yes) next-token logit, no scene graph",
            "llava_sg":    f"P(yes) next-token logit, SG injected when |asymmetry| >= {args.sg_threshold}",
        },
        "sg_parser": "YUKINO-SG TextSceneGraphParser (SVO, prep, existential, copular, possessive)",
    }

    saves = {
        "summary":     summary,
        "per_example": per_example,
        "tags":        tag_analysis or {},
        "config":      config,
    }
    for name, data in saves.items():
        path = out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"{name:<12} → {path}")

    log.info("Done.")


if __name__ == "__main__":
    main()