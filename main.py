# explanation: Import typing helpers for type hints (Optional, Tuple, Dict)
from typing import Optional, Tuple, Dict

# explanation: FastAPI framework primitives
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# explanation: Pydantic models for request/response schemas
from pydantic import BaseModel

# explanation: Standard libs (env, regex, json) and PyTorch for model device
import os, re, json
import torch

# explanation: Hugging Face Transformers — auto and Marian-specific classes
from transformers import (
    AutoTokenizer,             # explanation: works for many model families (mBART/T5/etc.)
    AutoModelForSeq2SeqLM,     # explanation: generic seq2seq model loader
    MarianMTModel,             # explanation: specific to Marian MT models
    MarianTokenizer,           # explanation: specific tokenizer for Marian models
    AutoConfig,                # explanation: lets us sniff the model type to choose the right classes
)

# explanation: Give the API a friendly title visible in OpenAPI docs
APP_TITLE = "Offline Smart Translator (Auto-Detect + Grammar + Slang Normalizer)"

# explanation: Create the FastAPI app instance
app = FastAPI(title=APP_TITLE)

# explanation: Read environment variables for configuration (with defaults)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")  # explanation: CORS allowed origins (comma-separated)
ENABLE_GRAMMAR  = os.getenv("ENABLE_GRAMMAR", "1") == "1"       # explanation: toggle English grammar correction
ENABLE_SLANG    = os.getenv("ENABLE_SLANG", "1") == "1"         # explanation: toggle slang/abbrev normalization
REPLACE_PLAIN_CAUSE = os.getenv("REPLACE_PLAIN_CAUSE", "0") == "1"  # explanation: risky: map "cause"→"because"
CACHE_DIR       = os.getenv("HF_CACHE_DIR", "./cache")          # explanation: where HF models are cached
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu" # explanation: use GPU if available else CPU
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "256"))    # explanation: safe max length to tokenize inputs

# explanation: Add CORS middleware so browsers/apps from other domains can call this API
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all external domains
    allow_origin_regex=".*",      # allow PWA / Appilix (Origin=null)
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,      # IMPORTANT: must be False when allow_origins=["*"]
)

# explanation: Map (source_lang, target_lang) to a specific pretrained translation model
# explanation: We include Marian pairs and an mBART (English→Telugu) example
MODEL_MAP: Dict[Tuple[str, str], str] = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-ta",
    ("en", "te"): "Meher2006/english-to-telugu-model",   # explanation: mBART family; loader handles it
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
    ("te", "en"): "Helsinki-NLP/opus-mt-te-en",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
}

# explanation: Grammar correction model for English (T5-based)
GRAMMAR_MODEL_NAME = "vennify/t5-base-grammar-correction"

# explanation: Lazy globals for grammar model/tokenizer
grammar_tokenizer = None
grammar_model = None

# explanation: Try loading grammar corrector if enabled; if it fails, continue without it
if ENABLE_GRAMMAR:
    try:
        grammar_tokenizer = AutoTokenizer.from_pretrained(GRAMMAR_MODEL_NAME, cache_dir=CACHE_DIR)
        grammar_model = AutoModelForSeq2SeqLM.from_pretrained(GRAMMAR_MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
    except Exception as e:
        print(f"[WARN] Grammar model not loaded: {e}")
        ENABLE_GRAMMAR = False

# explanation: Default slang/abbreviation replacements (regex → replacement)
# explanation: We keep it conservative and English-focused to avoid hurting other languages
DEFAULT_SLANG_MAP = {
    r"\bbtw\b": "by the way",
    r"\bidk\b": "I don't know",
    r"\bimo\b": "in my opinion",
    r"\bimho\b": "in my humble opinion",
    r"\btbh\b": "to be honest",
    r"\basap\b": "as soon as possible",
    r"\bbrb\b": "be right back",
    r"\bomw\b": "on my way",
    r"\bafaik\b": "as far as I know",
    r"\blmk\b": "let me know",
    r"\bnp\b": "no problem",
    r"\btho\b": "though",
    r"\bthx\b": "thanks",
    r"\btnx\b": "thanks",
    r"\bty\b": "thanks",
    r"\btysm\b": "thank you so much",
    r"\bgonna\b": "going to",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "have to",
    r"\bkinda\b": "kind of",
    r"\bsorta\b": "sort of",
    r"\bcoz\b": "because",
    r"\bcuz\b": "because",
    r"\bbc\b": "because",
    r"\bbcz\b": "because",
    r"\b'cause\b": "because",
    # explanation: plain "cause" → "because" is optional (can be wrong for noun/verb use)
}

# explanation: Optionally add plain "cause" mapping if user explicitly enables it
if REPLACE_PLAIN_CAUSE:
    DEFAULT_SLANG_MAP[r"\bcause\b"] = "because"

# explanation: Helper to preserve input word casing in replacements (UPPER/Capitalized/lower)
def _preserve_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src[:1].isupper() and src[1:].islower():
        return repl.capitalize()
    return repl

# explanation: Load user-provided slang overrides from slang_custom.json (if present)
def _load_custom_slang(path="slang_custom.json") -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}

# explanation: Compile regex patterns once at startup for performance
compiled_slang_patterns = []
if ENABLE_SLANG:
    merged = dict(DEFAULT_SLANG_MAP)          # explanation: start with defaults
    merged.update(_load_custom_slang())       # explanation: allow user overrides/additions
    compiled_slang_patterns = [
        (re.compile(pattern, flags=re.IGNORECASE), replacement)
        for pattern, replacement in merged.items()
    ]

# explanation: Apply all slang/abbrev replacements to input text
def normalize_slang(text: str) -> str:
    if not ENABLE_SLANG or not text or not compiled_slang_patterns:
        return text
    out = text
    for rx, replacement in compiled_slang_patterns:
        out = rx.sub(lambda m: _preserve_case(m.group(0), replacement), out)
    return out

# explanation: Cache of loaded translation models/tokenizers by language pair (avoid reloading)
translation_cache: Dict[Tuple[str, str], Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}

# explanation: Pydantic request schema for /translate
class TranslationRequest(BaseModel):
    text: str                   # explanation: text to translate
    target_lang: str            # explanation: target language code (e.g., "hi", "te")
    source_lang: Optional[str] = None  # explanation: optional source lang; auto-detect if None

# explanation: Pydantic response schema for /translate
class TranslationResponse(BaseModel):
    detected_source_lang: str   # explanation: auto-detected (or provided) source language
    target_language: str        # explanation: target language code
    original_text: str          # explanation: original input from user
    normalized_text: Optional[str] = None  # explanation: after slang normalization (if changed)
    corrected_text: Optional[str] = None   # explanation: after grammar correction (English only, if changed)
    translated_text: str        # explanation: final translated output
    note: Optional[str] = None  # explanation: extra info (e.g., "translated via English pivot")

# explanation: Simple language detector wrapper (we limit allowed tags to our supported set)
def detect_language(text: str) -> str:
    from langdetect import detect
    try:
        raw = detect(text)
    except Exception:
        return "en"
    allowed = {"en", "hi", "ta", "te", "fr", "de", "es", "it"}
    return raw if raw in allowed else "en"

# explanation: Grammar correction only for English using T5 model
def correct_grammar_if_english(text: str, lang: str) -> str:
    if not ENABLE_GRAMMAR or lang != "en" or not text.strip():
        return text
    inputs = grammar_tokenizer(
        "grammar: " + text,               # explanation: T5 prompt format the model expects
        return_tensors="pt",
        truncation=True,                  # explanation: truncate to avoid overly long inputs
        padding=True,                     # explanation: pad batch to same length
        max_length=MAX_INPUT_TOKENS,      # explanation: cap tokenized input length
    ).to(DEVICE)
    with torch.no_grad():                 # explanation: inference only (no gradients)
        outputs = grammar_model.generate(**inputs, max_length=256, num_beams=5)
    return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# explanation: Load a translation model the right way based on its "model_type" (Marian vs others)
def get_translation_model(lang_pair: Tuple[str, str]):
    if lang_pair not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported translation pair {lang_pair}")
    if lang_pair in translation_cache:
        return translation_cache[lang_pair]

    model_name = MODEL_MAP[lang_pair]
    print(f"[LOAD] {lang_pair} -> {model_name}")

    # explanation: Read config to decide which tokenizer/model classes to use
    try:
        cfg = AutoConfig.from_pretrained(model_name, cache_dir=CACHE_DIR)
        model_type = getattr(cfg, "model_type", "").lower()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config for {model_name}: {e}")

    # explanation: If Marian, use Marian classes; otherwise, use Auto classes (mBART/T5/etc.)
    try:
        if model_type == "marian":
            tok = MarianTokenizer.from_pretrained(
                model_name, cache_dir=CACHE_DIR, model_max_length=MAX_INPUT_TOKENS
            )
            mdl = MarianMTModel.from_pretrained(model_name, cache_dir=CACHE_DIR).to(DEVICE)
        else:
            tok = AutoTokenizer.from_pretrained(
                model_name, cache_dir=CACHE_DIR, model_max_length=MAX_INPUT_TOKENS
            )
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=CACHE_DIR).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {model_name}: {e}")

    # explanation: Store in cache and return
    translation_cache[lang_pair] = (tok, mdl)
    return tok, mdl

# explanation: Health endpoint to confirm server status and feature toggles
@app.get("/", tags=["health"])
def health():
    return {
        "status": "ok",
        "message": "Translator running",
        "device": DEVICE,
        "grammar": ENABLE_GRAMMAR,
        "slang_normalizer": ENABLE_SLANG,
        "replace_plain_cause": REPLACE_PLAIN_CAUSE,
    }

# explanation: Endpoint to list supported language pairs and their underlying models
@app.get("/models", tags=["info"])
def models():
    return {
        "supported_pairs": sorted(
            [{"src": s, "tgt": t, "model": m} for (s, t), m in MODEL_MAP.items()],
            key=lambda x: (x["src"], x["tgt"])
        )
    }

# explanation: Main translation endpoint — normalizes slang, detects language, corrects grammar, translates
@app.post("/translate", response_model=TranslationResponse, tags=["translate"])
def translate(req: TranslationRequest):
    try:
        # explanation: 1) Normalize slang/abbreviations to standard English where possible
        normalized = normalize_slang(req.text)

        # explanation: 2) Detect source language on normalized text (or trust provided source_lang)
        src = (req.source_lang or detect_language(normalized)).lower()
        tgt = req.target_lang.lower()

        # explanation: If same language, return normalized/optionally corrected text without translating
        if src == tgt:
            corrected_if_en = (
                correct_grammar_if_english(normalized, src) if src == "en" else normalized
            )
            return TranslationResponse(
                detected_source_lang=src,
                target_language=tgt,
                original_text=req.text,
                normalized_text=normalized if normalized != req.text else None,
                corrected_text=corrected_if_en if (src == "en" and corrected_if_en != normalized) else None,
                translated_text=corrected_if_en,
                note="Source and target are the same; normalization/grammar applied only.",
            )

        # explanation: 3) Grammar correction (English only) before translation
        text_for_translation = correct_grammar_if_english(normalized, src)

        # explanation: 4) If there is no direct model, pivot via English (src→en→tgt)
        if (src, tgt) not in MODEL_MAP and src != "en" and tgt != "en":
            # explanation: src → en
            tok1, mdl1 = get_translation_model((src, "en"))
            mid_inputs = tok1(
                text_for_translation,
                return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_TOKENS
            ).to(DEVICE)
            with torch.no_grad():
                mid_out = mdl1.generate(**mid_inputs, max_length=256, num_beams=5)
            mid_text = tok1.decode(mid_out[0], skip_special_tokens=True).replace("▁", " ").strip()

            # explanation: en → tgt
            tok2, mdl2 = get_translation_model(("en", tgt))
            fin_inputs = tok2(
                mid_text,
                return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_TOKENS
            ).to(DEVICE)
            with torch.no_grad():
                fin_out = mdl2.generate(**fin_inputs, max_length=256, num_beams=5)
            final_text = tok2.decode(fin_out[0], skip_special_tokens=True).replace("▁", " ").strip()

            # explanation: Return full response with notes about the pivot
            return TranslationResponse(
                detected_source_lang=src,
                target_language=tgt,
                original_text=req.text,
                normalized_text=normalized if normalized != req.text else None,
                corrected_text=text_for_translation if (src == "en" and text_for_translation != normalized) else None,
                translated_text=final_text,
                note="Translated via English pivot.",
            )

        # explanation: 5) Direct translation path (src → tgt)
        tok, mdl = get_translation_model((src, tgt))
        inputs = tok(
            text_for_translation,
            return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_TOKENS
        ).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_length=256, num_beams=5)
        translated = tok.decode(out[0], skip_special_tokens=True).replace("▁", " ").strip()

        # explanation: Build and return the final response JSON
        return TranslationResponse(
            detected_source_lang=src,
            target_language=tgt,
            original_text=req.text,
            normalized_text=normalized if normalized != req.text else None,
            corrected_text=text_for_translation if (src == "en" and text_for_translation != normalized) else None,
            translated_text=translated,
        )

    # explanation: Pass through explicit HTTP errors unchanged
    except HTTPException:
        raise
    # explanation: For unexpected errors, return a 500 with message
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}")

