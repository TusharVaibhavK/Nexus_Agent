# # model_runner.py
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch, json, re
# from datetime import datetime

# MODEL_NAME = "google/flan-t5-small"
# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# _tokenizer = None
# _model = None

# def load_model():
#     global _tokenizer, _model
#     if _tokenizer is None or _model is None:
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device)
#     return _tokenizer, _model

# # simple heuristics (fast, reliable for structured queries)
# _YEAR_RE = re.compile(r'\b([1-9][0-9]?)\s*(?:st|nd|rd|th)?\s*(?:year|yr)\b', re.IGNORECASE)
# _SEM_RE = re.compile(r'\b(?:sem(?:ester)?\s*)([1-9][0-9]?)\b', re.IGNORECASE)
# _COMMON_SUBJECTS = [
#     "data structures", "operating systems", "mathematics", "physics",
#     "computer networks", "database", "algorithms", "discrete math", "computer networks"
# ]

# def heuristic_extract(text: str):
#     text_l = text.lower()
#     year = None
#     sem = None
#     subject = None

#     y = _YEAR_RE.search(text_l)
#     if y:
#         try:
#             year = int(y.group(1))
#         except:
#             year = None

#     s = _SEM_RE.search(text_l)
#     if s:
#         try:
#             sem = int(s.group(1))
#         except:
#             sem = None

#     for subj in _COMMON_SUBJECTS:
#         if subj in text_l:
#             subject = subj.title()
#             break

#     # fallback: capitalized two-word guesses
#     if not subject:
#         caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', text)
#         if caps:
#             subject = caps[0]

#     keywords = []
#     # basic tokens as keywords
#     for token in re.findall(r'\b\w{3,}\b', text_l):
#         if token not in keywords:
#             keywords.append(token)
#     keywords = keywords[:8]

#     return {"year": year, "semester": sem, "subject": subject, "keywords": keywords}

# # model fallback to extract structured JSON if heuristics fail
# def call_model_json(user_text: str):
#     tokenizer, model = load_model()

#     # small few-shot example to force JSON structure
#     example_input = "Show my 1 year results for Data Structures."
#     example_output = """### BEGIN JSON
# {
#   "intent":"get_subject_marks",
#   "confidence":0.95,
#   "keywords":["1 year","Data Structures","results"],
#   "entities":{"year":1,"semester":null,"subject":"Data Structures","student_id":null},
#   "explanation":"User requests Data Structures marks for year 1",
#   "query_descriptor":{"type":"table_lookup","table":"marks","filters":{"year":1,"subject":"data structures"},"limit":100},
#   "next_action":"call_table_agent"
# }
# ### END JSON
# """
#     prompt = (
#         "Return EXACTLY valid JSON between the markers ### BEGIN JSON and ### END JSON.\n"
#         "Use this schema: intent, confidence (float 0-1), keywords (list), entities (object), explanation, query_descriptor, next_action.\n\n"
#         f"Example:\nUser: \"{example_input}\"\n{example_output}\nNow process this new message and return only JSON between the markers.\n\nUser: \"{user_text}\"\n### BEGIN JSON\n"
#     )

#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(_device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=220,
#         do_sample=True,
#         temperature=0.2,
#         top_p=0.95,
#         num_beams=1,
#         early_stopping=True
#     )
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # ensure end marker exists
#     if "### END JSON" not in decoded:
#         decoded = decoded + "\n### END JSON"

#     # extract first {...} or between markers
#     try:
#         start = decoded.find("### BEGIN JSON")
#         if start != -1:
#             start = decoded.find("{", start)
#         else:
#             start = decoded.find("{")
#         end = decoded.rfind("}")
#         json_str = decoded[start:end+1]
#         return json.loads(json_str)
#     except Exception:
#         return None

# def build_response(request_id: str, text: str):
#     heur = heuristic_extract(text)
#     # if heuristics produced a subject or year, use deterministic template
#     if heur["year"] or heur["subject"] or heur["semester"]:
#         subject_norm = heur["subject"].lower() if heur["subject"] else None
#         filters = {}
#         if heur["year"]:
#             filters["year"] = heur["year"]
#         if heur["semester"]:
#             filters["semester"] = heur["semester"]
#         if subject_norm:
#             filters["subject"] = subject_norm

#         resp = {
#             "request_id": request_id,
#             "intents":[{"name":"get_subject_marks","confidence":0.9}],
#             "primary_intent":"get_subject_marks",
#             "entities": {"year": heur["year"], "semester": heur["semester"], "subject": heur["subject"], "student_id": None},
#             "query_descriptor": {"type":"table_lookup","table":"marks","filters": filters,"limit":200},
#             "next_action":"call_table_agent",
#             "explanation": f"Detected subject marks request for {heur['subject']} and year {heur['year']}" if heur["subject"] or heur["year"] else "",
#             "timestamp": datetime.utcnow().isoformat() + "Z"
#         }
#         return resp

#     # fallback to model
#     parsed = call_model_json(text)
#     if parsed and isinstance(parsed, dict):
#         # build stable response with defaults
#         intent = parsed.get("intent","unknown")
#         conf = float(parsed.get("confidence",0.0))
#         keywords = parsed.get("keywords",[]) if isinstance(parsed.get("keywords",list)) else []
#         entities = parsed.get("entities",{}) if isinstance(parsed.get("entities",dict)) else {}
#         qd = parsed.get("query_descriptor",{"type":"table_lookup","table":"marks","filters":{},"limit":100})
#         next_action = parsed.get("next_action","ask_clarification")
#         explanation = parsed.get("explanation","")
#         resp = {
#             "request_id": request_id,
#             "intents":[{"name":intent,"confidence":conf}],
#             "primary_intent":intent,
#             "entities": entities,
#             "query_descriptor": qd,
#             "next_action": next_action,
#             "explanation": explanation,
#             "timestamp": datetime.utcnow().isoformat() + "Z"
#         }
#         return resp

#     # last fallback: unknown
#     return {
#         "request_id": request_id,
#         "intents":[{"name":"unknown","confidence":0.0}],
#         "primary_intent":"unknown",
#         "entities": {},
#         "query_descriptor":{"type":"table_lookup","table":"marks","filters":{},"limit":100},
#         "next_action":"ask_clarification",
#         "explanation": text,
#         "timestamp": datetime.utcnow().isoformat() + "Z"
#     }


# model_runner.py
"""
Final model runner for the Intent Agent.

Capabilities:
- Heuristic-first extraction for common intents (fast, deterministic).
- Template-based query_descriptor generation for known intents.
- LLM fallback to dynamically generate a validated query_descriptor when templates don't apply.
- In-memory caching of LLM-generated descriptors.
- Defensive parsing/validation to avoid unsafe or invalid descriptors.
"""

import re
import json
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# transformers + torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------------
# Configuration & globals
# ---------------------------
MODEL_NAME = "google/flan-t5-small"   # change if you want a larger/smaller model
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None

# simple in-memory cache for generated descriptors
_descriptor_cache: Dict[str, Dict[str, Any]] = {}

# ---------------------------
# Intent templates (safe)
# ---------------------------
INTENT_TEMPLATES = {
    "get_subject_marks": {
        "type": "table_lookup",
        "table": "marks",
        "filters": {"year": "{year}", "semester": "{semester}", "subject": "{subject}"},
        "limit": 200,
        "required_slots": ["subject"]
    },
    "get_year_results": {
        "type": "table_lookup",
        "table": "marks",
        "filters": {"year": "{year}"},
        "limit": 200,
        "required_slots": ["year"]
    },
    "get_cgpa": {
        "type": "agg_query",
        "table": "marks",
        "agg": "AVG(marks)",
        "filters": {"student_external_id": "{student_id}", "year": "{year}"},
        "limit": 1,
        "required_slots": ["student_id"]
    },
    "get_timetable": {
        "type": "table_lookup",
        "table": "timetable",
        "filters": {"semester": "{semester}", "day": "{day}", "subject_name": "{subject_name}"},
        "limit": 100,
        "required_slots": ["semester"]
    }
}

# ---------------------------
# Allowed tables + filters (for validation)
# ---------------------------
ALLOWED_TABLES = {"marks", "timetable", "students", "subjects", "documents_embeddings"}
ALLOWED_FILTERS = {
    "marks": {"year", "semester", "subject", "student_external_id"},
    "timetable": {"semester", "day", "subject_name"},
    "students": {"external_id", "year", "program"},
    "subjects": {"subject_code", "name", "semester", "year"},
    "documents_embeddings": {"metadata", "source_table", "source_id"}
}

# ---------------------------
# Heuristics / regexes
# ---------------------------
_YEAR_RE = re.compile(r'\b([1-9][0-9]?)\s*(?:st|nd|rd|th)?\s*(?:year|yr)\b', re.IGNORECASE)
_SEM_RE = re.compile(r'\b(?:sem(?:ester)?\s*)([1-9][0-9]?)\b', re.IGNORECASE)
_CGPA_RE = re.compile(r'\bcgpa\b|\bgrade point average\b|\bgrade point\b', re.IGNORECASE)
_TIMETABLE_RE = re.compile(r'\btimetable\b|\bclass schedule\b|\bnext class\b|\bwhen is\b', re.IGNORECASE)

_SUBJECT_LIST = [
    "data structures","operating systems","mathematics","physics",
    "computer networks","database","algorithms","discrete math","machine learning"
]

# ---------------------------
# LLM prompt template (for dynamic descriptor generation)
# ---------------------------
PROMPT_TEMPLATE = (
    "You are an assistant that must generate a structured JSON query_descriptor for a backend Table Agent.\n"
    "Return only valid JSON between the markers ### BEGIN JSON and ### END JSON.\n\n"
    "Schema for query_descriptor:\n"
    "{\n"
    '  "type": one of ["table_lookup","agg_query","semantic_search"],\n'
    '  "table": "<table name>",\n'
    '  "filters": { "<column>": value, ... },\n'
    '  "limit": integer\n'
    "}\n\n"
    "Rules:\n"
    "- Use only these allowed tables: marks, timetable, students, subjects, documents_embeddings.\n"
    "- Allowed filter keys for each table:\n"
    "  - marks: year, semester, subject, student_external_id\n"
    "  - timetable: semester, day, subject_name\n"
    "  - students: external_id, year, program\n"
    "  - subjects: subject_code, name, semester, year\n"
    "  - documents_embeddings: metadata (as object), source_table, source_id\n"
    "- Normalize subject names to lowercase strings.\n"
    "- Cast numeric-looking slots to integers.\n"
    "- If you cannot produce a valid descriptor with reasonable confidence, return:\n"
    '{ "next_action":"ask_clarification", "missing_slots": ["list","of","missing"], "explanation":"short reason" }\n\n'
    "Examples:\n\n"
    "User: \"Show my 2nd year marks for Computer Networks\"\n"
    "### BEGIN JSON\n"
    "{\n"
    '  "type":"table_lookup",\n'
    '  "table":"marks",\n'
    '  "filters":{"year":2,"subject":"computer networks"},\n'
    '  "limit":200\n'
    "}\n"
    "### END JSON\n\n"
    "User: \"What's my CGPA?\"\n"
    "### BEGIN JSON\n"
    "{\n"
    '  "type":"agg_query",\n'
    '  "table":"marks",\n'
    '  "filters":{"student_external_id": null},\n'
    '  "limit":1\n'
    "}\n"
    "### END JSON\n\n"
    "Now produce JSON for this user message (return only the JSON between the markers):\n\n"
    "User: \"{user_text}\"\n"
    "### BEGIN JSON\n"
)

# ---------------------------
# Model loading
# ---------------------------
def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print(f"[model_runner] Loading model {MODEL_NAME} on device {_device}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device)
        print("[model_runner] Model loaded")
    return _tokenizer, _model

# ---------------------------
# Heuristic extractor
# ---------------------------
def simple_subject_find(text: str) -> Optional[str]:
    tl = text.lower()
    for subj in _SUBJECT_LIST:
        if subj in tl:
            return subj.title()
    caps = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', text)
    if caps:
        return caps[0]
    return None

def heuristic_intent_and_entities(text: str) -> Dict[str, Any]:
    tl = text.lower()
    year = None
    sem = None
    student_id = None
    subject = None

    my = _YEAR_RE.search(text)
    if my:
        try:
            year = int(my.group(1))
        except:
            year = None

    ms = _SEM_RE.search(text)
    if ms:
        try:
            sem = int(ms.group(1))
        except:
            sem = None

    subject = simple_subject_find(text)

    # CGPA intent
    if _CGPA_RE.search(tl):
        return {"intent": "get_cgpa", "entities": {"student_id": None, "year": year}}

    # timetable intent
    if _TIMETABLE_RE.search(tl):
        return {"intent": "get_timetable", "entities": {"semester": sem, "day": None}}

    # subject marks if subject present
    if subject:
        return {"intent": "get_subject_marks", "entities": {"year": year, "semester": sem, "subject": subject, "student_id": student_id}}

    # year results if only year present
    if year and not subject:
        return {"intent": "get_year_results", "entities": {"year": year}}

    # fallback
    return {"intent": "fallback", "entities": {}}

# ---------------------------
# Template filling / validation
# ---------------------------
def fill_template(intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    tmpl = INTENT_TEMPLATES.get(intent)
    if not tmpl:
        return {"type": "unknown", "table": None, "filters": {}, "limit": 0}
    filters = {}
    for fk, fv in tmpl["filters"].items():
        if isinstance(fv, str) and fv.startswith("{") and fv.endswith("}"):
            slot = fv[1:-1]
            val = entities.get(slot)
            if val is not None:
                if slot == "subject" and isinstance(val, str):
                    val = val.strip()
                filters[fk] = val
        else:
            filters[fk] = fv
    return {"type": tmpl["type"], "table": tmpl["table"], "filters": filters, "limit": tmpl.get("limit", 100)}

def missing_slots_for_intent(intent: str, entities: Dict[str, Any]) -> list:
    tmpl = INTENT_TEMPLATES.get(intent, {})
    req = tmpl.get("required_slots", [])
    missing = []
    for s in req:
        if entities.get(s) is None:
            missing.append(s)
    return missing

# ---------------------------
# Descriptor fingerprinting & normalize helpers
# ---------------------------
def _fingerprint_text(text: str) -> str:
    key = text.strip().lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _normalize_descriptor(d: Dict[str, Any]) -> Dict[str, Any]:
    filters = {}
    for k, v in (d.get("filters") or {}).items():
        if v is None:
            filters[k] = None
            continue
        if isinstance(v, str):
            vn = v.strip()
            if k in {"subject", "subject_name", "name"}:
                filters[k] = vn.lower()
            else:
                if re.fullmatch(r"\d+", vn):
                    filters[k] = int(vn)
                else:
                    filters[k] = vn
        else:
            filters[k] = v
    d["filters"] = filters
    try:
        d["limit"] = int(d.get("limit", 100))
    except:
        d["limit"] = 100
    return d

def _validate_descriptor(d: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "descriptor not an object"
    t = d.get("type")
    if t not in {"table_lookup", "agg_query", "semantic_search"}:
        return False, f"invalid type {t}"
    table = d.get("table")
    if table not in ALLOWED_TABLES:
        return False, f"table {table} not allowed"
    filters = d.get("filters", {})
    if not isinstance(filters, dict):
        return False, "filters must be object"
    allowed = ALLOWED_FILTERS.get(table, set())
    for k in filters.keys():
        if k not in allowed:
            return False, f"filter {k} not allowed for table {table}"
    limit = d.get("limit", 0)
    try:
        li = int(limit)
        if li < 0 or li > 10000:
            return False, "limit out of range"
    except Exception:
        return False, "limit not integer"
    return True, ""

# ---------------------------
# LLM-based descriptor generation (dynamic)
# ---------------------------
def generate_descriptor_with_llm(user_text: str, max_new_tokens: int = 220) -> Dict[str, Any]:
    """
    Returns either a validated descriptor dict OR an 'ask_clarification' dict.
    Uses caching.
    """
    key = _fingerprint_text(user_text)
    if key in _descriptor_cache:
        return _descriptor_cache[key]

    tokenizer, model = load_model()
    prompt = PROMPT_TEMPLATE.replace("{user_text}", user_text)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(_device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.25,
            top_p=0.95,
            num_beams=1,
            early_stopping=True
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print("[model_runner] LLM generation failed:", e)
        result = {"next_action": "ask_clarification", "missing_slots": [], "explanation": "LLM generation error"}
        _descriptor_cache[key] = result
        return result

    # extract JSON substring between markers or first {...}
    start = decoded.find("### BEGIN JSON")
    if start != -1:
        start = decoded.find("{", start)
    else:
        start = decoded.find("{")
    end = decoded.rfind("}")
    json_str = None
    if start != -1 and end != -1 and end > start:
        json_str = decoded[start:end + 1]

    if not json_str:
        print("[model_runner] LLM did not produce JSON between markers. Raw:", decoded[:400])
        result = {"next_action": "ask_clarification", "missing_slots": [], "explanation": "LLM did not produce descriptor"}
        _descriptor_cache[key] = result
        return result

    try:
        d = json.loads(json_str)
    except Exception as e:
        print("[model_runner] Failed to parse JSON from LLM:", e)
        print("raw:", decoded[:500])
        result = {"next_action": "ask_clarification", "missing_slots": [], "explanation": "LLM produced invalid JSON"}
        _descriptor_cache[key] = result
        return result

    # normalize and validate
    d = _normalize_descriptor(d)
    ok, reason = _validate_descriptor(d)
    if not ok:
        print("[model_runner] Generated descriptor invalid:", d, "reason:", reason)
        result = {"next_action": "ask_clarification", "missing_slots": [], "explanation": f"Invalid descriptor: {reason}"}
        _descriptor_cache[key] = result
        return result

    _descriptor_cache[key] = d
    return d

# ---------------------------
# High-level build_response (used by FastAPI app)
# ---------------------------
def build_response(request_id: str, text: str) -> Dict[str, Any]:
    """
    Main entry. Try heuristics & templates first; if they fail or required slots missing,
    use LLM to dynamically generate a validated descriptor (or ask for clarification).
    """
    heur = heuristic_intent_and_entities(text)

    # Heuristic path (fast)
    if heur["intent"] != "fallback":
        intent = heur["intent"]
        entities = heur.get("entities", {})
        missing = missing_slots_for_intent(intent, entities)
        if not missing:
            qd = fill_template(intent, entities)
            return {
                "request_id": request_id,
                "intents": [{"name": intent, "confidence": 0.9}],
                "primary_intent": intent,
                "entities": entities,
                "query_descriptor": qd,
                "next_action": "call_table_agent",
                "explanation": f"Heuristic match for {intent}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        else:
            # Ask for missing slots rather than calling LLM for trivial clarifications
            return {
                "request_id": request_id,
                "intents": [{"name": intent, "confidence": 0.6}],
                "primary_intent": intent,
                "entities": entities,
                "query_descriptor": {"type": "table_lookup", "table": "marks", "filters": {}, "limit": 0},
                "next_action": "ask_clarification",
                "missing_slots": missing,
                "explanation": f"Missing required fields: {missing}",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    # LLM dynamic descriptor path
    descriptor_or_action = generate_descriptor_with_llm(text)
    if descriptor_or_action.get("next_action") == "ask_clarification":
        # LLM couldn't produce a safe descriptor
        return {
            "request_id": request_id,
            "intents": [{"name": "fallback", "confidence": 0.0}],
            "primary_intent": "fallback",
            "entities": {},
            "query_descriptor": {"type": "table_lookup", "table": "marks", "filters": {}, "limit": 0},
            "next_action": "ask_clarification",
            "missing_slots": descriptor_or_action.get("missing_slots", []),
            "explanation": descriptor_or_action.get("explanation", "LLM requested clarification"),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    # descriptor_or_action is a validated descriptor
    qd = descriptor_or_action
    # optionally extract entities from filters for convenience
    entities = {}
    for k, v in qd.get("filters", {}).items():
        # map some known keys to entity names
        if k in {"year", "semester", "subject", "student_external_id", "subject_name"}:
            entities[k if k != "student_external_id" else "student_id"] = v

    return {
        "request_id": request_id,
        "intents": [{"name": "dynamic", "confidence": 0.7}],
        "primary_intent": "dynamic",
        "entities": entities,
        "query_descriptor": qd,
        "next_action": "call_table_agent",
        "explanation": "LLM-generated descriptor",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

# ---------------------------
# Utility: clear cache (for dev)
# ---------------------------
def clear_descriptor_cache():
    global _descriptor_cache
    _descriptor_cache = {}

# If you want to warm-load the model at module import, uncomment:
# load_model()
