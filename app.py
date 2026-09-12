
import os
import re
import json
import hashlib
from io import BytesIO
from datetime import datetime

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# VIZA PILOT — GERMANY STUDENT VISA READINESS MVP
# Scope:
#   Nationality: Pakistan
#   Destination: Germany
#   Visa type: Student
#
# Stack:
#   Streamlit + Python + FAISS + Sentence Transformers + Groq
#
# IMPORTANT:
# - Readiness is NOT visa approval probability.
# - Official requirements should always be reverified before launch.
# - This MVP uses structured rules derived from the supplied guide.
# ============================================================


APP_NAME = "Viza Pilot"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 160

st.set_page_config(
    page_title="Viza Pilot | Germany Student Visa",
    page_icon="✈️",
    layout="wide",
)


# ============================================================
# 1. VERIFIED KNOWLEDGE / STRUCTURED RULES
# ============================================================

# These rules are based on the Germany Visa Guide supplied for this MVP.
# They should be reverified against the linked German Missions pages
# before production launch.

REQUIREMENTS = [
    {
        "id": "passport",
        "category": "Identity",
        "title": "Passport",
        "required": True,
        "description": "A valid passport is required for the student visa application.",
        "accepted_doc_types": ["passport"],
        "source_page": 8,
        "source_label": "Germany Visa Guide — Required Documents Quick Reference",
        "priority": 100,
    },
    {
        "id": "cnic",
        "category": "Identity",
        "title": "CNIC",
        "required": True,
        "description": "CNIC is listed as a core identity document.",
        "accepted_doc_types": ["cnic"],
        "source_page": 8,
        "source_label": "Germany Visa Guide — Required Documents Quick Reference",
        "priority": 95,
    },
    {
        "id": "admission_letter",
        "category": "Academic",
        "title": "Admission letter",
        "required": True,
        "description": "Evidence of admission to the German study programme.",
        "accepted_doc_types": ["admission_letter"],
        "source_page": 8,
        "source_label": "Germany Visa Guide — Required Documents Quick Reference",
        "priority": 100,
    },
    {
        "id": "degree_certificate",
        "category": "Academic",
        "title": "Previous degree certificate(s)",
        "required": True,
        "description": "Past degree certificates are described as mandatory where applicable.",
        "accepted_doc_types": ["degree_certificate"],
        "source_page": 5,
        "source_label": "Germany Visa Guide — Mandatory Documents",
        "priority": 95,
    },
    {
        "id": "transcript",
        "category": "Academic",
        "title": "Degree transcript(s)",
        "required": True,
        "description": "Degree transcripts are described as mandatory.",
        "accepted_doc_types": ["transcript"],
        "source_page": 5,
        "source_label": "Germany Visa Guide — Mandatory Documents",
        "priority": 95,
    },
    {
        "id": "tuition_payment",
        "category": "Academic",
        "title": "Tuition-fee payment evidence",
        "required": False,
        "conditional_key": "tuition_applicable",
        "description": "Proof of payment of tuition fees is required where applicable to the course.",
        "accepted_doc_types": ["tuition_payment"],
        "source_page": 5,
        "source_label": "Germany Visa Guide — Mandatory Documents",
        "priority": 80,
    },
    {
        "id": "financial_proof",
        "category": "Financial",
        "title": "Financial proof",
        "required": True,
        "description": (
            "The guide lists three recognised routes: blocked account, deed of obligation, "
            "or an accepted official scholarship."
        ),
        "accepted_doc_types": [
            "blocked_account",
            "deed_of_obligation",
            "scholarship",
        ],
        "source_page": 7,
        "source_label": "Germany Visa Guide — Blocked Account & Financial Requirements",
        "priority": 100,
    },
    {
        "id": "health_insurance",
        "category": "Insurance",
        "title": "Health insurance",
        "required": True,
        "description": "Health insurance appropriate to the applicant's stay in Germany.",
        "accepted_doc_types": ["health_insurance"],
        "source_page": 8,
        "source_label": "Germany Visa Guide — Required Documents Quick Reference",
        "priority": 85,
    },
    {
        "id": "accommodation",
        "category": "Accommodation",
        "title": "Proof of housing arrangements",
        "required": True,
        "description": "Proof of housing/accommodation arrangements in Germany.",
        "accepted_doc_types": ["accommodation"],
        "source_page": 8,
        "source_label": "Germany Visa Guide — Required Documents Quick Reference",
        "priority": 85,
    },
    {
        "id": "portal_application",
        "category": "Application",
        "title": "Consular Services Portal application",
        "required": True,
        "description": (
            "Student visa applications are handled through the Consular Services Portal, "
            "which generates a tailored document list after the questionnaire."
        ),
        "accepted_doc_types": ["portal_application"],
        "source_page": 5,
        "source_label": "Germany Visa Guide — Consular Services Portal Procedure",
        "priority": 100,
    },
    {
        "id": "appointment_confirmation",
        "category": "Application",
        "title": "Appointment confirmation",
        "required": False,
        "conditional_key": "appointment_booked",
        "description": "Printed appointment confirmation is part of the application-day checklist.",
        "accepted_doc_types": ["appointment_confirmation"],
        "source_page": 8,
        "source_label": "Germany Visa Guide — Application-Day Checklist",
        "priority": 70,
    },
]


KNOWLEDGE_PASSAGES = [
    {
        "page": 5,
        "title": "Student visa process",
        "text": (
            "Germany's student visa process for Pakistan is described as fully digital through "
            "the Consular Services Portal. The applicant registers, completes an interactive "
            "questionnaire, receives a situation-specific document list, uploads documents, "
            "and waits for initial screening. The applicant is then either sent an appointment "
            "notification or asked to submit corrected or additional documents."
        ),
    },
    {
        "page": 5,
        "title": "Mandatory academic documents",
        "text": (
            "The guide warns that some portal items may appear optional even though they are "
            "mandatory. It specifically names past degree certificates, degree transcripts, "
            "and proof of tuition-fee payment where applicable."
        ),
    },
    {
        "page": 5,
        "title": "Document preparation",
        "text": (
            "Applicants are advised to prepare documents in advance, use well-readable A4 copies, "
            "and arrange documents in the order specified by the portal."
        ),
    },
    {
        "page": 7,
        "title": "Financial evidence",
        "text": (
            "For long-stay student applications, sufficient funds are central. The guide lists "
            "three recognised routes: a blocked account, a deed of obligation from a sponsor in "
            "Germany, or an accepted official scholarship."
        ),
    },
    {
        "page": 7,
        "title": "Blocked account amount in supplied guide",
        "text": (
            "The supplied guide states that from 1 January 2025 the blocked-account amount is "
            "€992 per month or €11,904 per year. This is time-sensitive and must be reverified "
            "against the current official German Missions guidance before relying on it."
        ),
    },
    {
        "page": 7,
        "title": "Scholarships",
        "text": (
            "The guide states that a scholarship from a Pakistani university alone is not "
            "sufficient for the stated financial-evidence route. It describes scholarships "
            "from HEC or German/other foreign official scholarship bodies as accepted."
        ),
    },
    {
        "page": 8,
        "title": "Student quick-reference documents",
        "text": (
            "The student quick-reference lists passport and CNIC; Consular Services Portal "
            "questionnaire; degree certificates, transcripts, tuition-fee payment proof and "
            "admission letter; financial proof; health insurance; and proof of housing "
            "arrangements in Germany."
        ),
    },
    {
        "page": 8,
        "title": "Application-day checklist",
        "text": (
            "The application-day checklist includes appointment confirmation, original passport(s), "
            "the completed application form, two recent biometric photos, the complete document "
            "set in the specified order, the visa fee, and financial-proof documentation relevant "
            "to the long-stay/student application."
        ),
    },
    {
        "page": 8,
        "title": "Fraud warning",
        "text": (
            "The guide warns that appointments cannot be bought or guaranteed. Applicants should "
            "use official portals, should not disclose portal credentials, and should verify "
            "communications against the official diplo.de domain."
        ),
    },
    {
        "page": 9,
        "title": "Source verification warning",
        "text": (
            "The supplied guide says it is an independent informational summary compiled from "
            "German Missions in Pakistan sources. It advises applicants to verify current fees, "
            "blocked-account amounts and procedural details directly with the official sources "
            "because requirements change."
        ),
    },
]


# ============================================================
# 2. GENERAL HELPERS
# ============================================================

def get_secret(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


def groq_is_configured():
    return bool(get_secret("GROQ_API_KEY"))


def get_groq_client():
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add it in Streamlit Cloud → App settings → Secrets."
        )
    return Groq(api_key=key)


@st.cache_resource(show_spinner=False)
def embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def build_knowledge_index():
    model = embedding_model()
    texts = [x["text"] for x in KNOWLEDGE_PASSAGES]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve_knowledge(query, k=TOP_K):
    index = build_knowledge_index()
    model = embedding_model()

    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(k, len(KNOWLEDGE_PASSAGES))
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(KNOWLEDGE_PASSAGES[idx])
            item["score"] = float(score)
            results.append(item)
    return results


def safe_text(value):
    return (value or "").strip()


# ============================================================
# 3. SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "screen": "dashboard",
        "profile": {
            "full_name": "",
            "dob": "",
            "passport_number": "",
            "nationality": "Pakistan",
            "destination": "Germany",
            "visa_type": "Student",
            "study_level": "",
            "university": "",
            "course": "",
            "admission_received": False,
            "funding_method": "",
            "tuition_applicable": False,
            "tuition_paid": False,
            "appointment_booked": False,
            "accommodation_arranged": False,
            "insurance_arranged": False,
            "profile_confirmed": False,
        },
        "documents": [],
        "confirmed_documents": {},
        "manual_requirement_states": {},
        "issues": [],
        "chat": [],
        "last_readiness": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_application():
    for key in [
        "profile",
        "documents",
        "confirmed_documents",
        "manual_requirement_states",
        "issues",
        "chat",
        "last_readiness",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    init_state()
    st.session_state.screen = "dashboard"


def go(screen):
    st.session_state.screen = screen
    st.rerun()


# ============================================================
# 4. PERSONALIZED CHECKLIST
# ============================================================

def requirement_is_applicable(req):
    profile = st.session_state.profile

    if req.get("required"):
        return True

    conditional_key = req.get("conditional_key")
    if conditional_key:
        return bool(profile.get(conditional_key))

    return False


def personalized_requirements():
    return [r for r in REQUIREMENTS if requirement_is_applicable(r)]


def requirement_status(req):
    accepted_types = req["accepted_doc_types"]

    confirmed = st.session_state.confirmed_documents
    for doc_type in accepted_types:
        if confirmed.get(doc_type, False):
            return "complete"

    manual = st.session_state.manual_requirement_states.get(req["id"])
    if manual == "complete":
        return "complete"
    if manual == "not_applicable":
        return "not_applicable"

    return "missing"


# ============================================================
# 5. DOCUMENT EXTRACTION
# ============================================================

DOC_TYPE_HINTS = {
    "passport": [
        "passport",
        "date of expiry",
        "place of birth",
        "nationality",
    ],
    "cnic": [
        "national identity card",
        "cnic",
        "identity card",
    ],
    "admission_letter": [
        "admission",
        "offer of admission",
        "university",
        "programme",
        "program",
        "enrolment",
        "enrollment",
    ],
    "degree_certificate": [
        "degree certificate",
        "bachelor",
        "master",
        "awarded the degree",
        "graduated",
    ],
    "transcript": [
        "transcript",
        "credits",
        "semester",
        "grade",
        "cgpa",
        "gpa",
    ],
    "tuition_payment": [
        "tuition",
        "fee payment",
        "payment receipt",
        "fees paid",
    ],
    "blocked_account": [
        "blocked account",
        "sperrkonto",
        "blocked amount",
    ],
    "deed_of_obligation": [
        "verpflichtungserklärung",
        "declaration of commitment",
        "deed of obligation",
    ],
    "scholarship": [
        "scholarship",
        "stipend",
        "funding award",
        "hec",
    ],
    "health_insurance": [
        "insurance",
        "health cover",
        "health insurance",
    ],
    "accommodation": [
        "accommodation",
        "housing",
        "tenancy",
        "rental",
        "dormitory",
        "wohnheim",
    ],
    "portal_application": [
        "consular services portal",
        "visa application",
        "application form",
    ],
    "appointment_confirmation": [
        "appointment",
        "appointment confirmation",
        "booking confirmation",
    ],
}


def extract_text_from_upload(uploaded_file):
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(data))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages).strip(), None
        except Exception as exc:
            return "", f"PDF extraction failed: {type(exc).__name__}"

    if name.endswith(".txt"):
        try:
            return data.decode("utf-8", errors="ignore"), None
        except Exception as exc:
            return "", f"Text extraction failed: {type(exc).__name__}"

    return "", "Unsupported file type for this MVP."


def classify_document(text, filename):
    haystack = f"{filename}\n{text}".lower()
    scores = {}

    for doc_type, hints in DOC_TYPE_HINTS.items():
        score = sum(1 for hint in hints if hint in haystack)
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        return "unknown", "low"

    confidence = "high" if best_score >= 2 else "medium"
    return best_type, confidence


def extract_fields(text, doc_type):
    fields = {}

    # Generic possible name patterns
    name_patterns = [
        r"(?:name|full name|student name|applicant name)\s*[:\-]\s*([A-Z][A-Za-z .'\-]{3,60})",
        r"(?:surname)\s*[:\-]\s*([A-Z][A-Za-z .'\-]{2,40})",
    ]
    for pattern in name_patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            fields["name"] = re.sub(r"\s+", " ", m.group(1)).strip()
            break

    passport_patterns = [
        r"(?:passport(?: number| no\.?| #)?)\s*[:\-]?\s*([A-Z]{1,3}\d{5,9})",
        r"\b([A-Z]{2}\d{7})\b",
    ]
    for pattern in passport_patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            fields["passport_number"] = m.group(1).upper()
            break

    dob_patterns = [
        r"(?:date of birth|dob|birth date)\s*[:\-]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})",
        r"(?:date of birth|dob|birth date)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    ]
    for pattern in dob_patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            fields["date_of_birth"] = m.group(1).strip()
            break

    expiry_patterns = [
        r"(?:date of expiry|expiry date|expires|valid until)\s*[:\-]?\s*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})",
        r"(?:date of expiry|expiry date|expires|valid until)\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    ]
    for pattern in expiry_patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            fields["expiry_date"] = m.group(1).strip()
            break

    if doc_type == "admission_letter":
        university_match = re.search(
            r"([A-Z][A-Za-z&,\- ]{3,80}(?:University|Universität|Institute|College))",
            text,
            flags=re.I,
        )
        if university_match:
            fields["possible_institution"] = re.sub(
                r"\s+", " ", university_match.group(1)
            ).strip()

    if doc_type == "blocked_account":
        amount_match = re.search(
            r"(?:€|EUR)\s*([0-9][0-9,.\s]{2,12})",
            text,
            flags=re.I,
        )
        if amount_match:
            fields["possible_amount"] = amount_match.group(1).strip()

    return fields


def analyze_uploaded_file(uploaded_file):
    text, error = extract_text_from_upload(uploaded_file)
    doc_type, confidence = classify_document(text, uploaded_file.name)
    extracted = extract_fields(text, doc_type) if text else {}

    warnings = []
    if error:
        warnings.append(error)
    if not text:
        warnings.append(
            "No machine-readable text was extracted. This MVP does not yet include OCR for scanned documents."
        )
    if doc_type == "unknown":
        warnings.append("Document type could not be identified confidently.")

    return {
        "id": hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12],
        "filename": uploaded_file.name,
        "doc_type": doc_type,
        "confidence": confidence,
        "extracted": extracted,
        "warnings": warnings,
        "text_preview": text[:1500],
        "confirmed": False,
    }


# ============================================================
# 6. CONSISTENCY / ISSUE ENGINE
# ============================================================

def normalize_name(value):
    value = safe_text(value).lower()
    value = re.sub(r"[^a-z]", "", value)
    return value


def names_similar(a, b):
    a = normalize_name(a)
    b = normalize_name(b)
    if not a or not b:
        return True

    if a == b:
        return True

    # Simple containment tolerance for titles/middle names.
    if a in b or b in a:
        return True

    # Lightweight edit similarity
    import difflib
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= 0.88


def build_issues():
    issues = []
    profile = st.session_state.profile
    docs = st.session_state.documents

    # Missing requirements
    for req in personalized_requirements():
        status = requirement_status(req)
        if status == "missing":
            issues.append({
                "severity": "blocking" if req["required"] else "action",
                "type": "missing_document",
                "title": f"Missing: {req['title']}",
                "detail": req["description"],
                "requirement_id": req["id"],
                "source_page": req["source_page"],
                "priority": req["priority"],
            })

    # Unconfirmed AI extraction
    for doc in docs:
        if not doc.get("confirmed"):
            issues.append({
                "severity": "review",
                "type": "unconfirmed_extraction",
                "title": f"Confirm extracted data: {doc['filename']}",
                "detail": (
                    "AI-extracted document information has not yet been confirmed by the user."
                ),
                "doc_id": doc["id"],
                "priority": 85,
            })

    # Classification uncertainty / extraction errors
    for doc in docs:
        if doc.get("confidence") == "low" or doc.get("warnings"):
            issues.append({
                "severity": "review",
                "type": "document_quality",
                "title": f"Review document: {doc['filename']}",
                "detail": " ".join(doc.get("warnings", [])) or "Low-confidence document classification.",
                "doc_id": doc["id"],
                "priority": 80,
            })

    # Cross-document name consistency
    names = []
    if safe_text(profile.get("full_name")):
        names.append(("Applicant profile", profile["full_name"]))

    for doc in docs:
        if doc.get("confirmed") and safe_text(doc.get("extracted", {}).get("name")):
            names.append((doc["filename"], doc["extracted"]["name"]))

    if len(names) >= 2:
        reference_label, reference_name = names[0]
        for label, name in names[1:]:
            if not names_similar(reference_name, name):
                issues.append({
                    "severity": "blocking",
                    "type": "name_inconsistency",
                    "title": "Name inconsistency detected",
                    "detail": (
                        f"{reference_label} shows '{reference_name}', while {label} shows '{name}'. "
                        "Check whether the difference is legitimate before submission."
                    ),
                    "priority": 100,
                })

    # Passport number consistency
    profile_passport = safe_text(profile.get("passport_number")).upper()
    for doc in docs:
        if not doc.get("confirmed"):
            continue
        extracted_passport = safe_text(
            doc.get("extracted", {}).get("passport_number")
        ).upper()

        if profile_passport and extracted_passport and profile_passport != extracted_passport:
            issues.append({
                "severity": "blocking",
                "type": "passport_inconsistency",
                "title": "Passport number inconsistency",
                "detail": (
                    f"Applicant profile shows '{profile_passport}', while "
                    f"{doc['filename']} shows '{extracted_passport}'."
                ),
                "priority": 100,
            })

    # Profile completeness
    key_profile_fields = {
        "Full name": profile.get("full_name"),
        "Study level": profile.get("study_level"),
        "University": profile.get("university"),
        "Course": profile.get("course"),
        "Funding method": profile.get("funding_method"),
    }
    for label, value in key_profile_fields.items():
        if not safe_text(value):
            issues.append({
                "severity": "review",
                "type": "profile_incomplete",
                "title": f"Profile information missing: {label}",
                "detail": "Complete the applicant profile so requirements can be checked accurately.",
                "priority": 70,
            })

    issues.sort(key=lambda x: x.get("priority", 0), reverse=True)
    st.session_state.issues = issues
    return issues


# ============================================================
# 7. READINESS ENGINE
# ============================================================

def calculate_readiness():
    reqs = personalized_requirements()
    issues = build_issues()

    required_reqs = [r for r in reqs if r["required"]]
    optional_applicable_reqs = [r for r in reqs if not r["required"]]

    required_completed = sum(
        requirement_status(r) == "complete" for r in required_reqs
    )
    optional_completed = sum(
        requirement_status(r) in {"complete", "not_applicable"}
        for r in optional_applicable_reqs
    )

    req_denominator = max(1, len(required_reqs))
    completeness_ratio = required_completed / req_denominator

    # User-confirmed extraction quality
    if st.session_state.documents:
        confirmed_ratio = sum(
            bool(d.get("confirmed")) for d in st.session_state.documents
        ) / len(st.session_state.documents)
    else:
        confirmed_ratio = 0.0

    blocking_issues = [i for i in issues if i["severity"] == "blocking"]
    review_issues = [i for i in issues if i["severity"] == "review"]

    consistency_score = 1.0
    inconsistency_count = sum(
        i["type"] in {"name_inconsistency", "passport_inconsistency"}
        for i in blocking_issues
    )
    if inconsistency_count:
        consistency_score = max(0.0, 1.0 - 0.4 * inconsistency_count)

    profile = st.session_state.profile
    profile_fields = [
        profile.get("full_name"),
        profile.get("study_level"),
        profile.get("university"),
        profile.get("course"),
        profile.get("funding_method"),
    ]
    profile_ratio = sum(bool(safe_text(v)) for v in profile_fields) / len(profile_fields)

    # Deterministic score:
    # 55% mandatory requirement completeness
    # 15% user-confirmed extraction
    # 15% cross-document consistency
    # 15% profile completeness
    score = round(
        completeness_ratio * 55
        + confirmed_ratio * 15
        + consistency_score * 15
        + profile_ratio * 15
    )

    # Conservative cap when blocking issues remain.
    if blocking_issues:
        score = min(score, 79)

    result = {
        "score": max(0, min(score, 100)),
        "required_complete": required_completed,
        "required_total": len(required_reqs),
        "optional_complete": optional_completed,
        "optional_total": len(optional_applicable_reqs),
        "blocking_count": len(blocking_issues),
        "review_count": len(review_issues),
        "issues": issues,
        "components": {
            "mandatory_requirements": round(completeness_ratio * 100),
            "confirmed_extraction": round(confirmed_ratio * 100),
            "consistency": round(consistency_score * 100),
            "profile": round(profile_ratio * 100),
        },
    }

    st.session_state.last_readiness = result
    return result


def prioritized_actions():
    issues = build_issues()

    actions = []
    seen = set()

    for issue in issues:
        title = issue["title"]
        if title not in seen:
            seen.add(title)
            actions.append({
                "severity": issue["severity"],
                "title": title,
                "detail": issue["detail"],
            })

    if not actions:
        actions.append({
            "severity": "complete",
            "title": "Review your final application package",
            "detail": (
                "No unresolved issues were detected by this MVP. Reverify time-sensitive "
                "requirements against the official German Missions sources before submission."
            ),
        })

    return actions[:5]


# ============================================================
# 8. GROQ / RAG
# ============================================================

RAG_SYSTEM_PROMPT = """
You are Viza Pilot, an AI assistant for Pakistani applicants preparing a
German student visa application.

The structured Visa Readiness Engine, not you, determines checklist status,
document state, consistency issues and readiness score.

Use ONLY the supplied retrieved visa-guide context for factual visa
requirements.

Rules:
1. Do not invent official visa requirements.
2. If the retrieved context is insufficient, say:
   "I could not verify that from the supplied Germany visa guide."
3. Do not guarantee approval.
4. Never describe the readiness score as a visa-grant probability.
5. When explaining a requirement, mention the supplied guide page where useful.
6. Clearly distinguish an official/verified requirement from general organisational advice.
7. Remind the user that time-sensitive requirements must be checked against current German Missions sources.
8. Do not fabricate applicant facts.
9. Be practical, concise and supportive.
"""


def ask_viza_pilot(question):
    results = retrieve_knowledge(question)

    context = "\n\n".join(
        f"[Page {x['page']}] {x['title']}\n{x['text']}"
        for x in results
    )

    readiness = calculate_readiness()
    actions = prioritized_actions()

    app_context = {
        "profile": st.session_state.profile,
        "readiness_score": readiness["score"],
        "blocking_issues": readiness["blocking_count"],
        "top_actions": [a["title"] for a in actions],
    }

    if not groq_is_configured():
        return (
            "Groq is not connected yet. Add `GROQ_API_KEY` in Streamlit Secrets. "
            "The deterministic checklist and readiness engine will still work without Groq."
        ), results

    client = get_groq_client()
    model_name = get_secret("GROQ_MODEL") or DEFAULT_GROQ_MODEL

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"Current application context:\n{json.dumps(app_context, default=str)}",
        },
        {
            "role": "user",
            "content": (
                f"RETRIEVED VISA GUIDE CONTEXT:\n{context}\n\n"
                f"USER QUESTION:\n{question}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=900,
        include_reasoning=False,
    )

    return response.choices[0].message.content, results


# ============================================================
# 9. UI COMPONENTS
# ============================================================

def app_header():
    st.title("✈️ Viza Pilot")
    st.caption(
        "Germany Student Visa Readiness MVP for Pakistani applicants"
    )

    if groq_is_configured():
        st.success("Groq AI connected", icon="✅")
    else:
        st.info(
            "Groq is not connected yet. The readiness engine still works; "
            "add GROQ_API_KEY in Streamlit Secrets for AI explanations."
        )


def sidebar():
    with st.sidebar:
        st.markdown("## Viza Pilot")

        profile = st.session_state.profile
        st.caption("Current application")
        st.write("🇵🇰 Pakistan → 🇩🇪 Germany")
        st.write("🎓 Student Visa")

        st.divider()

        nav = [
            ("🏠 Dashboard", "dashboard"),
            ("👤 Applicant profile", "profile"),
            ("📋 Checklist", "checklist"),
            ("📄 Upload documents", "upload"),
            ("🔍 Review extracted data", "review"),
            ("⚠️ Issue center", "issues"),
            ("📊 Readiness", "readiness"),
            ("➡️ What to do next", "next_actions"),
            ("💬 Ask Viza Pilot", "chat"),
        ]

        for label, key in nav:
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                go(key)

        st.divider()

        if st.button("Reset application", use_container_width=True):
            reset_application()
            st.rerun()

        st.caption(
            "Readiness measures preparation completeness, not visa approval probability."
        )


def severity_badge(severity):
    if severity == "blocking":
        return "🔴 BLOCKING"
    if severity == "review":
        return "🟠 REVIEW"
    if severity == "action":
        return "🟡 ACTION"
    return "🟢 COMPLETE"


def doc_type_label(doc_type):
    labels = {
        "passport": "Passport",
        "cnic": "CNIC",
        "admission_letter": "Admission letter",
        "degree_certificate": "Degree certificate",
        "transcript": "Transcript",
        "tuition_payment": "Tuition payment evidence",
        "blocked_account": "Blocked-account evidence",
        "deed_of_obligation": "Deed of obligation",
        "scholarship": "Scholarship evidence",
        "health_insurance": "Health insurance",
        "accommodation": "Accommodation proof",
        "portal_application": "Portal application",
        "appointment_confirmation": "Appointment confirmation",
        "unknown": "Unclassified",
    }
    return labels.get(doc_type, doc_type.replace("_", " ").title())


# ============================================================
# 10. SCREENS
# ============================================================

def screen_dashboard():
    st.header("Application dashboard")

    readiness = calculate_readiness()
    actions = prioritized_actions()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readiness", f"{readiness['score']}%")
    c2.metric(
        "Required documents",
        f"{readiness['required_complete']}/{readiness['required_total']}",
    )
    c3.metric("Blocking issues", readiness["blocking_count"])
    c4.metric("Documents uploaded", len(st.session_state.documents))

    st.progress(readiness["score"] / 100)

    st.warning(
        "The readiness score measures how complete and internally consistent "
        "your preparation appears. It is not a probability of visa approval."
    )

    st.subheader("What should I do next?")

    for idx, action in enumerate(actions[:3], start=1):
        st.markdown(
            f"**{idx}. {action['title']}**  \n"
            f"{severity_badge(action['severity'])}  \n"
            f"{action['detail']}"
        )
        st.divider()

    if not st.session_state.profile.get("profile_confirmed"):
        if st.button("Start with applicant profile", type="primary"):
            go("profile")
    elif readiness["required_complete"] < readiness["required_total"]:
        if st.button("Continue application", type="primary"):
            go("checklist")
    else:
        if st.button("Review readiness", type="primary"):
            go("readiness")


def screen_profile():
    st.header("Applicant profile")
    st.write(
        "Tell Viza Pilot about the applicant. These answers determine which requirements apply."
    )

    p = st.session_state.profile

    with st.form("profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input(
                "Full name (as on passport)",
                value=p.get("full_name", ""),
            )
            dob = st.text_input(
                "Date of birth",
                value=p.get("dob", ""),
                placeholder="e.g. 12/03/1998",
            )
            passport_number = st.text_input(
                "Passport number",
                value=p.get("passport_number", ""),
            )
            study_level = st.selectbox(
                "Study level",
                ["", "Bachelor's", "Master's", "PhD", "Other"],
                index=["", "Bachelor's", "Master's", "PhD", "Other"].index(
                    p.get("study_level", "")
                )
                if p.get("study_level", "") in ["", "Bachelor's", "Master's", "PhD", "Other"]
                else 0,
            )
            university = st.text_input(
                "German university/institution",
                value=p.get("university", ""),
            )

        with col2:
            course = st.text_input(
                "Course/programme",
                value=p.get("course", ""),
            )
            admission_received = st.checkbox(
                "I have received admission",
                value=p.get("admission_received", False),
            )
            funding_method = st.selectbox(
                "Main funding method",
                [
                    "",
                    "Blocked account",
                    "Deed of obligation",
                    "Official scholarship",
                    "Other / not sure",
                ],
                index=[
                    "",
                    "Blocked account",
                    "Deed of obligation",
                    "Official scholarship",
                    "Other / not sure",
                ].index(p.get("funding_method", ""))
                if p.get("funding_method", "") in [
                    "",
                    "Blocked account",
                    "Deed of obligation",
                    "Official scholarship",
                    "Other / not sure",
                ]
                else 0,
            )
            tuition_applicable = st.checkbox(
                "My course requires tuition-fee payment evidence",
                value=p.get("tuition_applicable", False),
            )
            tuition_paid = st.checkbox(
                "I have paid the applicable tuition fee",
                value=p.get("tuition_paid", False),
                disabled=not tuition_applicable,
            )
            appointment_booked = st.checkbox(
                "I have received/booked my visa appointment",
                value=p.get("appointment_booked", False),
            )
            accommodation_arranged = st.checkbox(
                "I have arranged accommodation/housing",
                value=p.get("accommodation_arranged", False),
            )
            insurance_arranged = st.checkbox(
                "I have arranged health insurance",
                value=p.get("insurance_arranged", False),
            )

        submitted = st.form_submit_button(
            "Save and confirm profile",
            type="primary",
        )

    if submitted:
        st.session_state.profile.update({
            "full_name": full_name,
            "dob": dob,
            "passport_number": passport_number,
            "study_level": study_level,
            "university": university,
            "course": course,
            "admission_received": admission_received,
            "funding_method": funding_method,
            "tuition_applicable": tuition_applicable,
            "tuition_paid": tuition_paid if tuition_applicable else False,
            "appointment_booked": appointment_booked,
            "accommodation_arranged": accommodation_arranged,
            "insurance_arranged": insurance_arranged,
            "profile_confirmed": True,
        })
        st.success("Applicant profile saved.")
        st.button(
            "Continue to checklist",
            type="primary",
            on_click=lambda: setattr(st.session_state, "screen", "checklist"),
        )


def screen_checklist():
    st.header("Personalized verified checklist")
    st.write(
        "This checklist is generated from the structured Germany student-visa requirements "
        "currently included in the MVP."
    )

    st.info(
        "Production rule: every requirement should be reverified against the current "
        "official German Missions source before being marked as current."
    )

    reqs = personalized_requirements()

    categories = []
    for req in reqs:
        if req["category"] not in categories:
            categories.append(req["category"])

    for category in categories:
        st.subheader(category)
        for req in [r for r in reqs if r["category"] == category]:
            status = requirement_status(req)

            if status == "complete":
                icon = "✅"
                status_text = "Complete"
            elif status == "not_applicable":
                icon = "➖"
                status_text = "Not applicable"
            else:
                icon = "❌"
                status_text = "Missing"

            st.markdown(f"### {icon} {req['title']}")
            st.write(req["description"])
            st.caption(
                f"Status: {status_text} · Guide page {req['source_page']} · "
                f"{req['source_label']}"
            )

            if status == "missing":
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button(
                        "Mark as already available",
                        key=f"manual_complete_{req['id']}",
                    ):
                        st.session_state.manual_requirement_states[req["id"]] = "complete"
                        st.rerun()
                with col2:
                    if not req["required"] and st.button(
                        "Mark not applicable",
                        key=f"manual_na_{req['id']}",
                    ):
                        st.session_state.manual_requirement_states[req["id"]] = "not_applicable"
                        st.rerun()

            st.divider()

    if st.button("Upload documents", type="primary"):
        go("upload")


def screen_upload():
    st.header("Secure document upload — prototype")
    st.write(
        "Upload PDF or TXT documents for classification and field extraction."
    )

    st.warning(
        "This MVP processes files in the current Streamlit session only. "
        "It is not yet a production document vault."
    )

    uploads = st.file_uploader(
        "Upload application documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploads and st.button("Analyze uploaded documents", type="primary"):
        existing_ids = {d["id"] for d in st.session_state.documents}

        with st.spinner("Analyzing documents..."):
            for uploaded in uploads:
                result = analyze_uploaded_file(uploaded)
                if result["id"] not in existing_ids:
                    st.session_state.documents.append(result)

        st.success("Document analysis complete.")
        go("review")

    if st.session_state.documents:
        st.subheader("Already analyzed")
        for doc in st.session_state.documents:
            st.write(
                f"• **{doc['filename']}** → {doc_type_label(doc['doc_type'])}"
            )


def screen_review():
    st.header("Review AI-extracted document data")

    if not st.session_state.documents:
        st.info("No documents have been analyzed yet.")
        if st.button("Upload documents", type="primary"):
            go("upload")
        return

    st.write(
        "AI-extracted data must be confirmed by the applicant before Viza Pilot relies on it."
    )

    for idx, doc in enumerate(st.session_state.documents):
        with st.expander(
            f"{'✅' if doc['confirmed'] else '🟠'} "
            f"{doc['filename']} — {doc_type_label(doc['doc_type'])}",
            expanded=not doc["confirmed"],
        ):
            doc_types = list(DOC_TYPE_HINTS.keys()) + ["unknown"]
            selected_type = st.selectbox(
                "Document type",
                doc_types,
                index=doc_types.index(doc["doc_type"])
                if doc["doc_type"] in doc_types
                else doc_types.index("unknown"),
                format_func=doc_type_label,
                key=f"doc_type_{doc['id']}",
            )
            doc["doc_type"] = selected_type

            st.caption(f"Classification confidence: {doc['confidence']}")

            extracted = doc.setdefault("extracted", {})

            editable_keys = [
                ("name", "Name"),
                ("passport_number", "Passport number"),
                ("date_of_birth", "Date of birth"),
                ("expiry_date", "Expiry date"),
                ("possible_institution", "Institution"),
                ("possible_amount", "Financial amount"),
            ]

            for key, label in editable_keys:
                if key in extracted or key in {
                    "name",
                    "passport_number",
                }:
                    extracted[key] = st.text_input(
                        label,
                        value=extracted.get(key, ""),
                        key=f"{doc['id']}_{key}",
                    )

            if doc["warnings"]:
                for warning in doc["warnings"]:
                    st.warning(warning)

            if st.checkbox(
                "I confirm the extracted information above is correct",
                value=doc["confirmed"],
                key=f"confirm_{doc['id']}",
            ):
                doc["confirmed"] = True
                if doc["doc_type"] != "unknown":
                    st.session_state.confirmed_documents[doc["doc_type"]] = True
            else:
                doc["confirmed"] = False

    st.session_state.documents = st.session_state.documents

    if st.button("Run consistency checks", type="primary"):
        build_issues()
        go("issues")


def screen_issues():
    st.header("Issue center")

    issues = build_issues()

    if not issues:
        st.success("No unresolved issues were detected by the MVP checks.")
        st.info(
            "This does not guarantee visa approval. Reverify time-sensitive requirements before submission."
        )
        return

    counts = {
        "blocking": sum(i["severity"] == "blocking" for i in issues),
        "review": sum(i["severity"] == "review" for i in issues),
        "action": sum(i["severity"] == "action" for i in issues),
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Blocking", counts["blocking"])
    c2.metric("Review", counts["review"])
    c3.metric("Action", counts["action"])

    for issue in issues:
        st.markdown(f"### {severity_badge(issue['severity'])} — {issue['title']}")
        st.write(issue["detail"])
        if issue.get("source_page"):
            st.caption(f"Source: Germany Visa Guide, page {issue['source_page']}")
        st.divider()

    if st.button("Calculate readiness", type="primary"):
        go("readiness")


def screen_readiness():
    st.header("Application readiness")

    result = calculate_readiness()

    st.metric("Readiness score", f"{result['score']} / 100")
    st.progress(result["score"] / 100)

    st.warning(
        "This score measures preparation completeness and internal consistency. "
        "It is NOT a prediction or probability of visa approval."
    )

    st.subheader("Score components")

    cols = st.columns(4)
    cols[0].metric(
        "Mandatory requirements",
        f"{result['components']['mandatory_requirements']}%",
    )
    cols[1].metric(
        "Confirmed extraction",
        f"{result['components']['confirmed_extraction']}%",
    )
    cols[2].metric(
        "Consistency",
        f"{result['components']['consistency']}%",
    )
    cols[3].metric(
        "Applicant profile",
        f"{result['components']['profile']}%",
    )

    st.subheader("Readiness summary")
    st.write(
        f"**Required checklist items:** "
        f"{result['required_complete']} of {result['required_total']} complete"
    )
    st.write(f"**Blocking issues:** {result['blocking_count']}")
    st.write(f"**Items requiring review:** {result['review_count']}")

    if result["blocking_count"] > 0:
        st.error(
            "Your application still has blocking issues. Address these before treating the file as submission-ready."
        )
    elif result["score"] >= 90:
        st.success(
            "Your application appears highly complete within the checks implemented by this MVP."
        )
    else:
        st.info(
            "No blocking inconsistency is currently detected, but additional preparation may still be needed."
        )

    if st.button("Show me what to do next", type="primary"):
        go("next_actions")


def screen_next_actions():
    st.header("What should I do next?")

    actions = prioritized_actions()

    for idx, action in enumerate(actions, start=1):
        st.markdown(
            f"## {idx}. {action['title']}\n"
            f"{severity_badge(action['severity'])}"
        )
        st.write(action["detail"])
        st.divider()

    st.info(
        "Before final submission, recheck time-sensitive requirements against the current "
        "German Missions in Pakistan guidance."
    )

    if st.button("Ask Viza Pilot about these actions", type="primary"):
        go("chat")


def screen_chat():
    st.header("Ask Viza Pilot")
    st.write(
        "Ask questions about the Germany student visa guide or your current readiness issues."
    )

    if not st.session_state.chat:
        st.session_state.chat.append({
            "role": "assistant",
            "content": (
                "I can explain requirements, financial evidence, missing documents, "
                "or the issues currently affecting your readiness."
            ),
        })

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Visa-guide evidence used"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**Page {src['page']} — {src['title']}**"
                        )
                        st.write(src["text"])

    question = st.chat_input("Ask Viza Pilot...")

    if question:
        st.session_state.chat.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking the visa guide and your application..."):
                    answer, sources = ask_viza_pilot(question)

                st.markdown(answer)

                with st.expander("Visa-guide evidence used"):
                    for src in sources:
                        st.markdown(
                            f"**Page {src['page']} — {src['title']}**"
                        )
                        st.write(src["text"])

                st.session_state.chat.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as exc:
                error = (
                    "I could not generate the AI explanation right now. "
                    f"The readiness engine still works. Details: {type(exc).__name__}"
                )
                st.error(error)
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": error,
                })


# ============================================================
# 11. START APP
# ============================================================

init_state()
app_header()
sidebar()
st.divider()

screens = {
    "dashboard": screen_dashboard,
    "profile": screen_profile,
    "checklist": screen_checklist,
    "upload": screen_upload,
    "review": screen_review,
    "issues": screen_issues,
    "readiness": screen_readiness,
    "next_actions": screen_next_actions,
    "chat": screen_chat,
}

screens.get(st.session_state.screen, screen_dashboard)()
