import re

# ---------- BLOCKLISTS ----------
BLOCKED_PATTERNS = [
    r"ignore (all|previous|above) instructions",
    r"you are now",
    r"system prompt",
    r"jailbreak",
    r"disregard.*instructions",
    r"do not follow.*rules",
]

PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE_IN": r"\b[6-9]\d{9}\b",
    "PHONE_US": r"\b\d{3}[-. ]\d{3}[-. ]\d{4}\b",
    "AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
}

TOXIC_WORDS = ["kill", "bomb", "hack into", "steal password", "make weapon"]

ALLOWED_TOPICS = ["rag", "retrieval", "guardrail", "document", "embedding", "vector", "llm", "ai", "project"]

class GuardrailViolation(Exception):
    pass

def check_input_guardrails(query: str) -> str:
    """Runs BEFORE retrieval + LLM. Returns cleaned query or raises."""
    if not query or len(query.strip()) < 3:
        raise GuardrailViolation("Query is too short.")

    if len(query) > 2000:
        raise GuardrailViolation("Query too long. Keep it under 2000 chars.")

    q_lower = query.lower()

    # 1. Prompt Injection
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, q_lower):
            raise GuardrailViolation("Prompt injection detected. Blocked.")

    # 2. Toxic / Harmful Intent
    for word in TOXIC_WORDS:
        if word in q_lower:
            raise GuardrailViolation(f"Harmful request blocked: '{word}'")

    # 3. PII Masking (don't send PII to LLM)
    cleaned = query
    for name, pat in PII_PATTERNS.items():
        cleaned = re.sub(pat, f"[REDACTED_{name}]", cleaned)

    return cleaned

def check_output_guardrails(answer: str, contexts: list[str]) -> str:
    """Runs AFTER LLM. Checks hallucination, PII leak, refusal."""
    if not answer or len(answer.strip()) < 5:
        return "I don't know based on the provided documents."

    # 1. PII Leak in Output
    for name, pat in PII_PATTERNS.items():
        if re.search(pat, answer):
            answer = re.sub(pat, f"[REDACTED_{name}]", answer)

    # 2. Hallucination Check - Simple Groundedness
    # If answer has no overlap with context, flag it
    # This is a cheap heuristic. RAGAS does this properly in evaluate.py
    answer_words = set(re.findall(r"\w+", answer.lower()))
    context_text = " ".join(contexts).lower()
    context_words = set(re.findall(r"\w+", context_text))

    if len(answer_words) > 0:
        overlap = len(answer_words & context_words) / len(answer_words)
        if overlap < 0.15 and "don't know" not in answer.lower():
            return "I don't know based on the provided documents. The retrieved context doesn't support a confident answer."

    # 3. Refusal / Empty guard
    if "as an ai" in answer.lower() and len(answer) < 100:
        return "I don't know based on the provided documents."

    return answer

def check_relevance(query: str, contexts: list[str]) -> bool:
    """If no relevant docs found, don't hallucinate."""
    if not contexts:
        return False
    return True