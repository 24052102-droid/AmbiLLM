import re
import spacy

from utils.text_cleaner import normalize_text

# Transformer model (better NER accuracy)
nlp = spacy.load("en_core_web_trf")

# Entity types that carry no useful ambiguity signal
IGNORE_LABELS = {
    "DATE",
    "TIME",
    "MONEY",
    "PERCENT",
    "ORDINAL",
    "CARDINAL",
    "QUANTITY",
    "LANGUAGE",
}

# Leading articles to strip so "the Marriott Hotel" → "Marriott Hotel"
_LEADING_ARTICLES = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _clean_entity_text(text: str) -> str:
    """Normalize unicode and remove leading articles."""
    text = normalize_text(text)          # unicode normalization (NFKD)
    text = _LEADING_ARTICLES.sub("", text).strip()
    return text


def extract_entities(text: str) -> list[dict]:
    """
    Extract named entities from *text*, skipping unwanted label types and
    cleaning up common noise such as leading determiners.

    Returns a list of dicts: {"entity": str, "label": str}
    De-duplicates by (cleaned_text, label) so the same entity mentioned
    twice is not checked twice.
    """
    doc = nlp(text)

    seen = set()
    entities = []

    for ent in doc.ents:
        if ent.label_ in IGNORE_LABELS:
            continue

        cleaned = _clean_entity_text(ent.text)

        if not cleaned:
            continue

        key = (cleaned.lower(), ent.label_)
        if key in seen:
            continue

        seen.add(key)
        entities.append({
            "entity": cleaned,
            "label": ent.label_
        })

    return entities