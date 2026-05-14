import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Apply Unicode NFKD normalization to *text*.
    This converts compatibility characters (e.g. ligatures, full-width letters)
    to their canonical equivalents and makes downstream comparisons more robust.
    """
    return unicodedata.normalize("NFKD", text)


def strip_whitespace(text: str) -> str:
    """Collapse multiple internal spaces and strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def clean_sentence(text: str) -> str:
    """
    Apply all cleaning steps to a raw sentence string:
      1. Unicode normalisation
      2. Whitespace normalisation
      3. Remove stray newlines / tabs

    This is the function callers should generally use.
    """
    text = normalize_text(text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = strip_whitespace(text)
    return text