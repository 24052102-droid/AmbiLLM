import re
import time
import requests

from utils.cache import load_cache, save_cache

cache = load_cache()

_MAX_RETRIES = 3
_BASE_DELAY = 1.5       # seconds between API calls
_SIMILARITY_THRESHOLD = 0.5  # minimum word-overlap ratio to accept a match

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_HEADERS = {
    "User-Agent": "ambg_trainer/1.0 (entity-hallucination-detector; educational)"
}


# ---------------------------------------------------------------------------
# Title similarity helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """
    Lowercase and split into alphabetic tokens, dropping stop words.
    Used for word-overlap similarity so 'the' and 'of' don't inflate scores.
    """
    STOP = {"the", "a", "an", "of", "in", "at", "for", "and", "or", "to"}
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


def _title_similarity(entity: str, title: str) -> float:
    """
    Jaccard-style word overlap between entity tokens and Wikipedia title tokens.

    Examples
    --------
    'Harvard University'  vs 'Harvard University'       → 1.0  (exact)
    'Elara Vex'           vs 'Sun Valley High School'   → 0.0  (no overlap)
    'World Health Org'    vs 'World Health Organization' → 0.67 (good)
    """
    e_tokens = _tokenize(entity)
    t_tokens = _tokenize(title)

    if not e_tokens or not t_tokens:
        return 0.0

    intersection = e_tokens & t_tokens
    union = e_tokens | t_tokens
    return len(intersection) / len(union)


def _is_acceptable_match(entity: str, matched_title: str) -> bool:
    """
    Return True only when the Wikipedia title is close enough to the
    entity we searched for.

    Acceptance rules (any one is enough):
    1. Exact match after lowercasing.
    2. Matched title *starts with* the entity name (e.g. 'Nikola Tesla' in
       'Nikola Tesla (inventor)').
    3. Entity name is fully contained in the matched title.
    4. Word-overlap Jaccard similarity >= _SIMILARITY_THRESHOLD.
    """
    e_lower = entity.lower().strip()
    t_lower = matched_title.lower().strip()

    if e_lower == t_lower:
        return True

    if t_lower.startswith(e_lower):
        return True

    if e_lower in t_lower:
        return True

    if _title_similarity(entity, matched_title) >= _SIMILARITY_THRESHOLD:
        return True

    return False


# ---------------------------------------------------------------------------
# Wikipedia API calls
# ---------------------------------------------------------------------------

def _api_get(params: dict) -> dict | None:
    """
    Single GET to the Wikipedia API with timeout.
    Returns parsed JSON or None on network error.
    """
    try:
        resp = requests.get(
            _WIKI_API,
            params=params,
            headers=_HEADERS,
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            raise   # let caller handle rate-limit retry
        return None
    except Exception:
        return None


def _fetch_extract(title: str) -> dict | None:
    """Fetch the intro extract for a Wikipedia page title."""
    data = _api_get({
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "redirects": 1,
        "titles": title,
        "format": "json",
        "utf8": 1
    })
    if not data:
        return None

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), None)

    if page is None or page.get("pageid", -1) == -1:
        return None

    return page


def _search_wikipedia(entity_name: str) -> dict:
    """
    Search Wikipedia for entity_name and validate the match quality.

    Strategy
    --------
    1. Search for up to 5 candidate titles.
    2. For each candidate (best first), check title similarity.
    3. Accept the first candidate whose title passes _is_acceptable_match.
    4. If no candidate passes, return exists=False (hallucinated).
    5. Retry the whole flow on transient network errors.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            time.sleep(_BASE_DELAY)

            # Step 1: Search
            search_data = _api_get({
                "action": "query",
                "list": "search",
                "srsearch": entity_name,
                "srlimit": 5,
                "format": "json",
                "utf8": 1
            })

            if search_data is None:
                raise ConnectionError("Empty response from Wikipedia search")

            candidates = search_data.get("query", {}).get("search", [])

            if not candidates:
                return {"exists": False, "matched_title": None, "summary": None}

            # Step 2 & 3: Walk candidates, accept first good match
            for candidate in candidates:
                c_title = candidate["title"]

                if not _is_acceptable_match(entity_name, c_title):
                    print(
                        f"  [SKIP] '{c_title}' rejected "
                        f"(similarity={_title_similarity(entity_name, c_title):.2f})"
                    )
                    continue

                # Step 4: Fetch extract for accepted candidate
                page = _fetch_extract(c_title)

                if page is None:
                    continue

                summary = (page.get("extract", "") or "")[:200]
                return {
                    "exists": True,
                    "matched_title": page.get("title", c_title),
                    "summary": summary
                }

            # No candidate passed similarity check → entity doesn't exist
            return {"exists": False, "matched_title": None, "summary": None}

        except requests.exceptions.Timeout:
            wait = _BASE_DELAY * (2 ** attempt)
            print(f"[RETRY {attempt}/{_MAX_RETRIES}] Timeout for '{entity_name}'. Waiting {wait}s …")
            time.sleep(wait)

        except requests.exceptions.ConnectionError as e:
            wait = _BASE_DELAY * (2 ** attempt)
            print(f"[RETRY {attempt}/{_MAX_RETRIES}] Connection error for '{entity_name}': {e}. Waiting {wait}s …")
            time.sleep(wait)

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = _BASE_DELAY * (2 ** attempt)
                print(f"[RETRY {attempt}/{_MAX_RETRIES}] Rate limited for '{entity_name}'. Waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"[ERROR] HTTP error for '{entity_name}': {e}")
                break

        except Exception as e:
            print(f"[ERROR] Unexpected error for '{entity_name}': {e}")
            break

    return {
        "exists": "UNKNOWN",
        "matched_title": None,
        "summary": f"Failed after {_MAX_RETRIES} attempts"
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def check_entity_exists(entity_name: str) -> dict:
    """
    Check whether entity_name exists in Wikipedia.
    Results are cached to avoid redundant API calls.
    """
    cache_key = entity_name.strip()

    if cache_key in cache:
        print(f"[CACHE HIT] {entity_name}")
        return cache[cache_key]

    print(f"[API CALL] {entity_name}")

    result = _search_wikipedia(cache_key)
    cache[cache_key] = result
    save_cache(cache)

    return result