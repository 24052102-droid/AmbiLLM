import json
import os
import tempfile

CACHE_FILE = "datasets/entity_cache.json"


def load_cache() -> dict:
    """Load the entity cache from disk. Returns an empty dict if missing/corrupt."""
    if not os.path.exists(CACHE_FILE):
        return {}

    with open(CACHE_FILE, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except (json.JSONDecodeError, ValueError):
            # Cache file is corrupt — start fresh rather than crashing
            print(
                f"[CACHE] Warning: {CACHE_FILE} is corrupt or empty. "
                "Starting with a fresh cache."
            )
            return {}


def save_cache(cache: dict) -> None:
    """
    Persist the entity cache to disk using an atomic write.

    Writing to a temp file first and then renaming prevents a crash mid-write
    from leaving a corrupt cache file behind.
    """
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)

    dir_name = os.path.dirname(CACHE_FILE) or "."

    # Write to a temporary file in the same directory, then atomically replace
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=dir_name,
        delete=False,
        suffix=".tmp"
    ) as tmp:
        json.dump(cache, tmp, indent=4, ensure_ascii=False)
        tmp_path = tmp.name

    os.replace(tmp_path, CACHE_FILE)


def clear_cache() -> None:
    """Delete the cache file from disk and reset to a clean state."""
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"[CACHE] Cleared: {CACHE_FILE}")