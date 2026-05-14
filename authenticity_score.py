def calculate_authenticity_score(
    real_entities: int,
    total_entities: int,
    unknown_entities: int = 0
) -> dict:
    """
    Calculate an authenticity score for a sentence's entities.

    Parameters
    ----------
    real_entities   : count of entities confirmed to exist (exists == True)
    total_entities  : total number of entities extracted
    unknown_entities: count of entities that could not be verified
                      (exists == "UNKNOWN") — previously silently ignored,
                      which caused misleadingly low / zero scores.

    Returns
    -------
    A dict with:
      - score         : ratio of real entities to (total − unknown),
                        i.e. only over the entities we could actually judge.
      - coverage      : ratio of entities we *could* judge (total − unknown)
                        to the total extracted, indicating result reliability.
      - raw_counts    : breakdown for transparency / debugging.
    """
    if total_entities == 0:
        return {
            "score": 0.0,
            "coverage": 0.0,
            "raw_counts": {
                "real": 0,
                "hallucinated": 0,
                "unknown": 0,
                "total": 0
            }
        }

    judged = total_entities - unknown_entities
    hallucinated = total_entities - real_entities - unknown_entities

    # Score over judged entities only to avoid penalising transient API errors
    score = round(real_entities / judged, 2) if judged > 0 else 0.0

    # Coverage tells us how trustworthy the score is
    coverage = round(judged / total_entities, 2)

    return {
        "score": score,
        "coverage": coverage,
        "raw_counts": {
            "real": real_entities,
            "hallucinated": hallucinated,
            "unknown": unknown_entities,
            "total": total_entities
        }
    }