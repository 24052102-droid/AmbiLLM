import ollama


# Model name in one place — easy to swap
_MODEL = "llama3"


def generate_sentences(prompt: str, count: int = 10) -> list[str]:
    """
    Call the local Llama3 model *count* times and return the responses.

    Each call uses a slightly varied system instruction so the model is less
    likely to produce structurally identical sentences across iterations.

    Returns only non-empty, stripped strings.
    """
    sentences = []

    variation_hints = [
        "Be creative and unpredictable.",
        "Use surprising or unusual entity combinations.",
        "Pick obscure but plausible organizations.",
        "Try a different continent for the location.",
        "Use a historical figure if appropriate.",
        "Keep the sentence under 25 words.",
        "Experiment with sentence structure.",
        "Avoid any entities used in previous sentences.",
        "Use an entity from a non-English-speaking country.",
        "Make the sentence sound like a news headline.",
    ]

    for i in range(count):
        # Rotate through variation hints to encourage diversity
        hint = variation_hints[i % len(variation_hints)]

        try:
            response = ollama.chat(
                model=_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise sentence generator. "
                            f"{hint} "
                            "Return ONLY the single sentence — no explanation, "
                            "no notes, no bullet points."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    # Slightly raise temperature for variety; still coherent
                    "temperature": 0.9,
                    "top_p": 0.95,
                }
            )

            text = response["message"]["content"].strip()

            if text:
                sentences.append(text)

        except ollama.ResponseError as e:
            print(f"[LLM ERROR] ollama.ResponseError on call {i + 1}: {e}")

        except Exception as e:
            print(f"[LLM ERROR] Unexpected error on call {i + 1}: {e}")

    return sentences