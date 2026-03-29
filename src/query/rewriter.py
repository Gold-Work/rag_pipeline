import os
from openai import OpenAI
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger("rewriter")
config = get_config()

enabled = config["rewriter"].get("enabled", True)
n_variants = config["rewriter"]["n_variants"]
temperature = config["rewriter"]["temperature"]
model = config["llm"]["model"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REWRITE_PROMPT = config["rewriter"]["system_prompt"]


def rewrite_query(question: str) -> list[str]:
    if not enabled:
        return [question]

    logger.info(f"✏️  Réécriture de la query : '{question}'")
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=200,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(n=n_variants, question=question)}],
        )

        raw = (response.choices[0].message.content or "").strip()
        variants = [line.strip("-•1234567890. ").strip() for line in raw.split("\n") if line.strip()]
        variants = variants[:n_variants]

        # Déduplication et ajout question originale
        all_queries = list(dict.fromkeys([question] + variants))

        logger.info(f"📝 {len(all_queries)} queries générées : {all_queries}")
        return all_queries
    except Exception as e:
        logger.warning(f"⚠️ Query rewriting échoué : {e} — utilisation query originale")
        return [question]
