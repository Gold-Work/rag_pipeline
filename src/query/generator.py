import os
import time

import openai
from openai import OpenAI

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.observability import langfuse_generation

logger = get_logger("generator")
config = get_config()

model = config["llm"]["model"]
temperature = config["llm"]["temperature"]
max_tokens = config["llm"]["max_tokens"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_MESSAGE = (
    "Tu es un assistant expert en analyse documentaire. "
    "Réponds uniquement à partir des documents fournis dans le contexte. "
    "Si la réponse n'est pas dans le contexte, indique-le clairement. "
    "Sois précis et factuel."
)

_MAX_RETRIES = 3


def generate(prompt: str) -> str:
    """Génère une réponse LLM avec retry exponentiel sur erreurs transitoires."""
    with langfuse_generation("llm-generation", model=model, input={"system": SYSTEM_MESSAGE, "user": prompt}) as out:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                logger.info(f"🤖 Génération LLM (model={model}, tentative={attempt}/{_MAX_RETRIES})")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                answer = (response.choices[0].message.content or "").strip()
                out["output"] = answer
                return answer

            except openai.AuthenticationError:
                # Erreur permanente — inutile de retenter
                logger.error("❌ Clé OpenAI invalide — arrêt immédiat")
                raise

            except (openai.RateLimitError, openai.APITimeoutError) as e:
                wait = 2**attempt  # 2s, 4s, 8s
                if attempt < _MAX_RETRIES:
                    logger.warning(f"⚠️ Tentative {attempt}/{_MAX_RETRIES} : {e} — retry dans {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"❌ Génération abandonnée après {_MAX_RETRIES} tentatives : {e}")
                    out["output"] = f"error: {e}"
                    return "Désolé, impossible de générer une réponse pour le moment."

            except Exception as e:
                # Erreur inattendue — on propage pour que FastAPI retourne un 500
                logger.error(f"❌ Erreur inattendue lors de la génération : {e}", exc_info=True)
                out["output"] = f"error: {e}"
                raise

    return "Désolé, impossible de générer une réponse pour le moment."  # unreachable
