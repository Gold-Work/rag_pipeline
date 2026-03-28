import os

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


def generate(prompt: str) -> str:
    """Génère une réponse LLM à partir du prompt fourni."""
    with langfuse_generation("llm-generation", model=model, input=prompt) as out:
        try:
            logger.info(f"🤖 Génération LLM (model={model})")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            out["output"] = answer
            return answer
        except Exception as e:
            logger.warning(f"⚠️ Génération échouée : {e}")
            out["output"] = f"error: {e}"
            return "Désolé, impossible de générer une réponse pour le moment."
