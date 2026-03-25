import os
from openai import OpenAI
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger("generator")
config = get_config()

model = config["llm"]["model"]
temperature = config["llm"]["temperature"]
max_tokens = config["llm"]["max_tokens"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate(prompt: str) -> str:
    """Génère une réponse LLM à partir du prompt fourni."""
    try:
        logger.info(f"🤖 Génération LLM (model={model})")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        logger.warning(f"⚠️ Génération échouée : {e}")
        return "Désolé, impossible de générer une réponse pour le moment."