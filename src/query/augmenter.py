from langchain_core.documents import Document
from src.utils.logger import get_logger

logger = get_logger("augmenter")

PROMPT_TEMPLATE = """CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""


def build_prompt(query: str, documents: list[Document]) -> str:

    if not documents:
        logger.warning("⚠️  Aucun document pertinent trouvé — contexte vide")
        context = "Aucun document pertinent trouvé."
    else:
        context = "\n\n---\n\n".join(
            [f"[Source: {doc.metadata.get('source_file', '?')}]\n{doc.page_content}" for doc in documents]
        )

    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    logger.info(f"📝 Prompt construit — {len(documents)} chunks | {len(prompt)} chars")
    return prompt
