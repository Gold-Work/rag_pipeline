import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import get_logger
from src.query.retriever import retrieve
from src.query.augmenter import build_prompt
from src.query.generator import generate
from src.query.reranker import rerank

logger = get_logger("run_query")

def run_query(question: str, k: int = 5) -> str:
    logger.info("=" * 60)
    logger.info(f"❓ QUESTION : {question}")
    logger.info("=" * 60)

    # ÉTAPE 1 — Retrieve (top 20 pour le reranker)
    logger.info("🔍 ÉTAPE 1 : Recherche des chunks pertinents...")
    documents = retrieve(question, k=20)

    if not documents:
        logger.warning("⚠️  Aucun document pertinent trouvé")
        return "Je n'ai pas trouvé d'information pertinente pour répondre à cette question."

    # ÉTAPE 2 — Reranking
    logger.info("🎯 ÉTAPE 2 : Reranking...")
    documents = rerank(question, documents, top_k=5)

    # ÉTAPE 3 — Augment
    logger.info("📝 ÉTAPE 3 : Construction du prompt...")
    prompt = build_prompt(question, documents)

    # ÉTAPE 4 — Generate
    logger.info("🤖 ÉTAPE 4 : Génération de la réponse...")
    answer = generate(prompt)

    logger.info("=" * 60)
    logger.info("✅ RÉPONSE FINALE :")
    logger.info(f"\n{answer}\n")
    logger.info("=" * 60)

    return answer


def interactive_mode():
    logger.info("💬 MODE INTERACTIF — tape 'exit' pour quitter\\n")

    while True:
        try:
            question = input("\\n❓ Ta question : ").strip()

            if question.lower() in ["exit", "quit", "q"]:
                logger.info("👋 Au revoir !")
                break

            if not question:
                print("⚠️  Question vide, réessaie.")
                continue

            answer = run_query(question)
            print(f"\\n💡 RÉPONSE :\\n{answer}\\n")

        except KeyboardInterrupt:
            logger.info("\\n👋 Interruption — Au revoir !")
            break


if __name__ == "__main__":
    # Mode direct avec argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = run_query(question)
        print(f"\\n💡 RÉPONSE :\\n{answer}\\n")
    else:
        # Mode interactif
        interactive_mode()