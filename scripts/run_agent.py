"""Run the SAV agent from the terminal.

Usage
-----
    python scripts/run_agent.py "Où est ma commande CMD-2025-00142 ?"

Examples to try
---------------
    python scripts/run_agent.py "Où est ma commande CMD-2025-00142 ?"
    python scripts/run_agent.py "Je veux un remboursement"
    python scripts/run_agent.py "Vous livrez en Belgique ?"
    python scripts/run_agent.py "Quelle taille choisir pour un pull ?"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from src.agent.graph import build_graph

if len(sys.argv) < 2:
    print('Usage: python scripts/run_agent.py "<message>"')
    sys.exit(1)

question = sys.argv[1]

graph = build_graph()

result = graph.invoke(
    {
        "messages": [HumanMessage(content=question)],
        "customer_email": "client@example.com",
        "order_id": None,
        "rag_context": [],
        "tool_results": [],
        "pending_ticket": None,
        "resolved": False,
    }
)

print(result["messages"][-1].content)
