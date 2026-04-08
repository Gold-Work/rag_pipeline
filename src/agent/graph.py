"""LangGraph graph for the e-commerce SAV agent.

Graph layout
------------
START → decide → execute → respond → END
                ↘ (tool="none")  ↗
                       respond

Nodes
-----
- decide  : LLM picks the right tool (or none) given the conversation.
- execute : Dispatches to the chosen tool and stores the result in state.
- respond : Generates a final French-language reply from accumulated context.
"""

import json
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from src.agent.state import AgentState
from src.agent.tools import check_order, create_ticket, query_rag, send_email
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("agent.graph")
config = get_config()


# ---------------------------------------------------------------------------
# Structured output schema for the decide node
# ---------------------------------------------------------------------------


class ToolChoice(BaseModel):
    """Decision produced by the decide node.

    Attributes
    ----------
    tool : str
        Which tool to call, or "none" to answer directly.
    argument : str
        Single string argument passed to the tool.
        - query_rag      → the question to search
        - check_order    → the order reference (e.g. "CMD-2025-00142")
        - send_email     → JSON {"to": ..., "subject": ..., "body": ...}
        - create_ticket  → JSON {"issue": ..., "customer_email": ...}
        - none           → empty string
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: Literal["query_rag", "check_order", "send_email", "create_ticket", "none"]
    argument: str


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_DECIDE_SYSTEM = """Tu es un agent SAV pour ModaMaison.fr, une boutique e-commerce française de mode.

Tu dois choisir l'outil le plus adapté pour répondre au dernier message du client.

Outils disponibles :
- query_rag      : recherche dans la base documentaire (politique retours, livraisons, FAQ, guide tailles).
                   Utilise-le pour toute question générale sur les politiques, délais, procédures.
- check_order    : consulte le statut d'une commande à partir de sa référence (format CMD-XXXX-XXXXX).
                   Utilise-le dès qu'une référence de commande est mentionnée.
- send_email     : envoie un e-mail au client (confirmation, document, récapitulatif).
                   Argument : JSON {"to": "...", "subject": "...", "body": "..."}.
- create_ticket  : crée un ticket de support pour les problèmes non résolus automatiquement.
                   Argument : JSON {"issue": "...", "customer_email": "..."}.
- none           : réponds directement, sans outil (salutation, question hors périmètre, clarification).

Règles de décision :
1. Si le client mentionne une référence de commande → check_order en priorité.
2. Si la question porte sur retours, livraisons, tailles, FAQ → query_rag.
3. Si le problème ne peut pas être résolu avec les autres outils → create_ticket.
4. Pour les salutations ou questions hors périmètre → none.
"""

_RESPOND_SYSTEM = """Tu es un agent SAV pour ModaMaison.fr, une boutique e-commerce française de mode.

Réponds en français, avec courtoisie, précision et concision.
Utilise les informations fournies (résultats d'outils, contexte documentaire) pour répondre au client.
Ne révèle pas les détails techniques internes (noms d'outils, JSON brut, logs).
Si tu manques d'informations pour résoudre le problème, propose d'escalader vers un conseiller humain.
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def decide_node(state: AgentState) -> dict[str, Any]:
    """LLM reads the conversation and decides which tool to call (or none)."""
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore[arg-type]
    ).with_structured_output(ToolChoice)

    messages = [SystemMessage(content=_DECIDE_SYSTEM)] + list(state["messages"])
    choice: ToolChoice = llm.invoke(messages)  # type: ignore[assignment]

    logger.info(f"[decide] tool={choice.tool!r} argument={choice.argument[:80]!r}")

    # Serialise the decision into an AIMessage so _route and execute_node can read it
    return {"messages": [AIMessage(content=json.dumps({"tool": choice.tool, "argument": choice.argument}))]}


def execute_node(state: AgentState) -> dict[str, Any]:
    """Reads the tool choice from the last AIMessage and dispatches to the tool."""
    last_ai = state["messages"][-1]
    choice = json.loads(last_ai.content)
    tool_name: str = choice["tool"]
    argument: str = choice["argument"]

    logger.info(f"[execute] dispatching tool={tool_name!r}")

    if tool_name == "query_rag":
        result = query_rag(argument)
    elif tool_name == "check_order":
        result = check_order(argument)
    elif tool_name == "send_email":
        result = send_email(**json.loads(argument))
    elif tool_name == "create_ticket":
        result = create_ticket(**json.loads(argument))
    else:
        result = {"status": "error", "data": {"message": f"Unknown tool: {tool_name}"}}

    # Build state updates
    updates: dict[str, Any] = {
        "tool_results": state["tool_results"] + [{"tool": tool_name, **result}],
        "messages": [ToolMessage(content=json.dumps(result), tool_call_id=tool_name)],
    }

    # Side-effects: propagate useful data into dedicated state fields
    if tool_name == "query_rag" and result["status"] == "ok":
        updates["rag_context"] = result["data"]["chunks"]

    if tool_name == "check_order" and result["status"] == "ok":
        updates["order_id"] = result["data"]["order_id"]

    return updates


def respond_node(state: AgentState) -> dict[str, Any]:
    """Generates the final French-language reply using all accumulated context."""
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
        api_key=os.getenv("OPENAI_API_KEY"),  # type: ignore[arg-type]
    )

    # Build a context block with tool results and RAG chunks
    context_parts: list[str] = []

    if state["rag_context"]:
        rag_block = "\n\n---\n\n".join(state["rag_context"])
        context_parts.append(f"[Informations issues de la base documentaire]\n{rag_block}")

    for entry in state["tool_results"]:
        tool = entry.get("tool", "?")
        status = entry.get("status", "?")
        data = entry.get("data", {})
        if tool == "check_order" and status == "ok":
            context_parts.append(
                f"[Commande {data.get('order_id')}]\n"
                f"Statut : {data.get('status_label')}\n"
                f"Transporteur : {data.get('carrier')} — Suivi : {data.get('tracking') or 'non disponible'}\n"
                f"Adresse : {data.get('delivery_address')}\n"
                f"Délai de retour : {data.get('return_deadline') or 'non applicable'}"
            )
        elif tool == "create_ticket" and status == "ok":
            context_parts.append(
                f"[Ticket créé]\nRéférence : {data.get('ticket_id')} — Priorité : {data.get('priority')}"
            )
        elif tool == "send_email" and status == "ok":
            context_parts.append(f"[E-mail envoyé à {data.get('to')}]\nObjet : {data.get('subject')}")
        elif status in ("error", "not_found"):
            context_parts.append(f"[Résultat {tool}] {data.get('message', 'Erreur inconnue')}")

    # Conversation history (exclude internal AIMessages carrying raw JSON decisions)
    history = [
        m
        for m in state["messages"]
        if not (isinstance(m, AIMessage) and m.content.startswith('{"tool":')) and not isinstance(m, ToolMessage)
    ]

    messages = [SystemMessage(content=_RESPOND_SYSTEM)]
    messages.extend(history)
    if context_parts:
        messages.append(HumanMessage(content="Contexte disponible :\n\n" + "\n\n".join(context_parts)))

    response = llm.invoke(messages)
    logger.info(f"[respond] réponse générée ({len(response.content)} chars)")

    return {"messages": [AIMessage(content=response.content)], "resolved": True}


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _route(state: AgentState) -> str:
    """Read the tool choice serialised in the last AIMessage."""
    last = state["messages"][-1]
    choice = json.loads(last.content)
    return choice["tool"]  # "query_rag" | "check_order" | "send_email" | "create_ticket" | "none"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph():
    """Compile and return the SAV agent graph.

    Returns
    -------
    CompiledStateGraph
        Ready-to-invoke LangGraph graph. Call with::

            graph = build_graph()
            result = graph.invoke({
                "messages": [HumanMessage(content="...")],
                "customer_email": "client@example.com",
                "order_id": None,
                "rag_context": [],
                "tool_results": [],
                "pending_ticket": None,
                "resolved": False,
            })
    """
    graph = StateGraph(AgentState)

    graph.add_node("decide", decide_node)
    graph.add_node("execute", execute_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        _route,
        {
            "query_rag": "execute",
            "check_order": "execute",
            "send_email": "execute",
            "create_ticket": "execute",
            "none": "respond",
        },
    )
    graph.add_edge("execute", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
