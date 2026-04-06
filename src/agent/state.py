"""LangGraph state definition for the e-commerce customer service agent."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State shared across all nodes of the SAV agent graph.

    Fields
    ------
    messages : list
        Conversation history (HumanMessage / AIMessage / ToolMessage).
        Uses LangGraph's add_messages reducer: new messages are appended,
        never overwritten.
    customer_email : str
        E-mail address of the authenticated customer. Set at graph entry.
    order_id : str | None
        Order reference extracted from the conversation, if any.
    rag_context : list[str]
        Chunks retrieved and reranked from the knowledge base. Set by the
        query_rag tool and available to the generation node.
    tool_results : list[dict[str, Any]]
        Accumulated results from all tool calls during the conversation.
        Each entry has the shape {"tool": str, "status": str, "data": Any}.
    pending_ticket : dict[str, Any] | None
        Ticket being built before confirmation (issue, customer_email,
        priority). Set when a ticket is staged, cleared after creation.
    resolved : bool
        True once the agent considers the issue fully resolved and the
        conversation can be closed.
    """

    messages: Annotated[list, add_messages]
    customer_email: str
    order_id: str | None
    rag_context: list[str]
    tool_results: list[dict[str, Any]]
    pending_ticket: dict[str, Any] | None
    resolved: bool
