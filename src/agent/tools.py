"""Tools available to the SAV e-commerce agent.

Each tool returns a dict with two mandatory keys:
  - status : "ok" | "error" | "not_found"
  - data   : result payload (varies per tool)
"""

import re

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("agent.tools")
config = get_config()

# ---------------------------------------------------------------------------
# Simulated order database — 3 fictitious orders
# ---------------------------------------------------------------------------

_ORDERS: dict[str, dict] = {
    "CMD-2025-00142": {
        "order_id": "CMD-2025-00142",
        "customer_email": "marie.dupont@example.com",
        "status": "livree",
        "status_label": "Livree le 02/04/2025",
        "items": [
            {"ref": "ROB-SIGN-M-NOIR", "label": "Robe Signature taille M noir", "qty": 1, "price": 89.90},
            {"ref": "CEN-CUIR-MARRON", "label": "Ceinture cuir marron T38", "qty": 1, "price": 34.50},
        ],
        "total": 124.40,
        "carrier": "Colissimo",
        "tracking": "6X123456789FR",
        "delivery_address": "12 rue des Lilas, 75011 Paris",
        "return_deadline": "02/05/2025",
    },
    "CMD-2025-00287": {
        "order_id": "CMD-2025-00287",
        "customer_email": "thomas.martin@example.com",
        "status": "en_transit",
        "status_label": "En transit — livraison prevue le 08/04/2025",
        "items": [
            {"ref": "PULL-PRES-L-GRIS", "label": "Pull Prestige taille L gris chiné", "qty": 2, "price": 119.00},
        ],
        "total": 238.00,
        "carrier": "Chronopost",
        "tracking": "CH987654321FR",
        "delivery_address": "8 avenue Jean Jaures, 69007 Lyon",
        "return_deadline": None,
    },
    "CMD-2025-00391": {
        "order_id": "CMD-2025-00391",
        "customer_email": "lea.bernard@example.com",
        "status": "en_preparation",
        "status_label": "En preparation — expedition prevue le 07/04/2025",
        "items": [
            {"ref": "JEAN-SLIM-38-BLEU", "label": "Jean Slim W30/L32 bleu brut", "qty": 1, "price": 79.90},
            {"ref": "TEE-ESSEN-S-BLANC", "label": "T-shirt Essentiel taille S blanc", "qty": 3, "price": 19.90},
        ],
        "total": 139.60,
        "carrier": "Mondial Relay",
        "tracking": None,
        "delivery_address": "Point relais - Tabac Presse, 33000 Bordeaux",
        "return_deadline": None,
    },
}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalize_order_id(raw: str) -> str | None:
    """Resolve a raw user-provided string to a known order ID.

    Three lookup steps, in order:

    1. Exact match (case-insensitive, stripped).
    2. Suffix match: extract all digit sequences from *raw*, build candidate
       suffixes, and return the unique order ID whose numeric suffix matches.
       Returns ``None`` if zero or more than one order matches.
    3. Fuzzy match: compute Levenshtein distance against every known ID and
       return the closest one only if it is unique below the strict threshold
       ``max(2, len(raw) // 6)``.  Returns ``None`` if ambiguous or all
       distances exceed the threshold.

    Parameters
    ----------
    raw : str
        The user-provided order reference (may be partial, malformed, etc.)

    Returns
    -------
    str | None
        The canonical order ID key from ``_ORDERS``, or ``None`` if not found.
    """
    candidate = raw.strip().upper()

    # Step 1 — exact match
    if candidate in _ORDERS:
        logger.info(f"[normalize_order_id] exact match: {candidate!r}")
        return candidate

    # Step 2 — suffix match on the digit run extracted from raw
    digits = re.sub(r"\D", "", candidate)
    if digits:
        suffix_matches = [oid for oid in _ORDERS if oid.endswith(digits)]
        if len(suffix_matches) == 1:
            logger.info(f"[normalize_order_id] suffix match: {candidate!r} → {suffix_matches[0]!r}")
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            logger.warning(f"[normalize_order_id] suffix ambiguous for {candidate!r}: {suffix_matches}")

    # Step 3 — fuzzy Levenshtein match
    threshold = max(2, len(candidate) // 6)
    scored = [(oid, _levenshtein(candidate, oid)) for oid in _ORDERS]
    scored.sort(key=lambda x: x[1])
    best_id, best_dist = scored[0]
    if best_dist <= threshold:
        # Accept only if uniquely closest (no tie at same distance)
        if len(scored) == 1 or scored[1][1] > best_dist:
            logger.info(f"[normalize_order_id] fuzzy match (dist={best_dist}): {candidate!r} → {best_id!r}")
            return best_id
        logger.warning(f"[normalize_order_id] fuzzy ambiguous for {candidate!r}: dist={best_dist}")

    logger.warning(f"[normalize_order_id] no match for {candidate!r}")
    return None


def query_rag(question: str) -> dict:
    """Query the RAG knowledge base and return the top reranked chunks.

    Uses the same retrieve() + rerank() pipeline as the main API endpoint.

    Parameters
    ----------
    question : str
        The customer question to answer from the knowledge base.

    Returns
    -------
    dict
        status : "ok" | "error"
        data   : {"chunks": list[str], "sources": list[str], "n": int}
    """
    # Import here to avoid circular imports and heavy loading at module level
    from src.query.reranker import rerank
    from src.query.retriever import retrieve

    logger.info(f"[tool:query_rag] question='{question[:80]}'")
    try:
        docs, _ = retrieve(question, k=config["retrieval"]["top_k_retrieval"])
        docs = rerank(question, docs, top_k=config["retrieval"]["top_k_rerank"])
        chunks = [doc.page_content for doc in docs]
        sources = list(dict.fromkeys(doc.metadata.get("source_file", "?") for doc in docs))
        logger.info(f"[tool:query_rag] {len(chunks)} chunks retrieved from {sources}")
        return {
            "status": "ok",
            "data": {"chunks": chunks, "sources": sources, "n": len(chunks)},
        }
    except Exception as e:
        logger.error(f"[tool:query_rag] erreur : {e}", exc_info=True)
        return {"status": "error", "data": {"message": str(e)}}


def check_order(order_id: str) -> dict:
    """Look up an order by its reference number.

    Parameters
    ----------
    order_id : str
        Order reference, e.g. "CMD-2025-00142".

    Returns
    -------
    dict
        status : "ok" | "not_found"
        data   : order dict if found, {"message": str} otherwise
    """
    logger.info(f"[tool:check_order] order_id='{order_id}'")
    resolved = normalize_order_id(order_id)
    if resolved is None:
        logger.warning(f"[tool:check_order] commande introuvable : {order_id}")
        return {
            "status": "not_found",
            "data": {"message": f"Aucune commande trouvee pour la reference {order_id!r}."},
        }
    order = _ORDERS[resolved]
    logger.info(f"[tool:check_order] commande trouvee : {resolved} statut={order['status']}")
    return {"status": "ok", "data": order}


def send_email(to: str, subject: str, body: str) -> dict:
    """Send a transactional e-mail to a customer (logged, not actually sent).

    Parameters
    ----------
    to : str
        Recipient e-mail address.
    subject : str
        E-mail subject line.
    body : str
        Plain-text body of the e-mail.

    Returns
    -------
    dict
        status : "ok" | "error"
        data   : {"to": str, "subject": str, "preview": str}
    """
    logger.info(f"[tool:send_email] to='{to}' | subject='{subject}' | body_len={len(body)}")
    try:
        preview = body[:120].replace("\n", " ") + ("..." if len(body) > 120 else "")
        # Production: integrate with SendGrid / SES / Mailjet here
        logger.info(f"[tool:send_email] EMAIL ENVOYE (simulation) >> {preview}")
        return {"status": "ok", "data": {"to": to, "subject": subject, "preview": preview}}
    except Exception as e:
        logger.error(f"[tool:send_email] erreur : {e}", exc_info=True)
        return {"status": "error", "data": {"message": str(e)}}


def create_ticket(issue: str, customer_email: str) -> dict:
    """Create a support ticket for an unresolved customer issue (logged).

    Parameters
    ----------
    issue : str
        Description of the issue that could not be resolved automatically.
    customer_email : str
        Customer e-mail address to attach to the ticket.

    Returns
    -------
    dict
        status : "ok" | "error"
        data   : {"ticket_id": str, "issue": str, "customer_email": str, "priority": str}
    """
    import hashlib
    from datetime import datetime, timezone

    logger.info(f"[tool:create_ticket] email='{customer_email}' | issue='{issue[:80]}'")
    try:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
        sig = hashlib.md5(f"{customer_email}{issue}{ts}".encode(), usedforsecurity=False).hexdigest()[:6].upper()
        ticket_id = f"TKT-{ts[:8]}-{sig}"

        priority = (
            "haute"
            if any(w in issue.lower() for w in ("rembours", "defaut", "erreur", "urgent", "jamais"))
            else "normale"
        )

        logger.info(f"[tool:create_ticket] TICKET CREE (simulation) id={ticket_id} priorite={priority}")
        return {
            "status": "ok",
            "data": {
                "ticket_id": ticket_id,
                "issue": issue,
                "customer_email": customer_email,
                "priority": priority,
            },
        }
    except Exception as e:
        logger.error(f"[tool:create_ticket] erreur : {e}", exc_info=True)
        return {"status": "error", "data": {"message": str(e)}}
