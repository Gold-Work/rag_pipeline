import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Generator

from langfuse import Langfuse


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


# Holds the active Langfuse trace for the current thread/task context.
# ContextVar is safe under FastAPI's sync thread-pool: each request thread
# gets its own context copy, and trace_context resets on exit.
_active_trace: ContextVar = ContextVar("langfuse_trace", default=None)


def get_current_trace():
    return _active_trace.get()


@contextmanager
def trace_context(name: str, input: dict, user_id: str = "system") -> Generator:
    """Start a Langfuse root trace and make it active for the current thread.

    All langfuse_span / langfuse_generation calls within this block will
    automatically attach sub-spans to this trace.  The trace is reset and
    flushed on exit, even if an exception propagates.
    """
    langfuse = get_langfuse()
    trace = langfuse.trace(name=name, input=input, user_id=user_id)
    token = _active_trace.set(trace)
    try:
        yield trace
    finally:
        _active_trace.reset(token)
        langfuse.flush()


@contextmanager
def langfuse_span(name: str, input: dict | None = None) -> Generator:
    """Create a span on the active trace and yield a mutable output dict.

    The dict's final state becomes the span's output.
    No-op (yields empty dict) when no trace is active.

    Usage::

        with langfuse_span("vector-search", {"k": k}) as out:
            results = do_search(...)
            out["results_count"] = len(results)
    """
    trace = _active_trace.get()
    if trace is None:
        yield {}
        return

    span = trace.span(name=name, input=input or {})
    out: dict = {}
    try:
        yield out
    finally:
        span.end(output=out or None)


@contextmanager
def langfuse_generation(name: str, model: str, input: str | dict) -> Generator:
    """Create a Langfuse generation node on the active trace.

    Yields a mutable output dict.  Set ``out["output"]`` with the model
    response so Langfuse can display it correctly in the UI.
    No-op (yields empty dict) when no trace is active.

    Usage::

        with langfuse_generation("llm", model=model, input=prompt) as out:
            answer = call_llm(prompt)
            out["output"] = answer
    """
    trace = _active_trace.get()
    if trace is None:
        yield {}
        return

    gen = trace.generation(name=name, model=model, input=input)
    out: dict = {}
    try:
        yield out
    finally:
        gen.end(output=out.get("output"))
