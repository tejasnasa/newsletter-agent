"""The agent graph — the whole control-flow contract lives here.

    START
      -> plan -> research -> curate -> write_draft -> critique
      -> [route_after_critique]
            "revise" ------> revise -> critique          <- self-critique loop
            "human_review"
      -> [route_after_human_review]
            "revise" ------> revise -> critique -> human_review   <- human loop
            "render"
      -> render -> send -> END

``settings`` is bound into every node and router with :func:`functools.partial`,
so the node functions themselves stay plain ``(state, settings)`` functions that
can be unit-tested without the graph.
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src import nodes
from src.config import Settings
from src.state import NewsletterState


def build_graph(settings: Settings, *, checkpointer: BaseCheckpointSaver | None = None):
    """Compile the newsletter graph.

    A checkpointer is what makes the HITL pause real: without one, LangGraph
    cannot checkpoint and suspend at ``interrupt()``.
    """
    builder = StateGraph(NewsletterState)

    builder.add_node("plan", partial(nodes.plan_node, settings=settings))
    builder.add_node("research", partial(nodes.research_node, settings=settings))
    builder.add_node("curate", partial(nodes.curate_node, settings=settings))
    builder.add_node("write_draft", partial(nodes.write_draft_node, settings=settings))
    builder.add_node("critique", partial(nodes.critique_node, settings=settings))
    builder.add_node("revise", partial(nodes.revise_node, settings=settings))
    builder.add_node("human_review", partial(nodes.human_review_node, settings=settings))
    builder.add_node("render", partial(nodes.render_node, settings=settings))
    builder.add_node("send", partial(nodes.send_node, settings=settings))

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "curate")
    builder.add_edge("curate", "write_draft")
    builder.add_edge("write_draft", "critique")

    # Loop 1: critique loops back through revise until the verdict is "approve"
    # or `max_revisions` is reached.
    builder.add_conditional_edges(
        "critique",
        partial(nodes.route_after_critique, settings=settings),
        {"revise": "revise", "human_review": "human_review"},
    )
    builder.add_edge("revise", "critique")

    # Loop 2: a human asking for changes re-enters the same revise -> critique
    # cycle, then comes back here for another look.
    builder.add_conditional_edges(
        "human_review",
        partial(nodes.route_after_human_review, settings=settings),
        {"revise": "revise", "render": "render"},
    )

    builder.add_edge("render", "send")
    builder.add_edge("send", END)

    return builder.compile(checkpointer=checkpointer)
