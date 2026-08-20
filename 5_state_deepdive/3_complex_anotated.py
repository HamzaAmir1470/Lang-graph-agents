from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.graph import END
import operator


class SimpleState(TypedDict):
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[list[int], operator.concat]


def increment(state: SimpleState) -> SimpleState:
    """Increment the count in the state by 1."""
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "sum": new_count,
        "history": [new_count],
    }


def should_continue(state):
    if state["count"] < 5:
        return "continue"
    else:
        return "stop"


graph = StateGraph(SimpleState)

graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_conditional_edges(
    "increment", should_continue, {"continue": "increment", "stop": END}
)

app = graph.compile()

state = {"count": 0, "sum": 0, "history": []}
result = app.invoke(state)

print(result)
