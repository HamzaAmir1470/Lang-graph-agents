from typing import Annotated, List, Sequence
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"

# Define state schema using TypedDict and add_messages reducer
class State(TypedDict):
    messages: Annotated[list, add_messages]


def generate_node(state: State):
    response = generation_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def reflect_node(state: State):
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


def should_continue(state: State):
    if len(state["messages"]) > 4:
        return END
    return REFLECT


# Initialize StateGraph instead of deprecated MessageGraph
builder = StateGraph(State)

builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflect_node)

builder.add_edge(START, GENERATE)

# Explicitly pass path map {output_string: target_node}
builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        REFLECT: REFLECT,
        END: END,
    },
)

builder.add_edge(REFLECT, GENERATE)

app = builder.compile()

# Visualizations will now render the loop to 'reflect' and the path to '__end__'
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()