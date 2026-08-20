from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

load_dotenv()


def reason_node(state: AgentState):
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


# ToolNode handles tool lookup and execution automatically
act_node = ToolNode(tools)
