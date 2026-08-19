import json
from typing import List, Dict, Any
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_community.tools import TavilySearchResults

# Create the tavily search tool
tavily_tool = TavilySearchResults(max_results=5)


def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]

    # Extract tools Calls from the AI message
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queries
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            # Execute each search query using the TavilySearchResults tool
            query_results = {}
            for query in search_queries:
                query_results[query] = tavily_tool.invoke(query)
            # Create a tool message with the results
            tool_messages.append(
                ToolMessage(content=json.dumps(query_results), tool_call_id=call_id)
            )
    return tool_messages
