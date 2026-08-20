from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import add_messages, StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


# 1. Define State
class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Setup Tools
search_tool = TavilySearch(max_results=2)
tools = [search_tool]

# 3. Setup Model (Use a valid model string)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
llm_with_tools = llm.bind_tools(tools=tools)


# 4. Define Chatbot Node
def chatbot(state: BasicChatBot):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# 5. Build Graph
graph = StateGraph(BasicChatBot)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("chatbot")

# Built-in tools_condition handles checking tool_calls automatically
graph.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "chatbot")

app = graph.compile()


# 6. Interactive CLI Loop
if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "end", "quit"]:
            break

        # Pass inputs and stream/print response per iteration
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})

        # Print only the final AI response message content
        final_message = result["messages"][-1]
        print(f"Bot: {final_message.content}\n")
