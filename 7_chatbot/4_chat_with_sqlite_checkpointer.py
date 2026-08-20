from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages, StateGraph, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# 1. Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

sqlite_connection = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)


# 2. Define State
class BasicChatState(TypedDict):
    messages: Annotated[list, add_messages]


# 3. Define Chatbot Node
def chatbot(state: BasicChatState):
    return {"messages": [llm.invoke(state["messages"])]}


# 4. Construct Graph
graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

# 5. Attach Checkpointer
memory = SqliteSaver(sqlite_connection)
app = graph.compile(checkpointer=memory)

# 6. Correct Config Key structure: "configurable", thread_id as string
config = {"configurable": {"thread_id": "1"}}


# 7. Interactive Execution Loop
if __name__ == "__main__":
    print("Chatbot initialized with Memory. Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Pass config into invoke so MemorySaver persists history per thread_id
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )

        # Get the latest AI message from the message sequence
        latest_message = result["messages"][-1]
        print(f"Bot: {latest_message.content}\n")
