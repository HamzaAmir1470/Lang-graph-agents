import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent  # Modern import replaces LangGraph deprecation
from tavily import TavilyClient

load_dotenv()

# 1. Initialize Tavily Client
api_key = os.getenv("TAVILY_API_KEY")
print(f"Loaded API Key: {api_key}")  # This will no longer be None!

# 3. Initialize Tavily Client
tavily_client = TavilyClient(api_key=api_key)

# 2. Define search tool
@tool
def web_search(query: str) -> str:
    """Search the web for real-time information and news using Tavily."""
    print(f"\n[EXECUTION LOG] Attempting to call Tavily API with key: {os.getenv('TAVILY_API_KEY')}\n")
    
    response = tavily_client.search(query=query, max_results=3)
    results = [f"- {item['title']}: {item['content']}" for item in response.get("results", [])]
    
    print(f"[EXECUTION LOG] Search Results Returned: {results}\n")
    return "\n".join(results)
# 3. Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# 4. Build Agent
tools = [web_search]
agent_executor = create_agent(llm, tools)

# 5. Run prompt
response = agent_executor.invoke({"messages": [("user", "Give me the tweet for the current weather condition in Pakistan.")]})

# 6. Extract ONLY the string content from the final message
last_message = response["messages"][-1]

if isinstance(last_message.content, str):
    poem = last_message.content
elif isinstance(last_message.content, list):
    poem = "".join([chunk["text"] for chunk in last_message.content if isinstance(chunk, dict) and "text" in chunk])
else:
    poem = str(last_message.content)

print(poem.strip())