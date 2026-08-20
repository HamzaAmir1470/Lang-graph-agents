import datetime
from dotenv import load_dotenv

from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

search_tool = TavilySearchResults(search_depth="basic")


@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current system time in the specified format."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


tools = [search_tool, get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools, llm=llm, prompt=react_prompt)
