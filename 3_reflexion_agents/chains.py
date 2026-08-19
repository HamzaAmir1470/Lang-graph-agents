import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.7,
)

# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher.

Current time: {time}

1. {first_instruction}

2. Reflect and critique your answer. Be severe to maximize improvement.

3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# First Responder Chain
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide me a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# Revisor Chain
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
    - You must include numerical citations in your revised answer to ensure it can be verified. 
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of: 
        - [1] https://www.example.com
        - [2] https://www.example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# Note: Removed | output_parser so the node returns raw AIMessage required by MessageGraph
revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
