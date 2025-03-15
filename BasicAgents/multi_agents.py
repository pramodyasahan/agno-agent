from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Groq(id="llama3-70b-8192"),
    show_tool_calls=True,
    markdown=True,
    tools=[DuckDuckGoTools()],
    instructions="Always include all the details"
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_info=True)],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Groq(id="llama3-70b-8192"),
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response("Can you go to this link and extract me the Course Outline. "
                     "The link is https://www.iit.ac.lk/course/bsc-artificial-intelligence-and-data-science/")
