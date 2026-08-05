from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool
from langchain_groq import ChatGroq
import yfinance as yf


load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

@tool
def get_stock_price(ticker:str)->str:
    """Get the Current stock price of the giver ticker symbol"""
    stock = yf.Ticker(ticker)
    price = stock.info.get("currentprice")
    return price


agent = create_agent(
    tools=[get_stock_price],
    model=llm,
    system_prompt=(
        "You are a Finance Agent"
        
    )
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What is the current Stock price of AAPL?"
                ),
            },
        ]
    }
)

print(result["messages"][-1].content)

GREEN = "\033[92m"
RESET = "\033[0m"
print(GREEN + result["messages"][-1].content + RESET)