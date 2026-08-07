from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool
from langchain_groq import ChatGroq
import yfinance as yf


load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


agent = create_agent(
    tools=[],
    model=llm,
    system_prompt=(
        "You are an image Agent"
        
    )
)


result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Which animal is in this image? Print only the name of the animal : https://www.google.com/imgres?q=chick%20monk&imgurl=https%3A%2F%2Fwww.treehugger.com%2Fthmb%2FfOtMLld0bvT64ZMZvAsfMxfYgKE%3D%2F1500x0%2Ffilters%3Ano_upscale()%3Amax_bytes(150000)%3Astrip_icc()%2F__opt__aboutcom__coeus__resources__content_migration__mnn__images__2017__10__Eastern_chipmunk-002d11c556e04957a6a39ed1b86a5d2b.jpg&imgrefurl=https%3A%2F%2Fwww.treehugger.com%2Fthings-you-dont-know-about-chipmunks-4864283&docid=Edc_gvQxUwmIbM&tbnid=YQhGconqh0Uo1M&vet=12ahUKEwjjib6w-4iWAxXAlOEIHeIRJBQQnPAOegQIMhAA..i&w=1024&h=768&hcb=2&ved=2ahUKEwjjib6w-4iWAxXAlOEIHeIRJBQQnPAOegQIMhAA"
                ),
            },
        ]
    }
)

print(result["messages"][-1].content)

GREEN = "\033[92m"
RESET = "\033[0m"
print(GREEN + result["messages"][-1].content + RESET)