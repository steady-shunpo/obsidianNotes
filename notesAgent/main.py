from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import Literal, TypedDict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import os
import json

load_dotenv()
app = FastAPI()

origins = ['http://localhost:5173']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    print("Hello World")
    return {"message": "Hello World"}

obby_model = ChatGoogleGenerativeAI(model="learnlm-2.0-flash")
fire_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

firecrawl_params = StdioServerParameters(
    command = "npx",
    env= {
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
      },
    args=["-y", "firecrawl-mcp"]
)

obsidian_params = StdioServerParameters(
    command = f"{os.getenv('PATH_TO_SERVER')}\\.venv\\Scripts\\uv.EXE",
    args=[
        "run",
        "--with",
        "mcp[cli]",
        "--with",
        "requests",
        "mcp",
        "run",
        f"{os.getenv('PATH_TO_SERVER')}\\main.py"
      ]
)

class Route(BaseModel):
    step: Literal["note", "explanation"] = Field(
        None, description="The next step in the routing process"
    )

router = model.with_structured_output(Route)

route_builder = StateGraph(State)

route_builder.add_node("obsidian", chat_with_obsidian)
route_builder.add_node("firecrawl", chat_with_firecrawl)
route_builder.add_node("router", route_decision)

route_builder.add_edge(START, "router")
route_builder.add_conditional_edges(
    "router",
    decision,
    {
        "obby": "obsidian",
        "firecrawl": "firecrawl"
    }
)

route_builder.add_edge("obsidian", END)
route_builder.add_edge("firecrawl", END)

router_workflow = route_builder.compile()
messages = []

class State(TypedDict):
    input: list
    decision: str
    output: str


def route_decision(state: State):
    """Route input to the appropriate tool"""

    descision = router.invoke(
        [
        SystemMessage(content="Route the input to either explanation or note taking based on user's request"),
        HumanMessage(content=state["input"]),
        ]    
    )

    return {"decision": descision.step}


async def chat_with_obsidian(state: State):
    """Used for all obsidial vault related tasks. Supports listing, creating, editing and reading notes."""
    async with stdio_client(obsidian_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            print("hi")
            agent = create_react_agent(obby_model, tools)

            system_message = SystemMessage(content="You are a helpful note taker. Use the tools provided in sequence to answer questions. Think step by step")
            tempmessages = [system_message]

            human_message = HumanMessage(content=state["input"])
            tempmessages.append(human_message)

            response = await agent.ainvoke({"messages": tempmessages})

            return {"output": response["messages"][-1].content}

async def chat_with_firecrawl(state: State):
    """Used for all web related tasks. Supports scraping websites and providing information."""
    async with stdio_client(firecrawl_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            agent = create_react_agent(fire_model, tools)

            system_message = SystemMessage(content="You are a helpful assistant. You can use multiple tools in sequence to gather information and provide an explanation. Think step by step")
            tempmessages = [system_message]

            human_message = HumanMessage(content=state["input"])
            tempmessages.append(human_message)

            response = await agent.ainvoke({"messages": tempmessages})

            return {"output": response["messages"][-1].content}


def decision(state:State):
    if state["decision"] == "note":
        return "obby"
    else:
        return "firecrawl"




async def mainfunction():
    while True:
        user_input = input("You: ")
        if(user_input.lower() == "exit"):
            print("Exiting...")
            break
        messages.append(user_input)
        state = await router_workflow.ainvoke({"input": messages})
        messages.append(state["output"])
        print("AI: ", state["output"])
        
@app.get("/chat-clear")
async def chat_clear():
    print("Chat cleared")
    messages = [];


if __name__ == "__main__":
    asyncio.run(mainfunction())