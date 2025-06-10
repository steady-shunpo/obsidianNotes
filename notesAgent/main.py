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
from contextlib import asynccontextmanager # <--- Added this import
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

import threading
import logging
import traceback
import subprocess
import tempfile
import asyncio
import os
import sys
import json

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)


load_dotenv()

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

router_workflow = None
executor = ThreadPoolExecutor(max_workers=4)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global router_workflow
    myFunc()
    yield
    # Shutdown - cleanup if needed
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

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

obby_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
fire_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
testmodel = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

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

class State(TypedDict):
    input: list
    decision: str
    output: str

messages = []


def route_decision(state: State):
    """Route input to the appropriate tool"""
    logger.info(f"Routing decision for input: {state['input']}")
    input_text = state["input"]
    if isinstance(input_text, list):
        input_text = " ".join(str(item) for item in input_text)
    input_text = str(input_text).lower()
        
        # Check for note-related keywords
    decision = None
    note_keywords = ["note", "write", "save", "create", "obsidian", "document", "record", "vault", "obsidian"]
    if any(keyword in input_text for keyword in note_keywords):
        decision = "note"
    else:
        decision = "explanation"
    
    # logger.info(f"Decision made: {decision.step}")
    return {"decision": decision}


async def chat_with_obsidian(state: State):
    """Used for all obsidial vault related tasks. Supports listing, creating, editing and reading notes."""
    async with stdio_client(obsidian_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            # print("hi")
            agent = create_react_agent(obby_model, tools)

            system_message = SystemMessage(content="You are a helpful note taker. Use the tools provided in sequence to answer questions. Think step by step")
            tempmessages = [system_message]

            human_message = HumanMessage(content=state["input"])
            tempmessages.append(human_message)

            response = await asyncio.shield(agent.ainvoke({"messages": tempmessages}))

            return {"output": response["messages"][-1].content}

async def chat_with_firecrawl(state: State):
    """Used for all web related tasks. Supports scraping websites and providing information."""
    # print("hi2")
    async with stdio_client(firecrawl_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            logger.info(f"Firecrawl tools loaded: {len(tools)} tools")

            agent = create_react_agent(fire_model, tools)

            system_message = SystemMessage(content="You are a helpful assistant. You can use multiple tools in sequence to gather information and provide an explanation. Think step by step")
            tempmessages = [system_message]

            human_message = HumanMessage(content=state["input"])
            tempmessages.append(human_message)

            response = await asyncio.shield(agent.ainvoke({"messages": tempmessages}))
            logger.info("Firecrawl agent completed successfully")

            return {"output": response["messages"][-1].content}



# #WITHOUT LANGGRAPH
# # async def chat_with_obsidian(state: State):
# #     try:
# #         # Force synchronous execution
# #         def sync_obsidian():
# #             import nest_asyncio
# #             nest_asyncio.apply()
            
# #             loop = asyncio.new_event_loop()
# #             asyncio.set_event_loop(loop)
            
# #             async def run():
# #                 async with stdio_client(obsidian_params) as (read, write):
# #                     async with ClientSession(read, write) as session:
# #                         await session.initialize()
# #                         tools = await load_mcp_tools(session)
# #                         agent = create_react_agent(obby_model, tools)
                        
# #                         messages = [
# #                             SystemMessage(content="You are a helpful note taker."),
# #                             HumanMessage(content=state["input"])
# #                         ]
                        
# #                         response = await agent.ainvoke({"messages": messages})
# #                         return response["messages"][-1].content
            
# #             return loop.run_until_complete(run())
        
# #         output = sync_obsidian()
# #         return {"output": output}
# #     except Exception as e:
# #         return {"output": f"Error: {str(e)}"}

# # async def chat_with_firecrawl(state: State):
# #     try:
# #         # Force synchronous execution  
# #         def sync_firecrawl():
# #             import nest_asyncio
# #             nest_asyncio.apply()
            
# #             loop = asyncio.new_event_loop()
# #             asyncio.set_event_loop(loop)
            
# #             async def run():
# #                 async with stdio_client(firecrawl_params) as (read, write):
# #                     async with ClientSession(read, write) as session:
# #                         await session.initialize()
# #                         tools = await load_mcp_tools(session)
# #                         agent = create_react_agent(fire_model, tools)
                        
# #                         messages = [
# #                             SystemMessage(content="You are a helpful assistant."),
# #                             HumanMessage(content=state["input"])
# #                         ]
                        
# #                         response = await agent.ainvoke({"messages": messages})
# #                         return response["messages"][-1].content
            
# #             return loop.run_until_complete(run())
        
# #         output = sync_firecrawl()
# #         return {"output": output}
# #     except Exception as e:
# #         return {"output": f"Error: {str(e)}"}


# #TESTING
# async def chat_with_obsidian(state: State):
#     return {"output": f"Obsidian would process: {state['input']}"}

# # async def chat_with_firecrawl(state: State):
# #     return {"output": f"Firecrawl would process: {state['input']}"}


# #FIRECRAWL TEST
# # async def chat_with_firecrawl(state: State):
# #     try:
# #         async with stdio_client(firecrawl_params) as (read, write):
# #             async with ClientSession(read, write) as session:
# #                 await session.initialize()
# #                 tools = await load_mcp_tools(session)
                
# #                 # Don't create the agent yet - just return tool info
# #                 return {"output": f"Firecrawl connected with {len(tools)} tools for: {state['input']}"}
                
# #     except Exception as e:
# #         return {"output": f"Firecrawl connection error: {str(e)}"}



# #FIRECRAWL TEST 2
# # async def chat_with_firecrawl(state: State):
# #     try:
# #         async with stdio_client(firecrawl_params) as (read, write):
# #             async with ClientSession(read, write) as session:
# #                 await session.initialize()
# #                 tools = await load_mcp_tools(session)
# #                 agent = create_react_agent(fire_model, tools)
                
# #                 messages = [
# #                     SystemMessage(content="You are a helpful assistant."),
# #                     HumanMessage(content=state["input"])
# #                 ]
                
# #                 response = await agent.ainvoke({"messages": messages})
# #                 return {"output": response["messages"][-1].content}
                
# #     except Exception as e:

# #         return {"output": f"Firecrawl error: {str(e)}"}


# #FIRECRAWL TEST 3
# # async def chat_with_firecrawl(state: State):
# #     try:
# #         async with stdio_client(firecrawl_params) as (read, write):
# #             async with ClientSession(read, write) as session:
# #                 await session.initialize()
# #                 tools = await load_mcp_tools(session)
                
# #                 # Use the model directly with tools instead of react agent
# #                 model_with_tools = fire_model.bind_tools(tools)
                
# #                 response = await model_with_tools.ainvoke([
# #                     SystemMessage(content="You are a helpful assistant."),
# #                     HumanMessage(content=state["input"])
# #                 ])
                
# #                 return {"output": response.content}
                
# #     except Exception as e:
# #         return {"output": f"Firecrawl error: {str(e)}"}
    

# #FIRECRAWL TEST 4
# # async def chat_with_firecrawl(state: State):
# #     try:
# #         async with stdio_client(firecrawl_params) as (read, write):
# #             async with ClientSession(read, write) as session:
# #                 await session.initialize()
# #                 tools = await load_mcp_tools(session)
                
# #                 model_with_tools = fire_model.bind_tools(tools)
                
# #                 response = await model_with_tools.ainvoke([
# #                     HumanMessage(content=state["input"])
# #                 ])
                
# #                 return {"output": response.content}
                
# #     except Exception as e:
# #         return {"output": f"Model invoke error: {str(e)}"}
    
# #FIRECRAWL TEST 5
# # async def chat_with_firecrawl(state: State):
# #     try:
# #         # Use the synchronous invoke instead of ainvoke
# #         response = fire_model.invoke([
# #             HumanMessage(content=f"Process this request: {state['input']}")
# #         ])
        
# #         return {"output": response.content}
        
# #     except Exception as e:
# #         return {"output": f"Error: {str(e)}"}


# #FIRECRAWL TEST 6
# # async def chat_with_firecrawl(state: State):
# #     try:
# #         async with stdio_client(firecrawl_params) as (read, write):
# #             async with ClientSession(read, write) as session:
# #                 await session.initialize()
# #                 tools = await load_mcp_tools(session)
                
# #                 model_with_tools = fire_model.bind_tools(tools)
                
# #                 # Use synchronous invoke instead of ainvoke
# #                 response = model_with_tools.invoke([
# #                     HumanMessage(content=state["input"])
# #                 ])
                
# #                 return {"output": response.content}
                
# #     except Exception as e:
# #         return {"output": f"Error: {str(e)}"}


# #TEST 6
# def run_firecrawl_in_new_process(input_text):
#     """Run firecrawl in completely separate process to avoid async conflicts"""
#     import subprocess
#     import json
#     import tempfile
#     import os
    
#     # Create a temporary script
#     script_content = f'''
# import asyncio
# import sys
# import os
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from dotenv import load_dotenv

# load_dotenv()

# async def main():
#     try:
#         firecrawl_params = StdioServerParameters(
#             command="npx",
#             env={{"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")}},
#             args=["-y", "firecrawl-mcp"]
#         )
        
#         model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
#         async with stdio_client(firecrawl_params) as (read, write):
#             async with ClientSession(read, write) as session:
#                 await session.initialize()
#                 tools = await load_mcp_tools(session)
#                 agent = create_react_agent(model, tools)
                
#                 messages = [
#                     SystemMessage(content="You are a helpful assistant. You can use multiple tools in sequence to gather information and provide an explanation. Think step by step."),
#                     HumanMessage(content="{input_text}")
#                 ]
                
#                 response = await agent.ainvoke({{"messages": messages}})
#                 print(response["messages"][-1].content)
                
#     except Exception as e:
#         print(f"Error: {{str(e)}}")

# if __name__ == "__main__":
#     asyncio.run(main())
# '''
    
#     try:
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
#             f.write(script_content)
#             temp_script = f.name
        
#         # Run the script in a separate process
#         result = subprocess.run([sys.executable, temp_script], 
#                               capture_output=True, text=True, timeout=60)
        
#         # Cleanup
#         os.unlink(temp_script)
        
#         if result.returncode == 0:
#             return result.stdout.strip()
#         else:
#             return f"Process error: {result.stderr}"
            
#     except Exception as e:
#         return f"Subprocess error: {str(e)}"

# async def chat_with_firecrawl(state: State):
#     """Run firecrawl in separate process to avoid all async issues"""
#     try:
#         # Run in thread pool to avoid blocking
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(executor, run_firecrawl_in_new_process, state["input"])
#         return {"output": result}
#     except Exception as e:
#         return {"output": f"Error: {str(e)}"}


 
def decision(state:State):
    decision_value = state.get("decision", "explanation")
    logger.info(f"Making routing decision: {decision_value}")
    if state["decision"] == "note":
        return "obby"
    else:
        return "firecrawl"





# def myFunc():
#     global router_workflow
#     logger.info("Initializing workflow...")
    
#     # Simple synchronous routing instead of LangGraph
#     def simple_router(input_text):
#         if isinstance(input_text, list):
#             input_text = " ".join(str(item) for item in input_text)
#         input_text = str(input_text).lower()
        
#         note_keywords = ["note", "write", "save", "create", "obsidian", "document", "record", "vault"]
#         if any(keyword in input_text for keyword in note_keywords):
#             return "obsidian"
#         else:
#             return "firecrawl"
    
#     router_workflow = simple_router
#     logger.info("Simple router initialized")





#LANGGRAPH ROUTE
# def myFunc():
#     global router_workflow
#     logger.info("Initializing workflow...")
#     route_builder = StateGraph(State)

#     route_builder.add_node("obsidian", chat_with_obsidian)
#     route_builder.add_node("firecrawl", chat_with_firecrawl)
#     route_builder.add_node("router", route_decision)

#     route_builder.add_edge(START, "router")
#     route_builder.add_conditional_edges(
#         "router",
#         decision,
#         {
#             "obby": "obsidian", 
#             "firecrawl": "firecrawl"
#         }
#     )

#     route_builder.add_edge("obsidian", END)
#     route_builder.add_edge("firecrawl", END)

#     router_workflow = route_builder.compile()
#     logger.info("Workflow initialized successfully")

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

class ChatRequest(BaseModel):
    message: str
 


# def run_workflow_in_thread(message: str):
#     try:
#         route = router_workflow(message)
        
#         if route == "obsidian":
#             # Run obsidian directly
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             result = loop.run_until_complete(chat_with_obsidian({"input": message}))
#             loop.close()
#             return result
#         else:
#             # Run firecrawl directly
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             result = loop.run_until_complete(chat_with_firecrawl({"input": message}))
#             loop.close()
#             return result
            
#     except Exception as e:
#         return {"output": f"Error: {str(e)}"}




#LANGGRAPH THING
# def run_workflow_in_thread(message: str):
#     try:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
        
#         if sys.platform == 'win32':
#             asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
#         result = loop.run_until_complete(router_workflow.ainvoke({"input": message}))
#         loop.close()
#         return result
        
#     except Exception as e:
#         return {"output": f"Error: {str(e)}"}

@app.post("/chat")
async def chat_endpoint(chatrequest: ChatRequest):
    await mainfunction()

    # global messages
    # state = None
    # if hasattr(asyncio, 'to_thread'):
    #     state = await asyncio.to_thread(run_workflow_in_thread, chatrequest.message)
    # else:
    #     loop = asyncio.get_event_loop()
    #     state = await loop.run_in_executor(executor, run_workflow_in_thread, chatrequest.message)

    # messages.append(chatrequest.message)
    # logger.info("chat completed")
    # if state and "output" in state:
    #    messages.append(state["output"])
    # print(f"AI response: {state['output']}")
    # return{"reply": state["output"]}
    # # print("test: ", test)



    # global messages
    # print(f"Received message: {chatrequest.message}")
    # messages.append(chatrequest.message)
    # state = await router_workflow.ainvoke({"input": messages})
    # print("hi")
    # messages.append(state["output"])
    # print(f"AI response: {state['output']}")
    # return {"reply": state["output"]}
#     except Exception as e:
#         error_message = f"Error processing request: {str(e)}"
#         print(error_message)
#         return {"error": error_message, "status": "error"}


# @app.get("/test")
# async def test_endpoint():
#     """Test endpoint to verify basic functionality"""
#     try:
#         logger.info("Testing basic functionality")
        
#         # Test simple model invocation
#         test_message = HumanMessage(content="Hello, this is a test")
#         response = model.invoke([test_message])
        
#         return {"status": "success", "test_response": response.content}
        
#     except Exception as e:
#         logger.error(f"Test endpoint failed: {e}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
        
@app.get("/chat-clear")
async def chat_clear():
    global messages
    print("Chat cleared")
    messages = []
    return {"status": "chat history cleared"}


# async def test_message():
#     test = await router_workflow.ainvoke({"input": ["hello"]}) 
#     print("test: ", test) 
#     return test

# @app.on_event("startup")
# async def startup_event():
#     # Schedule test_message to run as a background task.
#     # The event loop is already running when @app.on_event("startup") is called.
#     asyncio.create_task(test_message())


async def mainfunction(text):
    # while True:
    #     user_input = input("You: ")
    #     if(user_input.lower() == "exit"):
    #         print("Exiting...")
    #         break
    #     messages.append(user_input)
    try:

        test1 = await router_workflow.ainvoke({"input": text})
        # messages.append(state["output"])
        # print("AI: ", state["output"])
        
        print(json.dumps({"reply": test1["output"]}))
    except ValueError as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    jsonString = sys.argv[1]
    data = json.loads(jsonString)
    asyncio.run(mainfunction(data))