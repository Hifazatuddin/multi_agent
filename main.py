
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import os
from dotenv import load_dotenv
import chainlit as cl

load_dotenv()


gemini_api_key=os.getenv("GEMINI_API_KEY")

external_client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client

)
config=RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True

)

backend_agent = Agent(
    name="Backend_Expert",
    instructions="""
You are a backend development expert.You help user with backend topics like APIs,database,authentication,server,framework
Dont not frontend or UI questions.
""",
)
frontend_agent= Agent(
    name="Frontend Expert",
    instructions="""
You are a frontend expert.You help with UI/UX using HTML,CSS,JavaScripts,React,Next.Js and Tailwind CSS.
Do Not Answer Backend-related questions.
""",
)
poetry_agent=Agent(
    name="poetry",
    instructions="You are specilized for love poetry ?",

)
math_agent=Agent(
    name="Math solve",
    instructions="Math solution,Math helper,Math Assistant",
)
English_Grammer_Checker_agent=Agent(
    name="Grammer_Checker",
    instructions="You are helper for Assistant to check Grammer checking",
)
web_dev_agent = Agent(
    name="Web Developer Agent",
    instructions="""
You are generalist web developer who decides weather a questions is about frontend or backend

if the user asks about UI,Html,React,etc.,hand off to the frontend or backend.
if the user asks about APIs,databases,servers,backend frameworks,etc., hand off to the backend expert.
if it's unrelated to both,politely decline.
""",
handoffs=[frontend_agent, backend_agent,poetry_agent,math_agent,English_Grammer_Checker_agent]
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Welcome to the chainlit-project! How can I assist you today?").send()

@cl.on_message
async def handle_on_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")
    await msg.send()

    result = Runner.run_streamed(
        web_dev_agent,
        input=history,
        run_config=config
        
    )
    async for event in result.stream_events():
        # Assuming event.data contains the streamed text chunk
        if event.type == "raw_response_event" and hasattr(event.data, "delta"):
            await msg.stream_token(event.data.delta)
        elif event.type == "raw_response_event" and isinstance(event.data, str):
            await msg.stream_token(event.data)
    history.append({"role": "assistant", "content": result.final_output})

    await cl.Message(content=result.final_output).send()

    cl.user_session.set("history", history)
