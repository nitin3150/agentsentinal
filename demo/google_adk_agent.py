from dotenv import load_dotenv
load_dotenv()

import os
import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

MODEL = os.environ["MODEL"]

agent = Agent(
    name="demo_agent",
    model=MODEL,
    instruction="You are a helpful assistant.",
)

session_service = InMemorySessionService()

runner = Runner(
    agent=agent,
    app_name="demo_app",
    session_service=session_service,
)

async def main():
    session = await session_service.create_session(
        app_name="demo_app",
        user_id="user1",
        session_id="session1",
    )

    content = types.Content(
        role="user",
        parts=[types.Part(text="What is Python?")]
    )

    events = runner.run(
        user_id="user1",
        session_id="session1",
        new_message=content,
    )

    async for event in events:
        if event.is_final_response():
            print(event.content.parts[0].text)

asyncio.run(main())