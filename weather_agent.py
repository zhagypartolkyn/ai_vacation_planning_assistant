import os
from datetime import date
from typing import Tuple
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage
)
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def plan_weather_advice(location: str, date_range: Tuple[date, date]) -> str:
    prompt = (
        f"Give me historical weather patterns for {location} between "
        f"{date_range[0].strftime('%B %d')} and {date_range[1].strftime('%B %d')}. "
        "Include typical precipitation, temperature ranges, and packing advice. Add relevant emojis."
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=300,
    )

    messages = [
        SystemMessage(content="You are a travel assistant expert in historical weather."),
        HumanMessage(content=prompt)
    ]

    llm_result = llm.generate([messages])
    text = llm_result.generations[0][0].text
    return text.strip()