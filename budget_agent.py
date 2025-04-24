import requests
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
import os
from datetime import date
from langchain_community.chat_models import ChatOpenAI

from typing import List, Tuple, Dict, Any

import requests
import pandas as pd

def allocate_budget(min_budget: int, max_budget: int) -> str:
    """
    Allocates a total budget range into 4 categories.
    Returns:
        A markdown-formatted string summarizing each category's budget range.
    """
    allocations = {
        "✈️ Flights": 0.30,
        "🏨 Hotels": 0.40,
        "🚗 Transport": 0.10,
        "🍽️ Eating": 0.20,
        "🎫 Activities/Itinerary": 0.15
    }
    output_lines = ["Here is a budget allocation for your trip: \n"]
    output_lines.append(f"Minimum budget:  {min_budget} \n")
    output_lines.append(f"Maximum budget:  {max_budget}")
    for category, pct in allocations.items():
        low = int(min_budget * pct)
        high = int(max_budget * pct)
        output_lines.append(f"- {category}: {low} – {high}")

    return "\n".join(output_lines)