import requests
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
import os

# from langchain.chat_models import ChatOpenAI
from datetime import date

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# # ------------------------- TOOLS --------------------------


from typing import List, Tuple, Dict, Any

# Optional, Any

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


class GeoapifyPlacesTool(BaseTool):
    name: str = "Geoapify Places Tool"
    description: str = (
        "Gets vacation POIs based on a destination and a list of activity keywords."
    )
    ACTIVITY_CATEGORY_MAP: dict = {
        # Outdoor activities
        "outdoor": [
            "leisure.park",
            "leisure.playground",
            "leisure.picnic",
            "sport.swimming_pool",
            "sport.pitch",
            "sport.sports_centre",
            "sport.track",
            "sport.stadium",
            "commercial.outdoor_and_sport",
            "commercial.outdoor_and_sport.water_sports",
            "commercial.outdoor_and_sport.ski",
            "commercial.outdoor_and_sport.diving",
            "commercial.outdoor_and_sport.hunting",
            "commercial.outdoor_and_sport.bicycle",
            "commercial.outdoor_and_sport.fishing",
            "commercial.outdoor_and_sport.golf",
            "commercial.vehicle",
            "entertainment.activity_park",
            "camping",
        ],
        # Cultural activities
        "cultural": [
            "tourism.sights",
            "tourism.attraction",
            "entertainment.culture",
            "entertainment.culture.theatre",
            "entertainment.culture.arts_centre",
            "entertainment.culture.gallery",
            "entertainment.museum",
            "tourism.sights.castle",
            "tourism.sights.monument",
        ],
        "shopping": [
            "commercial.supermarket",
            "commercial.marketplace",
            "commercial.shopping_mall",
            "commercial.department_store",
            "commercial.clothing",
            "commercial.clothing.clothes",
            "commercial.clothing.sport",
            "commercial.clothing.men",
            "commercial.clothing.women",
            "commercial.clothing.kids",
            "commercial.baby_goods",
            "commercial.second_hand",
            "commercial.discount_store",
        ],
        "passive": [
            "service.beauty",
            "service.beauty.hairdresser",
            "service.beauty.spa",
            "service.beauty.massage",
        ],
        # Default categories if nothing specified
        "default": ["tourism"],
    }

    def _run(self, location: str, activities: List[str]) -> str:
        # map each activity to its Geoapify categories
        cats = []
        for act in activities:
            key = act.lower()
            cats.extend(self.ACTIVITY_CATEGORY_MAP.get(key, []))
        if not cats:
            cats = self.ACTIVITY_CATEGORY_MAP["default"]
        # dedupe & limit
        cats = list(dict.fromkeys(cats))[:5]
        cat_string = ",".join(cats)

        # build & call API
        url = (
            f"https://api.geoapify.com/v2/places?"
            f"categories={cat_string}&"
            f"filter=place:{location}&"
            f"limit=60&apiKey={GEOAPIFY_API_KEY}"
        )
        resp = requests.get(url).json().get("features", [])
        if not resp:
            # fallback to lat/lon search…
            geo = (
                requests.get(
                    f"https://api.geoapify.com/v1/geocode/search?"
                    f"text={location}&format=json&apiKey={GEOAPIFY_API_KEY}"
                )
                .json()
                .get("results", [])
            )
            if not geo:
                return f"❌ Location not found: {location}"
            lat, lon = geo[0]["lat"], geo[0]["lon"]
            url = (
                f"https://api.geoapify.com/v2/places?"
                f"categories={cat_string}&"
                f"filter=circle:{lon},{lat},8000&"
                f"limit=50&apiKey={GEOAPIFY_API_KEY}"
            )
            resp = requests.get(url).json().get("features", [])

        # format into a single text blob
        out = f"📍 Suggested Activities in {location}:\n\n"
        for i, feat in enumerate(resp[:50], 1):
            p = feat["properties"]
            name = p.get("name", "Unnamed")
            addr = p.get("formatted") or ", ".join(
                filt := (
                    [p.get(k, "") for k in ("street", "city", "state", "country")]
                    if any(p.get(k) for k in ("street", "city", "state", "country"))
                    else ["No address available"]
                )
            )
            out += f"{i}. {name}\n   📪 Address: {addr}\n\n"
        return out


class TimeOptimizerTool(BaseTool):
    name: str = "Time Optimizer"
    description: str = (
        "Distributes activities into morning/afternoon/evening across days."
    )

    def _run(self, places: str) -> List[dict]:
        import re

        # 1) split the returned text into chunks per place
        lines = [l for l in places.splitlines() if l.strip()]
        raw_groups, buffer = [], []
        for line in lines:
            if re.match(r"^\d+\.\s", line):
                if buffer:
                    raw_groups.append(buffer)
                    buffer = []
            buffer.append(line)
        if buffer:
            raw_groups.append(buffer)

        # 2) flatten each place into a single string
        place_strings = ["\n".join(group) for group in raw_groups]

        # 3) chunk into days of up to 6 activities
        activities_per_day = 6
        day_chunks = [
            place_strings[i : i + activities_per_day]
            for i in range(0, len(place_strings), activities_per_day)
        ]

        # 4) build morning/afternoon/evening dicts
        optimized_days = []
        for chunk in day_chunks:
            optimized_days.append(
                {
                    "morning": chunk[0:2],  # now these are lists of strings
                    "afternoon": chunk[2:4],
                    "evening": chunk[4:6],
                }
            )
        return optimized_days


from typing import List, Union
from datetime import date
from crewai.tools import BaseTool


class ScheduleBuilderTool(BaseTool):
    name: str = "Schedule Builder"
    description: str = (
        "Formats a multi-day schedule given optimized activities and trip info."
    )

    def _run(
        self,
        optimized_days: List[dict],
        destination: str,
        start_date: date,
        end_date: date,
    ) -> str:
        num_days = (end_date - start_date).days + 1
        weekdays = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        out = f"🗓️ YOUR {num_days}-DAY {destination.upper()} ADVENTURE 🗓️\n\n"

        for i, day in enumerate(optimized_days[:num_days]):
            wd = weekdays[(start_date.weekday() + i) % 7]

            # Day header with a blank line after
            out += f"✨ DAY {i+1} ({wd}) ✨\n\n"

            # Helper to render each slot
            def render_slot(emoji: str, label: str, items: List[str]):
                nonlocal out
                if not items:
                    return
                out += f"{emoji} {label}:\n"
                for it in items:
                    out += f"- {it}\n"
                out += "\n"

            # Morning / Afternoon / Evening
            render_slot("🌅", "Morning", day.get("morning", []))
            render_slot("🌞", "Afternoon", day.get("afternoon", []))
            render_slot("🌇", "Evening", day.get("evening", []))

        # Tips footer
        out += (
            f"🌟 TIPS FOR YOUR {num_days}-DAY VISIT TO {destination.upper()} 🌟\n"
            "• Check opening hours on official sites\n"
            "• Book tickets in advance\n"
            "• Consider a transit pass\n"
            "• Check the local weather\n"
            f"• {num_days} days is perfect for exploring {destination}\n\n"
            "Enjoy your adventure! 🏙️💪🌍"
        )

        return out


def ItineraryAgent(llm):
    agent = Agent(
        role="Itinerary Planner",
        goal="Plan vacation itineraries from structured inputs.",
        backstory=(
            "You are a professional travel itinerary planner. "
            "You ALWAYS follow exactly 3 steps: "
            "1) use GeoapifyPlacesTool, "
            "2) use TimeOptimizerTool, "
            "3) use ScheduleBuilderTool."
        ),
        tools=[GeoapifyPlacesTool(), TimeOptimizerTool(), ScheduleBuilderTool()],
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    def run_itinerary(
        destination: str,
        date_range: Tuple[date, date],
        # start_date,
        # end_date,
        activities: List[str],
    ) -> str:
        start_date, end_date = date_range
        # Step 1: find places
        places_txt = GeoapifyPlacesTool()._run(destination, activities)
        # Step 2: optimize timing
        optimized = TimeOptimizerTool()._run(places_txt)
        # Step 3: build schedule
        itinerary = ScheduleBuilderTool()._run(
            optimized, destination, start_date, end_date
        )
        return itinerary

    return run_itinerary


def fetch_pois(destination: str, activities: List[str]) -> List[Dict]:
    # copy/paste your GeoapifyPlacesTool._run logic but return the raw .json()['features']
    # here’s a minimal example:
    cat_string = ",".join(["tourism"])  # build from activities …  # etc
    url = (
        f"https://api.geoapify.com/v2/places?"
        f"categories={cat_string}&"
        f"filter=place:{destination}&"
        f"limit=60&apiKey={GEOAPIFY_API_KEY}"
    )
    data = requests.get(url).json()
    return data.get("features", [])


def make_poi_df(features: List[Dict]) -> pd.DataFrame:
    rows = []
    for feat in features:
        props = feat["properties"]
        lat = props.get("lat") or feat["geometry"]["coordinates"][1]
        lon = props.get("lon") or feat["geometry"]["coordinates"][0]
        name = props.get("name", "Unnamed")
        rows.append({"lat": lat, "lon": lon, "name": name})
    return pd.DataFrame(rows)


llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=1500,
)

plan_itinerary = ItineraryAgent(llm)
