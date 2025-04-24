from datetime import date, timedelta
import streamlit as st
import os
import pandas as pd
import time
import logging
from typing import List, Dict, Any, Tuple, Optional

from langchain_community.chat_models import ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import folium
from streamlit_folium import st_folium


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our custom modules
from openai_connector import OpenAIConnector
from enhanced_budget_manager import BudgetManager
from enhanced_converters import EnhancedAirportCodeConverter, EnhancedCityCodeConverter
from amadeus_client import AmadeusClient

# Import agent modules - modified versions
from flights_agent import FlightAgent
from hotels_agent import HotelAgent
from rag_service import RAGSystem

from restaurant_agent import search_restaurants as restaurant_search
from itinerary_agent import plan_itinerary as itinerary_planner
from itinerary_agent import fetch_pois, make_poi_df
from weather_agent import plan_weather_advice
from budget_agent import allocate_budget
import pydeck as pdk


# ——————————————————————————————
#  Secrets & LLM instantiation
# ——————————————————————————————
def load_api_keys():
    """Load API keys from Streamlit secrets or environment variables"""
    # Check for the existence of secrets.toml
    try:
        keys = {
            "GEOAPIFY_API_KEY": st.secrets.get("GEOAPIFY_API_KEY")
            or os.getenv("GEOAPIFY_API_KEY"),
            "GOOGLE_PLACES_API_KEY": st.secrets.get("GOOGLE_PLACES_API_KEY")
            or os.getenv("GOOGLE_PLACES_API_KEY"),
            "AMADEUS_API_PROD_KEY": st.secrets.get("AMADEUS_API_PROD_KEY")
            or os.getenv("AMADEUS_API_PROD_KEY"),
            "AMADEUS_SECRET_PROD_KEY": st.secrets.get("AMADEUS_SECRET_PROD_KEY")
            or os.getenv("AMADEUS_SECRET_PROD_KEY"),
            "BROWSERLESS_API_KEY": st.secrets.get("BROWSERLESS_API_KEY")
            or os.getenv("BROWSERLESS_API_KEY"),
            "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            "SERPER_API_KEY": st.secrets.get("SERPER_API_KEY")
            or os.getenv("SERPER_API_KEY"),
        }
    except Exception as e:
        logger.warning(f"Error loading secrets: {e}. Using environment variables only.")
        # Fallback to environment variables only
        keys = {
            "GEOAPIFY_API_KEY": os.getenv("GEOAPIFY_API_KEY"),
            "GOOGLE_PLACES_API_KEY": os.getenv("GOOGLE_PLACES_API_KEY"),
            "AMADEUS_API_PROD_KEY": os.getenv("AMADEUS_API_PROD_KEY"),
            "AMADEUS_SECRET_PROD_KEY": os.getenv("AMADEUS_SECRET_PROD_KEY"),
            "BROWSERLESS_API_KEY": os.getenv("BROWSERLESS_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        }

    # Check for missing keys
    missing_keys = [k for k, v in keys.items() if v is None]
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")

    return keys


# Initialize API keys
api_keys = load_api_keys()

# Try to load langchain components only if OpenAI key is available
try:
    if api_keys.get("OPENAI_API_KEY"):
        from langchain_community.chat_models import ChatOpenAI
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1500,
            openai_api_key=api_keys.get("OPENAI_API_KEY"),
        )
        logger.info("LLM initialized successfully")
    else:
        llm = None
        logger.warning("OpenAI API key not found. LLM features will be disabled.")
except ImportError as e:
    logger.error(f"Error importing LangChain components: {e}")
    llm = None


def initialize_clients(api_keys):
    """Initialize API clients"""
    try:
        # Create OpenAI connector for fallbacks if API key is available
        openai_connector = None
        if api_keys.get("OPENAI_API_KEY"):
            openai_connector = OpenAIConnector(api_key=api_keys.get("OPENAI_API_KEY"))
            logger.info("OpenAI connector initialized successfully")

        # Create Amadeus client (shared between flight and hotel agents)
        amadeus_client = None
        if api_keys.get("AMADEUS_API_PROD_KEY") and api_keys.get(
            "AMADEUS_SECRET_PROD_KEY"
        ):
            amadeus_client = AmadeusClient(
                client_id=api_keys.get("AMADEUS_API_PROD_KEY"),
                client_secret=api_keys.get("AMADEUS_SECRET_PROD_KEY"),
            )
            logger.info("Amadeus client initialized successfully")
        else:
            logger.warning(
                "Amadeus API credentials not found. Flight and hotel search will be disabled."
            )

        # Create enhanced converters with fallbacks
        airport_converter = EnhancedAirportCodeConverter(
            amadeus_client=amadeus_client, openai_connector=openai_connector
        )

        city_converter = EnhancedCityCodeConverter(
            amadeus_client=amadeus_client, openai_connector=openai_connector
        )

        # Create RAG system
        rag_system = RAGSystem(vectorstore_path="./hotel_vectorstore")

        # Create agents with shared components
        flight_agent = None
        hotel_agent = None

        if amadeus_client:
            flight_agent = FlightAgent(
                amadeus_client=amadeus_client, airport_converter=airport_converter
            )

            hotel_agent = HotelAgent(
                amadeus_client=amadeus_client,
                city_converter=city_converter,
                rag_system=rag_system,
            )

        # Create budget manager (will update budget during search)
        initial_budget = 3000
        if (
            "budget_range" in st.session_state
            and st.session_state.budget_range
            and len(st.session_state.budget_range) == 2
        ):
            initial_budget = st.session_state.budget_range[
                1
            ]  # Use upper bound of budget range

        budget_manager = BudgetManager(total_budget=initial_budget)

        # Initialize LLM for itinerary planning if OpenAI API key is available
        llm = None
        itinerary_agent = None
        if api_keys.get("OPENAI_API_KEY"):
            from langchain_community.chat_models import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=1500,
                openai_api_key=api_keys.get("OPENAI_API_KEY"),
            )
            # Get the itinerary planning function
            itinerary_agent = plan_itinerary
            logger.info("Itinerary agent initialized successfully")
        else:
            logger.warning(
                "OpenAI API key not found. Itinerary planning will be disabled."
            )

        return {
            "amadeus_client": amadeus_client,
            "openai_connector": openai_connector,
            "airport_converter": airport_converter,
            "city_converter": city_converter,
            "rag_system": rag_system,
            "flight_agent": flight_agent,
            "hotel_agent": hotel_agent,
            "budget_manager": budget_manager,
            "itinerary_agent": itinerary_agent,  # Add itinerary agent
            # Restaurant agent is a function, not an object, so we'll use it directly
        }
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        logger.error(f"Error initializing clients: {str(e)}")
        return None


# ——————————————————————————————
#  Helper Functions
# ——————————————————————————————
def format_date_range(start_date, end_date):
    """Format date range for API calls"""
    return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"


def validate_inputs(origin, destination, date_range, budget):
    """Validate user inputs"""
    errors = []

    if not origin:
        errors.append("Origin is required")

    if not destination:
        errors.append("Destination is required")

    if not date_range or len(date_range) != 2:
        errors.append("Valid date range is required")
    else:
        start_date, end_date = date_range
        if start_date < date.today():
            errors.append("Start date cannot be in the past")
        if end_date < start_date:
            errors.append("End date must be after start date")
        if (end_date - start_date).days > 30:
            errors.append("Trip duration cannot exceed 30 days")

    if budget[0] < 100 or budget[1] > 20000:
        errors.append("Budget must be between $100 and $20,000")

    return errors


def search_flights(flight_agent, origin, destination, date_range, budget, preferences):
    """Search for flights with error handling"""
    try:
        # Format inputs for flight search
        date_str = format_date_range(date_range[0], date_range[1])
        budget_str = f"${budget[1]}"  # Use upper bound of budget range

        # Call flight agent
        flight_results = flight_agent.search_flights(
            origin=origin,
            destination=destination,
            date_range=date_str,
            preferences=preferences,
            budget=budget_str,
        )

        return flight_results
    except Exception as e:
        st.error(f"Error searching flights: {str(e)}")
        logger.error(f"Error searching flights: {str(e)}")
        st.info("💡 Tip: Try a different origin/destination or adjust your dates.")
        return []


def search_hotels(hotel_agent, destination, date_range, budget, preferences):
    """Search for hotels with error handling"""
    try:
        # Format inputs for hotel search
        date_str = format_date_range(date_range[0], date_range[1])
        budget_str = f"${budget[1]}"  # Use upper bound of budget range

        # Call hotel agent
        hotel_results = hotel_agent.search_hotels(
            destination=destination,
            date_range=date_str,
            preferences=preferences,
            budget=budget_str,
        )

        return hotel_results
    except Exception as e:
        st.error(f"Error searching hotels: {str(e)}")
        logger.error(f"Error searching hotels: {str(e)}")
        st.info("💡 Tip: Try a different destination or adjust your dates.")
        return []


def search_restaurants(destination, preferences):
    """Search for restaurants with error handling"""
    try:
        # Call restaurant search function with alias
        restaurant_results = restaurant_search(
            destination=destination, prefs=preferences
        )

        return restaurant_results
    except Exception as e:
        st.error(f"Error searching restaurants: {str(e)}")
        logger.error(f"Error searching restaurants: {str(e)}")
        st.info("💡 Tip: Try different cuisine preferences.")
        return "No restaurant recommendations available."


def plan_itinerary(itinerary_agent, destination, date_range, activities):
    """Plan itinerary with error handling"""
    try:
        # If itinerary_agent is None, return an error message
        if not itinerary_agent:
            logger.warning("Itinerary agent not initialized")
            return "Itinerary planning is disabled. Please check your OpenAI API key."

        # Convert string activities to list if needed
        if isinstance(activities, str):
            activities = [activities]

        # Make sure date_range is a tuple of date objects
        start_date, end_date = date_range

        # Some logging for debugging
        logger.info(
            f"Planning itinerary for {destination} from {start_date} to {end_date}"
        )
        logger.info(f"Activities: {activities}")

        # Call the plan_itinerary function from the imported module
        # This function is defined in itinerary_agent.py and returned from ItineraryAgent()
        itinerary_results = itinerary_agent(
            destination=destination,
            date_range=(start_date, end_date),
            activities=activities,
        )

        logger.info("Successfully generated itinerary")

        features = fetch_pois(destination, activities)
        poi_df = make_poi_df(features)

        if not poi_df.empty:
            st.subheader("📍 Map of Suggested Activities")

            # 1) Assign a uniform beige fill color (R=245, G=245, B=220, A=180)
            poi_df["color"] = [[245, 50, 220, 180]] * len(poi_df)

            # 2) Center view
            mid_lat = poi_df.lat.mean()
            mid_lon = poi_df.lon.mean()
            view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12)

            # 3) Scatter layer with beige markers
            scatter = pdk.Layer(
                "ScatterplotLayer",
                data=poi_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=120,  # adjust size as needed
                pickable=True,
                opacity=1,
            )

            # 4) Text layer with solid black labels
            text = pdk.Layer(
                "TextLayer",
                data=poi_df,
                pickable=False,
                get_position="[lon, lat]",
                get_text="name",
                get_size=14,
                get_color=[0, 0, 0, 255],  # black
                get_alignment_baseline="'bottom'",
            )

            # 5) Tooltip on hover
            view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=12)

            deck = pdk.Deck(
                layers=[scatter, text],
                initial_view_state=view,
                tooltip={"text": "{name}"},
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            )

            st.pydeck_chart(deck)

            return itinerary_results

    except Exception as e:
        logger.error(f"Error planning itinerary: {str(e)}")
        st.error(f"Error planning itinerary: {str(e)}")
        st.info("💡 Tip: Try fewer days or different activity preferences.")
        return (
            "Could not generate itinerary. Please try again with different parameters."
        )


def find_budget_combinations(
    budget_manager, flights, hotels, total_budget, date_range=None
):
    """Find combinations of flights and hotels within budget"""
    try:
        # if these are objects, convert them to dicts
        flights = [f.to_dict() if hasattr(f, "to_dict") else f for f in flights]
        hotels = [h.to_dict() if hasattr(h, "to_dict") else h for h in hotels]

        budget_manager.total_budget = total_budget

        # Calculate number of nights from date_range
        nights = 5  # default
        if date_range and len(date_range) == 2:
            # Calculate the number of days between dates
            start_date, end_date = date_range
            nights = (end_date - start_date).days
            logger.info(f"Calculated {nights} nights from date range")

        # Get budget statistics
        stats = budget_manager.get_budget_statistics(flights, hotels, nights=nights)

        # Find best combinations
        combinations = budget_manager.find_best_combinations(
            flights, hotels, nights=nights
        )

        return combinations, stats
    except Exception as e:
        st.error(f"Error finding combinations: {str(e)}")
        logger.error(f"Error finding combinations: {str(e)}")
        return [], {}


# ——————————————————————————————
#  Streamlit UI
# ——————————————————————————————
def build_sidebar():
    """Build sidebar with app info and settings"""
    with st.sidebar:
        st.title("🧰 App Info")
        st.info(
            """
            This AI Travel Assistant helps you plan your trip by finding:
            - ✈️ Flights from your origin to destination
            - 🏨 Hotels at your destination
            - 💰 Combined options within your budget
            - 👩🏻‍🍳 Restaurant options based on preferences
            - 🌁 Itinerary plan for every day of your journey
            - 💸 Budget allocation
            
            Everything is powered by AI and live data!
            """
        )

        st.subheader("🔍 How it works")
        st.write(
            """
            1. Enter your trip details in the form
            2. Select your preferences for flights and hotels
            3. Set your total budget
            4. Click the search button
            5. Explore options in the tabs below
            """
        )

        st.subheader("💸 Budget Allocation")
        st.write(
            """
            Your total budget is allocated as follows:
            - 35% for flights
            - 35% for hotels
            - 30% for other expenses (food, activities, etc.)
            """
        )

        # Vector store path configuration (for RAG)
        st.subheader("⚙️ Configuration")
        vector_store_path = st.text_input(
            "Vector Store Path",
            value="./hotel_vectorstore",
            help="Path to the FAISS vector store for hotel information",
        )

        st.subheader("⚠️ Troubleshooting")
        st.write(
            """
            If you encounter any issues:
            - Try different dates or destinations
            - Adjust your budget
            - Check your internet connection
            - Refresh the page
            """
        )

        return {"vector_store_path": vector_store_path}


def build_input_form():
    """Build input form for user to enter trip details"""
    st.title("🧳 AI Travel Assistant")
    st.info(
        """
        **Your personal AI-powered trip planner!**  
        Fill in your trip details below, then click the search button.
        The AI will find the best flights, hotels, restaurants, and create an itinerary within your budget.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        origin = st.text_input("↗️ Origin", placeholder="e.g. Paris, France")
        destination = st.text_input("📍 Destination", placeholder="e.g. Rome, Italy")
        date_range = st.date_input(
            "📅 Travel Dates",
            value=(
                date.today() + timedelta(days=30),
                date.today() + timedelta(days=37),
            ),
            min_value=date.today(),
            key="dates",
        )

    with col2:
        budget = st.slider("💰 Budget Range (USD)", 500, 10000, (1000, 3000), step=100)
        flight_pref = st.text_input(
            "✈️ Flight Preferences", placeholder="e.g. direct flight, business class"
        )
        hotel_pref = st.text_input(
            "🏨 Hotel Preferences", placeholder="e.g. 4-star, pool, breakfast included"
        )

    # Add restaurant preferences
    rest_pref = st.text_input(
        "🍴 Restaurant Preferences",
        placeholder="e.g. Italian cuisine, vegetarian options",
    )

    travelers = st.select_slider(
        "👥 Number of Travelers",
        options=[
            "1 adult",
            "2 adults",
            "2 adults, 1 child",
            "2 adults, 2 children",
            "Family (4+)",
        ],
        value="2 adults",
    )

    activity_pref = st.multiselect(
        "🎯 Activity Preferences",
        options=[
            "outdoor",
            "cultural",
            "shopping",
            "religious places",
            "passive",
            "active",
        ],
        default=["cultural"],
    )

    # Collect all inputs
    inputs = {
        "origin": origin,
        "destination": destination,
        "date_range": date_range,
        "budget": budget,
        "flight_pref": flight_pref,
        "hotel_pref": hotel_pref,
        "rest_pref": rest_pref,
        "travelers": travelers,
        "activity_pref": activity_pref,
    }

    return inputs


def display_flight_results(flights):
    """Display flight search results with improved handling of missing location names"""
    if not flights:
        st.warning("No flights found matching your criteria.")
        return

    st.subheader(f"✈️ Found {len(flights)} flights")

    # Create tabs for outbound and return flights
    flight_tabs = st.tabs(["All Flights", "Outbound", "Return"])

    with flight_tabs[0]:
        # Display all flights in a table
        flight_data = []
        for flight in flights:
            # Extract key information with better handling for missing values
            airline = flight.get("airline", "Unknown")
            flight_number = flight.get("flight_number", "")

            # Get origin name, falling back to code if name is missing
            origin_info = flight.get("origin", {})
            origin = (
                origin_info.get("name")
                if origin_info.get("name")
                else origin_info.get("code", "Unknown")
            )

            # Get destination name, falling back to code if name is missing
            destination_info = flight.get("destination", {})
            destination = (
                destination_info.get("name")
                if destination_info.get("name")
                else destination_info.get("code", "Unknown")
            )

            departure = flight.get("departure_time", "")
            arrival = flight.get("arrival_time", "")
            duration = flight.get("duration", "")
            stops = flight.get("stops", 0)

            # Handle price formatting more robustly
            price_info = flight.get("price", {})
            if isinstance(price_info, dict):
                price = price_info.get("total", "Unknown")
            else:
                price = str(price_info)

            flight_data.append(
                {
                    "Airline": f"{airline} {flight_number}",
                    "Route": f"{origin} → {destination}",
                    "Departure": departure,
                    "Arrival": arrival,
                    "Duration": duration,
                    "Stops": stops,
                    "Price": price,
                }
            )

        st.dataframe(pd.DataFrame(flight_data))

    with flight_tabs[1]:
        # Filter to outbound flights only - these typically have origin code matching the user's origin
        # Use the first flight's origin as the reference for outbound
        if flights:
            reference_origin = flights[0].get("origin", {}).get("code", "").upper()
            outbound = [
                f
                for f in flights
                if f.get("origin", {}).get("code", "").upper() == reference_origin
            ]
            if outbound:
                for i, flight in enumerate(outbound):
                    # Get origin and destination with fallbacks
                    origin_info = flight.get("origin", {})
                    origin_name = (
                        origin_info.get("name")
                        if origin_info.get("name")
                        else origin_info.get("code", "Unknown")
                    )

                    destination_info = flight.get("destination", {})
                    destination_name = (
                        destination_info.get("name")
                        if destination_info.get("name")
                        else destination_info.get("code", "Unknown")
                    )

                    with st.expander(
                        f"Flight {i+1}: {flight.get('airline')} {flight.get('flight_number')}"
                    ):
                        st.write(f"**From:** {origin_name}")
                        st.write(f"**To:** {destination_name}")
                        st.write(f"**Departure:** {flight.get('departure_time', '')}")
                        st.write(f"**Arrival:** {flight.get('arrival_time', '')}")
                        st.write(f"**Duration:** {flight.get('duration', '')}")
                        st.write(f"**Stops:** {flight.get('stops', 0)}")

                        # Handle price formatting more robustly
                        price_info = flight.get("price", {})
                        if isinstance(price_info, dict):
                            price = price_info.get("total", "Unknown")
                        else:
                            price = str(price_info)
                        st.write(f"**Price:** {price}")

                        st.write(f"**Class:** {flight.get('cabin_class', 'Economy')}")
            else:
                st.info("No outbound flights to display separately.")

    with flight_tabs[2]:
        # Filter to return flights only - these typically have origin code matching the user's destination
        # Use the first flight's destination as the reference for returns
        if flights:
            reference_dest = flights[0].get("destination", {}).get("code", "").upper()
            return_flights = [
                f
                for f in flights
                if f.get("origin", {}).get("code", "").upper() == reference_dest
            ]
            if return_flights:
                for i, flight in enumerate(return_flights):
                    # Get origin and destination with fallbacks
                    origin_info = flight.get("origin", {})
                    origin_name = (
                        origin_info.get("name")
                        if origin_info.get("name")
                        else origin_info.get("code", "Unknown")
                    )

                    destination_info = flight.get("destination", {})
                    destination_name = (
                        destination_info.get("name")
                        if destination_info.get("name")
                        else destination_info.get("code", "Unknown")
                    )

                    with st.expander(
                        f"Flight {i+1}: {flight.get('airline')} {flight.get('flight_number')}"
                    ):
                        st.write(f"**From:** {origin_name}")
                        st.write(f"**To:** {destination_name}")
                        st.write(f"**Departure:** {flight.get('departure_time', '')}")
                        st.write(f"**Arrival:** {flight.get('arrival_time', '')}")
                        st.write(f"**Duration:** {flight.get('duration', '')}")
                        st.write(f"**Stops:** {flight.get('stops', 0)}")

                        # Handle price formatting more robustly
                        price_info = flight.get("price", {})
                        if isinstance(price_info, dict):
                            price = price_info.get("total", "Unknown")
                        else:
                            price = str(price_info)
                        st.write(f"**Price:** {price}")

                        st.write(f"**Class:** {flight.get('cabin_class', 'Economy')}")
            else:
                st.info("No return flights to display separately.")


def display_hotel_results(hotels):
    """Display hotel search results with improved handling"""
    if not hotels:
        st.warning("No hotels found matching your criteria.")
        return

    st.subheader(f"🏨 Found {len(hotels)} hotels")

    # Create a list of hotel cards
    for i, hotel in enumerate(hotels):
        # Clean up hotel name - remove any amenities text accidentally included
        hotel_name = hotel.get("name", "Unknown Hotel")
        if "," in hotel_name and len(hotel_name.split(",")) > 1:
            hotel_name = hotel_name.split(",")[0].strip()

        # Create expander with clean hotel name
        with st.expander(f"Hotel {i+1}: {hotel_name}"):
            # Create two columns for hotel info
            col1, col2 = st.columns([2, 1])

            with col1:
                # Show hotel description, truncated if too long
                description = hotel.get("description", "No description available.")
                if description and len(description) > 300:
                    description = description[:300] + "..."
                st.write(f"**Description:** {description}")

                # Location info
                location = hotel.get("location", {})
                loc_parts = []
                if location.get("city"):
                    loc_parts.append(location.get("city"))
                if location.get("country"):
                    loc_parts.append(location.get("country"))

                if loc_parts:
                    st.write(f"**Location:** {', '.join(loc_parts)}")

                # Amenities with better handling
                amenities = hotel.get("amenities", [])

                # If amenities is somehow a string, convert to list
                if isinstance(amenities, str):
                    if "," in amenities:
                        amenities = [a.strip() for a in amenities.split(",")]
                    else:
                        amenities = [amenities.strip()]

                # Ensure we have a list
                if not isinstance(amenities, list):
                    amenities = []

                # Clean amenities - less strict filtering
                clean_amenities = []
                for amenity in amenities:
                    if isinstance(amenity, str):
                        # Keep amenity strings but limit length to avoid display issues
                        # Using 100 chars instead of 50 to be more lenient
                        clean_amenity = amenity.strip()
                        if len(clean_amenity) <= 100:
                            clean_amenities.append(clean_amenity)
                        else:
                            # For long strings, truncate rather than omit
                            clean_amenities.append(clean_amenity[:97] + "...")
                    elif isinstance(amenity, dict) and "name" in amenity:
                        # Handle case where amenity might be an object with name
                        clean_amenities.append(amenity["name"])

                # Display amenities (or message if none)
                if clean_amenities:
                    amenities_str = ", ".join(clean_amenities[:10])
                    if len(clean_amenities) > 10:
                        amenities_str += f" and {len(clean_amenities) - 10} more"
                    st.write(f"**Amenities:** {amenities_str}")
                else:
                    st.write("**Amenities:** Information not available")

            with col2:
                # Price
                price = hotel.get("price", {})
                if price and price.get("total"):
                    currency = price.get("currency", "USD")
                    st.write(f"**Price:** {price.get('total')} {currency} per night")
                else:
                    st.write("**Price:** Information not available")

                # Rating
                rating = hotel.get("rating")
                if rating:
                    st.write(f"**Rating:** {rating}/5")
                else:
                    st.write("**Rating:** Not rated")

                # Source
                source = hotel.get("source", "Unknown")
                st.write(f"**Source:** {source}")

                # Add sentiment category for RAG results
                if source == "RAG":
                    sentiment = hotel.get("description_sentiment_category", "")
                    if sentiment:
                        st.write(f"**Quality:** {sentiment}")

                # ID (useful for debugging)
                st.write(f"**ID:** {hotel.get('id', 'Unknown')}")

    # Add a map if we have location data
    hotels_with_coords = [
        hotel
        for hotel in hotels
        if hotel.get("location", {}).get("latitude")
        and hotel.get("location", {}).get("longitude")
    ]

    if hotels_with_coords:
        st.subheader("Hotel Locations")
        try:
            df = pd.DataFrame(
                [
                    {
                        "name": hotel.get("name", "Unknown").split(",")[
                            0
                        ],  # Clean name for map
                        "lat": float(hotel.get("location", {}).get("latitude")),
                        "lon": float(hotel.get("location", {}).get("longitude")),
                        "price": hotel.get("price", {}).get("total", "N/A"),
                    }
                    for hotel in hotels_with_coords
                ]
            )
            st.map(df)
        except Exception as e:
            st.error(f"Error displaying map: {str(e)}")
            st.info(
                "💡 Tip: Some hotel coordinates may be invalid. Try searching for different hotels."
            )


def display_combined_results(combinations, stats):
    """Display combined flight and hotel results"""
    if not combinations:
        st.warning("No combinations found matching your criteria.")
        return

    # Show budget statistics
    if stats:
        nights = stats.get("nights", 5)

        st.info("🔍 Budget Analysis")
        st.write(f"• Minimum flight price: ${stats.get('min_flight_price', 0):.2f}")
        st.write(
            f"• Minimum hotel price: ${stats.get('min_hotel_price', 0):.2f} per night (${stats.get('min_hotel_total', 0):.2f} total for {nights} nights)"
        )
        st.write(f"• Minimum total price: ${stats.get('min_total_price', 0):.2f}")
        st.write(
            f"• Your travel budget: ${stats.get('travel_budget', 0):.2f} (70% of total budget)"
        )

        # Show warning if budget is insufficient
        if not stats.get("budget_sufficient", True):
            st.warning(
                f"""
            ⚠️ **Budget Warning**
            Your current budget may not be sufficient for this trip.
            Recommended minimum budget: ${stats.get('recommended_budget', 0):.2f}
            """
            )

    # Display combinations
    st.subheader(f"🔀 Found {len(combinations)} combinations")

    for i, combo in enumerate(combinations):
        # Extract data
        flight = combo.get("flight", {})
        hotel = combo.get("hotel", {})
        flight_price = combo.get("flight_price", 0)
        hotel_price = combo.get("hotel_price", 0)  # Per night
        total_hotel_price = combo.get("total_hotel_price", 0)  # Total for stay
        total_price = combo.get("total_price", 0)
        within_budget = combo.get("within_budget", False)
        nights = combo.get("nights", 5)

        # Create an expander for each combination
        icon = "✅" if within_budget else "⚠️"
        with st.expander(f"{icon} Option {i+1}: ${total_price:.2f} total"):
            # Create columns for flight and hotel
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✈️ Flight")
                st.write(
                    f"**Airline:** {flight.get('airline', 'Unknown')} {flight.get('flight_number', '')}"
                )
                st.write(f"**From:** {flight.get('origin', {}).get('name', 'Unknown')}")
                st.write(
                    f"**To:** {flight.get('destination', {}).get('name', 'Unknown')}"
                )
                st.write(f"**Departure:** {flight.get('departure_time', '')}")
                st.write(f"**Duration:** {flight.get('duration', '')}")
                st.write(f"**Price:** ${flight_price:.2f}")

            with col2:
                st.subheader("🏨 Hotel")
                st.write(f"**Name:** {hotel.get('name', 'Unknown Hotel')}")
                location = hotel.get("location", {})
                loc_parts = []
                if location.get("city"):
                    loc_parts.append(location.get("city"))
                if location.get("country"):
                    loc_parts.append(location.get("country"))

                if loc_parts:
                    st.write(f"**Location:** {', '.join(loc_parts)}")

                # Show per-night and total prices

                st.write(f"**Price:** ${hotel_price:.2f} per night")
                st.write(
                    f"**Total for stay:** ${total_hotel_price:.2f} for {nights} nights"
                )

                # Show data source (API or RAG)
                source = hotel.get("source", "Unknown")
                st.write(f"**Source:** {source}")

                # Show sentiment category if available (for RAG results)
                if source == "RAG":
                    sentiment = hotel.get(
                        "sentiment_category",
                        hotel.get("description_sentiment_category", ""),
                    )
                    if sentiment:
                        st.write(f"**Quality:** {sentiment}")

                # Show a few amenities
                amenities = hotel.get("amenities", [])
                if amenities:
                    amenities_str = ", ".join(amenities[:5])
                    if len(amenities) > 5:
                        amenities_str += f" and {len(amenities) - 5} more"
                    st.write(f"**Amenities:** {amenities_str}")

            # Summary footer
            st.write(f"**Total Price:** ${total_price:.2f}")
            if not within_budget:
                st.warning(
                    f"⚠️ This option exceeds your allocated travel budget by ${total_price - (stats.get('travel_budget', 0)):.2f}"
                )
            else:
                remaining = (stats.get("travel_budget", 0)) - total_price
                st.success(
                    f"✅ This option is within budget! You'll have ${remaining:.2f} left from your travel budget."
                )


def main():
    """Main application function"""
    # Build sidebar and get configuration
    config = build_sidebar()

    # Load API keys
    api_keys = load_api_keys()

    # Initialize session state for results storage
    for key in [
        "flight_results",
        "hotel_results",
        "combined_results",
        "budget_stats",
        "restaurant_results",
        "itinerary",
        "weather_advice",
        "budget_allocation",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    # Check if any required API keys are missing
    required_keys = ["AMADEUS_API_PROD_KEY", "AMADEUS_SECRET_PROD_KEY"]
    missing_keys = [key for key in required_keys if not api_keys.get(key)]

    if missing_keys:
        st.error(f"Missing required API keys: {', '.join(missing_keys)}")
        st.stop()

    # Initialize clients
    clients = initialize_clients(api_keys)
    if not clients:
        st.error("Failed to initialize API clients. Please check your API keys.")
        st.stop()

    # Update RAG system path from config if needed
    if clients["rag_system"].vectorstore_path != config["vector_store_path"]:
        clients["rag_system"] = RAGSystem(vectorstore_path=config["vector_store_path"])
        clients["hotel_agent"].rag_system = clients["rag_system"]

    # Get user inputs
    inputs = build_input_form()

    # Create search button
    search_button = st.button("🔍 Search for Trip Options")

    # Process search when button is clicked
    if search_button:
        # Validate inputs
        errors = validate_inputs(
            inputs["origin"],
            inputs["destination"],
            inputs["date_range"],
            inputs["budget"],
        )

        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        # Search progress bar
        progress_bar = st.progress(0)

        # Search for flights
        with st.spinner("Searching for flights..."):
            flight_results = search_flights(
                clients["flight_agent"],
                inputs["origin"],
                inputs["destination"],
                inputs["date_range"],
                inputs["budget"],
                inputs["flight_pref"],
            )
            # Convert FlightInfo objects into dicts
            st.session_state.flight_results = [
                f.to_dict() if hasattr(f, "to_dict") else f for f in flight_results
            ]
            progress_bar.progress(20)

        # Search for hotels
        with st.spinner("Searching for hotels..."):
            hotel_results = search_hotels(
                clients["hotel_agent"],
                inputs["destination"],
                inputs["date_range"],
                inputs["budget"],
                inputs["hotel_pref"],
            )
            st.session_state.hotel_results = [
                h.to_dict() if hasattr(h, "to_dict") else h for h in hotel_results
            ]
            progress_bar.progress(40)

        # Get budget allocation using enhanced budget manager
        with st.spinner("Allocating budget..."):
            try:
                # Use the enhanced budget manager to get detailed allocation
                budget_allocation = clients[
                    "budget_manager"
                ].allocate_budget_from_agent(
                    min_budget=inputs["budget"][0], max_budget=inputs["budget"][1]
                )
                st.session_state.budget_allocation = budget_allocation

                # Find budget combinations using the adjusted budget
                combinations, stats = find_budget_combinations(
                    clients["budget_manager"],
                    st.session_state.flight_results,
                    st.session_state.hotel_results,
                    clients[
                        "budget_manager"
                    ].total_budget,  # Use possibly adjusted budget
                    inputs["date_range"],
                )
                st.session_state.combined_results = combinations
                st.session_state.budget_stats = stats
            except Exception as e:
                logger.error(f"Error allocating budget: {str(e)}")
                st.session_state.budget_allocation = (
                    f"Error allocating budget: {str(e)}"
                )
            progress_bar.progress(60)

        # Search for restaurants
        with st.spinner("Searching for restaurants..."):
            try:
                restaurant_results = restaurant_search(
                    destination=inputs["destination"],
                    prefs=inputs.get(
                        "rest_pref", "popular"
                    ),  # Use a default if rest_pref not provided
                )
                st.session_state.restaurant_results = restaurant_results
            except Exception as e:
                logger.error(f"Error searching restaurants: {str(e)}")
                st.session_state.restaurant_results = (
                    f"Error searching restaurants: {str(e)}"
                )
            progress_bar.progress(70)

        # Plan itinerary
        with st.spinner("Planning your itinerary..."):
            try:
                # Import the itinerary planner function directly
                from itinerary_agent import plan_itinerary

                # Call the function directly without passing itinerary_agent parameter
                itinerary = plan_itinerary(
                    destination=inputs["destination"],
                    date_range=(inputs["date_range"][0], inputs["date_range"][1]),
                    activities=inputs.get("activity_pref", ["cultural", "outdoor"]),
                )
                st.session_state.itinerary = itinerary
            except Exception as e:
                logger.error(f"Error planning itinerary: {str(e)}")
                st.session_state.itinerary = f"Error planning itinerary: {str(e)}"
            progress_bar.progress(85)

        # Get weather advice
        with st.spinner("Getting weather advice..."):
            try:
                weather_advice = plan_weather_advice(
                    location=inputs["destination"],
                    date_range=(inputs["date_range"][0], inputs["date_range"][1]),
                )
                st.session_state.weather_advice = weather_advice
            except Exception as e:
                logger.error(f"Error getting weather advice: {str(e)}")
                st.session_state.weather_advice = (
                    f"Error getting weather advice: {str(e)}"
                )
            progress_bar.progress(100)

    # Display results in tabs if search has been performed
    if (
        st.session_state.flight_results is not None
        or st.session_state.hotel_results is not None
    ):
        st.markdown("---")

        # Create tabs for different result categories
        results_tabs = st.tabs(
            [
                "Combined Options",
                "Flights",
                "Hotels",
                "Restaurants",
                "Itinerary",
                "Weather",
                "Budget",
            ]
        )

        with results_tabs[0]:
            display_combined_results(
                st.session_state.combined_results, st.session_state.budget_stats
            )

        with results_tabs[1]:
            display_flight_results(st.session_state.flight_results)

        with results_tabs[2]:
            display_hotel_results(st.session_state.hotel_results)

        with results_tabs[3]:
            if st.session_state.restaurant_results:
                st.markdown(st.session_state.restaurant_results)
            else:
                st.warning("No restaurant recommendations available.")

        with results_tabs[4]:
            if st.session_state.itinerary:
                st.markdown(st.session_state.itinerary)
            else:
                st.warning("No itinerary available.")

        with results_tabs[5]:
            if st.session_state.weather_advice:
                st.markdown(st.session_state.weather_advice)
            else:
                st.warning("No weather advice available.")

        with results_tabs[6]:
            if st.session_state.budget_allocation:
                # st.markdown(st.session_state.budget_allocation)

                # Add budget adjustment controls
                st.subheader("Adjust Budget Allocation")

                col1, col2 = st.columns(2)
                with col1:
                    flight_percent = (
                        st.slider(
                            "✈️ Flight Budget %",
                            min_value=10,
                            max_value=60,
                            value=int(clients["budget_manager"].flight_percent * 100),
                            step=5,
                        )
                        / 100.0
                    )

                with col2:
                    hotel_percent = (
                        st.slider(
                            "🏨 Hotel Budget %",
                            min_value=10,
                            max_value=60,
                            value=int(clients["budget_manager"].hotel_percent * 100),
                            step=5,
                        )
                        / 100.0
                    )

                    # Step 2: compute the rest of the categories automatically
                min_bud, max_bud = inputs["budget"]  # your original slider values
                allocation_text = allocate_budget(
                    min_bud, max_bud, flight_percent, hotel_percent
                )

                # Step 3: show it in one nice info‐box
                st.info(allocation_text, icon="💸")

                if st.button("Update Budget Allocation"):
                    # Update budget manager
                    clients["budget_manager"].update_budget_allocation(
                        # flight_percent=flight_percent / 100,
                        # hotel_percent=hotel_percent / 100,
                        flight_percent,
                        hotel_percent,
                    )

                    # Update combinations with new allocation
                    if (
                        st.session_state.flight_results
                        and st.session_state.hotel_results
                    ):
                        combinations, stats = find_budget_combinations(
                            clients["budget_manager"],
                            st.session_state.flight_results,
                            st.session_state.hotel_results,
                            clients["budget_manager"].total_budget,
                            inputs["date_range"],
                        )
                        st.session_state.combined_results = combinations
                        st.session_state.budget_stats = stats

                        # Show success message with rerun to update the UI
                        st.success(
                            "Budget allocation updated! Check the Combined Options tab."
                        )
                    # st.experimental_rerun()
            else:
                st.warning("No budget allocation available.")


if __name__ == "__main__":
    main()
