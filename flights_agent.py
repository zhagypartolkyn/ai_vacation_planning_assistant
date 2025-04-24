import os
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the external AmadeusClient and EnhancedAirportCodeConverter
from amadeus_client import AmadeusClient
from enhanced_converters import EnhancedAirportCodeConverter


@dataclass
class FlightInfo:
    """Class to store flight information in a structured format"""

    id: str
    airline: str
    flight_number: str
    origin: Dict[str, Any]
    destination: Dict[str, Any]
    departure_time: str
    arrival_time: str
    duration: str
    stops: int
    price: Dict[str, Any]
    seats_available: Optional[int] = None
    cabin_class: str = "ECONOMY"

    def __post_init__(self):
        if self.origin is None:
            self.origin = {}
        if self.destination is None:
            self.destination = {}
        if self.price is None:
            self.price = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert flight info to dictionary"""
        return {
            "id": self.id,
            "airline": self.airline,
            "flight_number": self.flight_number,
            "origin": self.origin,
            "destination": self.destination,
            "departure_time": self.departure_time,
            "arrival_time": self.arrival_time,
            "duration": self.duration,
            "stops": self.stops,
            "price": self.price,
            "seats_available": self.seats_available,
            "cabin_class": self.cabin_class,
        }


class FlightAgent:
    """Agent for flight search using Amadeus APIs"""

    def __init__(
        self,
        client_id=None,
        client_secret=None,
        amadeus_client=None,
        airport_converter=None,
    ):
        """
        Initialize the Flight Agent

        Args:
            client_id: Amadeus API key (optional if amadeus_client is provided)
            client_secret: Amadeus API secret (optional if amadeus_client is provided)
            amadeus_client: Optional pre-configured AmadeusClient instance
            airport_converter: Optional pre-configured EnhancedAirportCodeConverter instance
        """
        # Use provided client or create a new one
        self.amadeus_client = amadeus_client or AmadeusClient(client_id, client_secret)

        # Use provided converter or create a basic one
        self.airport_converter = airport_converter or EnhancedAirportCodeConverter(
            amadeus_client=self.amadeus_client
        )

        logger.info("FlightAgent initialized successfully")

    def parse_date_range(self, date_range: str) -> Tuple[str, Optional[str]]:
        """Parse date range string into departure and return dates"""
        # Try to handle various date formats
        formats = [
            "%Y-%m-%d to %Y-%m-%d",  # 2025-06-01 to 2025-06-05
            "%d/%m/%Y to %d/%m/%Y",  # 01/06/2025 to 05/06/2025
            "%m/%d/%Y to %m/%d/%Y",  # 6/1/2025 to 6/5/2025
            "%B %d to %B %d, %Y",  # June 1 to June 5, 2025
            "%d %B to %d %B %Y",  # 1 June to 5 June 2025
        ]

        # First check if it's a one-way trip (no 'to' or '-')
        is_one_way = " to " not in date_range and " - " not in date_range

        if is_one_way:
            # Try different single date formats
            single_formats = [
                "%Y-%m-%d",  # 2025-06-01
                "%d/%m/%Y",  # 01/06/2025
                "%m/%d/%Y",  # 6/1/2025
                "%B %d, %Y",  # June 1, 2025
                "%d %B %Y",  # 1 June 2025
            ]

            for fmt in single_formats:
                try:
                    departure_date = datetime.strptime(date_range.strip(), fmt)
                    return departure_date.strftime("%Y-%m-%d"), None
                except ValueError:
                    continue
        else:
            # Round-trip with departure and return dates
            for fmt in formats:
                try:
                    # Replace 'to' with a standard separator for parsing
                    cleaned_range = date_range.replace(" to ", " - ")
                    parts = cleaned_range.split(" - ")

                    if len(parts) == 2:
                        # If second part doesn't have year, add it from first part
                        if "%Y" not in fmt.split(" to ")[1] and parts[1].count("/") < 2:
                            year = parts[0].split("/")[-1]
                            parts[1] = f"{parts[1]}/{year}"

                        # Try to parse dates
                        departure_date = datetime.strptime(
                            parts[0].strip(), fmt.split(" to ")[0]
                        )
                        return_date = datetime.strptime(
                            parts[1].strip(), fmt.split(" to ")[1]
                        )

                        # Return in YYYY-MM-DD format
                        return departure_date.strftime(
                            "%Y-%m-%d"
                        ), return_date.strftime("%Y-%m-%d")
                except ValueError:
                    continue

        # Fallback to default date (tomorrow)
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        next_week = today + timedelta(days=8)

        departure_date = tomorrow.strftime("%Y-%m-%d")
        return_date = next_week.strftime("%Y-%m-%d")

        logger.warning(
            f"Could not parse date range '{date_range}'. Using default: {departure_date} to {return_date}"
        )
        return departure_date, return_date

    def parse_travelers(self, travelers: str) -> Tuple[int, int, int]:
        """Parse travelers string into adults, children, and infants"""
        adults, children, infants = 1, 0, 0

        try:
            # Extract numbers with labels
            import re

            # Look for patterns like "2 adults", "1 child", "1 infant"
            adult_match = re.search(r"(\d+)\s*adult", travelers.lower())
            child_match = re.search(r"(\d+)\s*child", travelers.lower())
            infant_match = re.search(r"(\d+)\s*infant", travelers.lower())

            if adult_match:
                adults = int(adult_match.group(1))

            if child_match:
                children = int(child_match.group(1))

            if infant_match:
                infants = int(infant_match.group(1))

            # If no specific types found, use the first number as adults
            if not any([adult_match, child_match, infant_match]):
                digits = re.findall(r"\d+", travelers)
                if digits:
                    adults = int(digits[0])
        except Exception as e:
            logger.warning(f"Error parsing travelers '{travelers}': {e}")
            # Default to 1 adult
            adults = 1

        return adults, children, infants

    def parse_cabin_class(self, preferences: str) -> str:
        """Parse preferences to determine cabin class"""
        preferences_lower = preferences.lower()

        if "first" in preferences_lower or "1st class" in preferences_lower:
            return "FIRST"
        elif "business" in preferences_lower:
            return "BUSINESS"
        elif "premium" in preferences_lower or "premium economy" in preferences_lower:
            return "PREMIUM_ECONOMY"
        else:
            return "ECONOMY"

    def parse_budget(self, budget: str) -> float:
        """Parse budget string into a float value"""
        try:
            # Remove currency symbols and commas
            cleaned_budget = (
                budget.replace("$", "").replace(",", "").replace("USD", "").strip()
            )
            return float(cleaned_budget)
        except Exception as e:
            logger.warning(f"Error parsing budget '{budget}': {e}")
            # Default to high budget to not restrict searches
            return 10000.0

    def convert_api_results_to_flight_info(
        self, api_results: List[Dict[str, Any]], dictionaries: Dict[str, Dict[str, Any]]
    ) -> List[FlightInfo]:
        """Convert Amadeus API results to FlightInfo objects with proper city names"""
        flights = []

        # Get dictionaries
        airlines = dictionaries.get("carriers", {})
        aircraft = dictionaries.get("aircraft", {})
        locations = dictionaries.get("locations", {})

        for offer in api_results:
            offer_id = offer.get("id", "unknown")
            price_data = offer.get("price", {})

            # Get price
            price = {
                "currency": price_data.get("currency", "USD"),
                "total": price_data.get("total", "0"),
                "base": price_data.get("base", "0"),
                "fees": price_data.get("fees", []),
                "grandTotal": price_data.get("grandTotal", "0"),
            }

            # Process each itinerary
            for itinerary in offer.get("itineraries", []):
                # Process each segment (flight leg)
                for segment in itinerary.get("segments", []):
                    departure = segment.get("departure", {})
                    arrival = segment.get("arrival", {})
                    carrier = segment.get("carrierCode", "")
                    flight_number = segment.get("number", "")

                    # Get origin information with better handling
                    origin_code = departure.get("iataCode", "")
                    # Try to get location from dictionaries first
                    origin_info = locations.get(origin_code, {})
                    origin_name = origin_info.get("name")

                    # If not found in dictionaries, use our reverse lookup
                    if not origin_name:
                        try:
                            origin_name = self.airport_converter.get_city_name(
                                origin_code
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error getting city name for {origin_code}: {e}"
                            )
                            origin_name = origin_code

                    origin = {
                        "code": origin_code,
                        "name": origin_name,
                        "terminal": departure.get("terminal", ""),
                        "time": departure.get("at", ""),
                    }

                    # Get destination information with better handling
                    destination_code = arrival.get("iataCode", "")
                    # Try to get location from dictionaries first
                    destination_info = locations.get(destination_code, {})
                    destination_name = destination_info.get("name")

                    # If not found in dictionaries, use our reverse lookup
                    if not destination_name:
                        try:
                            destination_name = self.airport_converter.get_city_name(
                                destination_code
                            )
                        except Exception as e:
                            logger.warning(
                                f"Error getting city name for {destination_code}: {e}"
                            )
                            destination_name = destination_code

                    destination = {
                        "code": destination_code,
                        "name": destination_name,
                        "terminal": arrival.get("terminal", ""),
                        "time": arrival.get("at", ""),
                    }

                    # Calculate duration
                    duration = segment.get("duration", "")
                    # Convert PT1H30M format to 1h 30m format
                    if duration.startswith("PT"):
                        duration = (
                            duration[2:].replace("H", "h ").replace("M", "m").strip()
                        )

                    # Get airline name
                    airline_name = airlines.get(carrier, carrier)

                    # Create flight info object
                    flight = FlightInfo(
                        id=f"{offer_id}-{carrier}{flight_number}",
                        airline=airline_name,
                        flight_number=f"{carrier}{flight_number}",
                        origin=origin,
                        destination=destination,
                        departure_time=origin.get("time", ""),
                        arrival_time=destination.get("time", ""),
                        duration=duration,
                        stops=0,  # This is a direct segment
                        price=price,
                        cabin_class=segment.get("cabin", "ECONOMY"),
                    )

                    flights.append(flight)

        return flights

    def filter_flights_by_budget(
        self, flights: List[FlightInfo], budget: float
    ) -> List[FlightInfo]:
        """Filter flights by budget"""
        if budget <= 0:
            return flights

        filtered_flights = []

        for flight in flights:
            # Get price if available
            total_price = flight.price.get("total", "0")

            try:
                # Parse price
                if isinstance(total_price, str) and total_price:
                    price_value = float(total_price.replace("$", "").replace(",", ""))
                elif isinstance(total_price, (int, float)):
                    price_value = float(total_price)
                else:
                    # No price info, keep flight
                    filtered_flights.append(flight)
                    continue

                # Compare with budget
                if price_value <= budget:
                    filtered_flights.append(flight)
            except Exception as e:
                logger.warning(f"Error parsing flight price '{total_price}': {e}")
                # Unable to parse price, keep flight
                filtered_flights.append(flight)

        return filtered_flights

    def filter_flights_by_preferences(
        self, flights: List[FlightInfo], preferences: str
    ) -> List[FlightInfo]:
        """Filter flights based on user preferences"""
        if not preferences:
            return flights

        pref_lower = preferences.lower()

        # Check for specific airline preferences
        airline_mentions = []
        common_airlines = [
            "delta",
            "united",
            "american",
            "lufthansa",
            "british airways",
            "air france",
            "emirates",
            "singapore airlines",
            "qantas",
        ]

        for airline in common_airlines:
            if airline in pref_lower:
                airline_mentions.append(airline)

        # Check for direct flight preference
        direct_only = (
            "direct" in pref_lower
            or "non-stop" in pref_lower
            or "nonstop" in pref_lower
        )

        # Apply filters
        filtered_flights = []
        for flight in flights:
            keep = True

            # Filter by airline if specified
            if airline_mentions and not any(
                am in flight.airline.lower() for am in airline_mentions
            ):
                keep = False

            # Filter for direct flights if requested
            if direct_only and flight.stops > 0:
                keep = False

            if keep:
                filtered_flights.append(flight)

        # If no flights match the filters, return original list
        return filtered_flights if filtered_flights else flights

    def search_flights(
        self,
        origin: str,
        destination: str,
        date_range: str,
        travelers: str = "1 adult",
        preferences: str = "",
        budget: str = "$5000",
        max_results: int = 20,
    ) -> List[FlightInfo]:
        """
        Search for flights using Amadeus API

        Args:
            origin: Origin city or airport
            destination: Destination city or airport
            date_range: Date range (e.g., "2025-06-01 to 2025-06-05")
            travelers: Number and type of travelers (e.g., "2 adults, 1 child")
            preferences: Flight preferences (e.g., "direct flight, business class")
            budget: Budget (e.g., "$5000")
            max_results: Maximum number of results to return

        Returns:
            List of FlightInfo objects
        """
        logger.info(
            f"Searching for flights from {origin} to {destination} for {date_range}..."
        )
        logger.info(f"Travelers: {travelers}")
        logger.info(f"Preferences: {preferences}")
        logger.info(f"Budget: {budget}")

        # Parse dates
        departure_date, return_date = self.parse_date_range(date_range)

        # Parse travelers
        adults, children, infants = self.parse_travelers(travelers)

        # Parse cabin class from preferences
        cabin_class = self.parse_cabin_class(preferences)

        # Parse budget
        budget_value = self.parse_budget(budget)

        # Convert city names to airport codes using our enhanced converter
        origin_code = self.airport_converter.get_code(origin)
        destination_code = self.airport_converter.get_code(destination)

        logger.info(f"Using airport codes: {origin_code} -> {destination_code}")

        # Search for flights
        flight_offers = self.amadeus_client.search_flight_offers(
            origin=origin_code,
            destination=destination_code,
            departure_date=departure_date,
            return_date=return_date,
            adults=adults,
            children=children,
            infants=infants,
            travel_class=cabin_class,
            max_results=max_results,
        )

        if not flight_offers:
            logger.warning("No flight offers found.")
            return []

        # Extract dictionaries from response
        dictionaries = flight_offers[0].get("dictionaries", {}) if flight_offers else {}

        # Convert API results to FlightInfo objects
        flights = self.convert_api_results_to_flight_info(flight_offers, dictionaries)

        # Filter by budget
        flights = self.filter_flights_by_budget(flights, budget_value)

        # Filter by preferences
        flights = self.filter_flights_by_preferences(flights, preferences)

        return flights

    def execute(
        self,
        origin: str,
        destination: str,
        date_range: str,
        travelers: str = "1 adult",
        preferences: str = "",
        budget: str = "$5000",
    ) -> Dict[str, Any]:
        """
        Execute flight search and return formatted results

        Args:
            origin: Origin city or airport
            destination: Destination city or airport
            date_range: Date range (e.g., "2025-06-01 to 2025-06-05")
            travelers: Number and type of travelers (e.g., "2 adults, 1 child")
            preferences: Flight preferences (e.g., "direct flight, business class")
            budget: Budget (e.g., "$5000")

        Returns:
            Dictionary with search results
        """
        try:
            # Search for flights
            flights = self.search_flights(
                origin=origin,
                destination=destination,
                date_range=date_range,
                travelers=travelers,
                preferences=preferences,
                budget=budget,
            )

            if not flights:
                return {
                    "success": False,
                    "message": "No flights found matching your criteria.",
                    "flights": [],
                }

            # Format results
            flight_results = [flight.to_dict() for flight in flights]

            # Parse dates for response
            departure_date, return_date = self.parse_date_range(date_range)
            is_round_trip = return_date is not None

            return {
                "success": True,
                "message": f"Found {len(flights)} flights matching your criteria.",
                "search_criteria": {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                    "return_date": return_date,
                    "is_round_trip": is_round_trip,
                    "travelers": travelers,
                    "preferences": preferences,
                    "budget": budget,
                },
                "flights": flight_results,
            }
        except Exception as e:
            logger.error(f"Error executing flight search: {str(e)}")

            return {
                "success": False,
                "message": f"Error searching for flights: {str(e)}",
                "flights": [],
            }


# Helper functions for formatting the results (kept from original)
def find_flights(
    origin, destination, date_range, travelers="1 adult", preferences="", budget="$5000"
):
    """Find flights based on origin, destination and dates"""
    logger.info(f"Finding flights from {origin} to {destination} for {date_range}...")

    # Create flight agent
    agent = FlightAgent()

    # Execute search
    results = agent.execute(
        origin=origin,
        destination=destination,
        date_range=date_range,
        travelers=travelers,
        preferences=preferences,
        budget=budget,
    )

    # Return formatted results
    return format_flight_results(results)


def format_flight_results(results: Dict[str, Any]) -> str:
    """Format flight search results as a string"""
    if not results.get("success"):
        return f"❌ {results.get('message', 'Error searching for flights.')}"

    # Get search criteria
    criteria = results.get("search_criteria", {})
    origin = criteria.get("origin", "")
    destination = criteria.get("destination", "")
