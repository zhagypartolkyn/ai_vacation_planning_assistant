import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import external dependencies
from amadeus_client import AmadeusClient
from enhanced_converters import EnhancedCityCodeConverter


@dataclass
class HotelInfo:
    """Class to store hotel information in a structured format"""

    id: str
    name: str
    description: str = ""
    amenities: List[str] = None
    rating: Optional[float] = None
    price: Optional[Dict[str, Any]] = None
    location: Optional[Dict[str, Any]] = None
    source: str = "RAG"  # 'RAG' or 'API'
    description_sentiment_category: str = ""  # Added this field

    def __post_init__(self):
        if self.amenities is None:
            self.amenities = []
        if self.price is None:
            self.price = {}
        if self.location is None:
            self.location = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert hotel info to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "amenities": self.amenities,
            "rating": self.rating,
            "price": self.price,
            "location": self.location,
            "source": self.source,
            "description_sentiment_category": self.description_sentiment_category,
        }


class HotelAgent:
    """Agent for hotel search using RAG and Amadeus APIs"""

    def __init__(
        self,
        client_id=None,
        client_secret=None,
        amadeus_client=None,
        city_converter=None,
        rag_system=None,
        vectorstore_path=None,
    ):
        """
        Initialize the Hotel Agent

        Args:
            client_id: Amadeus API key (optional if amadeus_client is provided)
            client_secret: Amadeus API secret (optional if amadeus_client is provided)
            amadeus_client: Optional pre-configured AmadeusClient instance
            city_converter: Optional pre-configured EnhancedCityCodeConverter instance
            rag_system: Optional pre-configured RAGSystem instance
            vectorstore_path: Path to the vector store (used if rag_system not provided)
        """
        # Use provided client or create a new one
        self.amadeus_client = amadeus_client or AmadeusClient(client_id, client_secret)

        # Use provided converter or create a basic one
        self.city_converter = city_converter or EnhancedCityCodeConverter(
            amadeus_client=self.amadeus_client
        )

        # Use provided RAG system or create a new one
        self.rag_system = rag_system
        if not self.rag_system:
            # Import here to avoid circular imports
            from rag_service import RAGSystem

            self.rag_system = RAGSystem(vectorstore_path=vectorstore_path)

        logger.info("HotelAgent initialized successfully")

    def parse_date_range(self, date_range: str) -> Tuple[str, str]:
        """Parse date range string into check-in and check-out dates"""
        # Try to handle various date formats
        formats = [
            "%Y-%m-%d to %Y-%m-%d",  # 2025-06-01 to 2025-06-05
            "%d/%m/%Y to %d/%m/%Y",  # 01/06/2025 to 05/06/2025
            "%m/%d/%Y to %m/%d/%Y",  # 6/1/2025 to 6/5/2025
            "%B %d to %B %d, %Y",  # June 1 to June 5, 2025
            "%d %B to %d %B %Y",  # 1 June to 5 June 2025
        ]

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
                    check_in_date = datetime.strptime(
                        parts[0].strip(), fmt.split(" to ")[0]
                    )
                    check_out_date = datetime.strptime(
                        parts[1].strip(), fmt.split(" to ")[1]
                    )

                    # Return in YYYY-MM-DD format
                    return check_in_date.strftime("%Y-%m-%d"), check_out_date.strftime(
                        "%Y-%m-%d"
                    )
            except ValueError:
                continue

        # Fallback to default dates (one month from today)
        today = datetime.now()
        check_in_date = (today + timedelta(days=30)).strftime("%Y-%m-%d")
        check_out_date = (today + timedelta(days=35)).strftime("%Y-%m-%d")

        logger.warning(
            f"Could not parse date range '{date_range}'. Using default: {check_in_date} to {check_out_date}"
        )
        return check_in_date, check_out_date

    def parse_travelers(self, travelers: str) -> int:
        """Parse travelers string into number of adults"""
        try:
            # Extract digits from string
            import re

            digits = re.findall(r"\d+", travelers)
            if digits:
                return int(digits[0])
        except Exception as e:
            logger.warning(f"Error parsing travelers '{travelers}': {e}")
            pass

        # Default to 2 adults
        return 2

    def extract_amenities_from_preferences(self, preferences: str) -> List[str]:
        """Extract amenities from preferences string"""
        # Common amenities to look for
        common_amenities = [
            "wifi",
            "pool",
            "spa",
            "gym",
            "fitness",
            "restaurant",
            "bar",
            "breakfast",
            "parking",
            "airport shuttle",
            "beach",
            "concierge",
            "room service",
            "balcony",
            "view",
            "business center",
            "conference",
            "pet friendly",
            "family",
            "kids club",
            "kitchen",
            "wheelchair",
            "accessible",
        ]

        found_amenities = []
        pref_lower = preferences.lower()

        for amenity in common_amenities:
            if amenity in pref_lower:
                found_amenities.append(amenity)

        return found_amenities

    def convert_api_results_to_hotel_info(
        self,
        api_results: List[Dict[str, Any]],
        offers_data: List[Dict[str, Any]] = None,
    ) -> List[HotelInfo]:
        """Convert Amadeus API results to HotelInfo objects"""
        hotels = []
        offers_by_hotel = {}

        # Organize offers by hotel ID
        if offers_data:
            for offer in offers_data:
                hotel_id = offer.get("hotel", {}).get("hotelId")
                if hotel_id:
                    offers_by_hotel[hotel_id] = offer

        for hotel_data in api_results:
            hotel_id = hotel_data.get("hotelId")

            # Skip if no hotel ID
            if not hotel_id:
                continue

            # Get basic hotel info
            name = hotel_data.get("name", "Unknown Hotel")

            # Get location info
            location = {
                "city": hotel_data.get("cityName", ""),
                "country": hotel_data.get("countryCode", ""),
                "address": hotel_data.get("address", {}).get("lines", [""])[0],
                "latitude": hotel_data.get("latitude"),
                "longitude": hotel_data.get("longitude"),
            }

            # Get price if available in offers
            price = {}
            if hotel_id in offers_by_hotel:
                offer = offers_by_hotel[hotel_id]
                offer_price = offer.get("offers", [{}])[0].get("price", {})

                price = {
                    "currency": offer_price.get("currency", "USD"),
                    "total": offer_price.get("total", ""),
                    "base": offer_price.get("base", ""),
                    "taxes": offer_price.get("taxes", []),
                    "variations": offer_price.get("variations", {}),
                }

            # Get amenities
            amenities = []
            if "amenities" in hotel_data:
                amenities = hotel_data.get("amenities", [])

            # Create hotel info object
            hotel = HotelInfo(
                id=hotel_id,
                name=name,
                description=hotel_data.get("description", {}).get("text", ""),
                amenities=amenities,
                rating=hotel_data.get("rating"),
                price=price,
                location=location,
                source="API",
            )

            hotels.append(hotel)

        return hotels

    def filter_hotels_by_amenities(
        self, hotels: List[HotelInfo], required_amenities: List[str]
    ) -> List[HotelInfo]:
        """Filter hotels by required amenities"""
        if not required_amenities:
            return hotels

        filtered_hotels = []

        for hotel in hotels:
            hotel_amenities = hotel.amenities

            # Convert to lowercase for case-insensitive comparison
            hotel_amenities_lower = [a.lower() for a in hotel_amenities]
            required_lower = [a.lower() for a in required_amenities]

            # Count matches
            matches = 0
            for req in required_lower:
                # Check if any amenity contains the required one
                if any(req in ha or ha in req for ha in hotel_amenities_lower):
                    matches += 1

            match_ratio = (
                matches / len(required_amenities) if required_amenities else 1.0
            )

            # Keep if at least 50% match
            if match_ratio >= 0.5:
                # Add match ratio to hotel for sorting
                hotel.amenity_match_ratio = match_ratio
                filtered_hotels.append(hotel)

        # Sort by match ratio
        return sorted(
            filtered_hotels,
            key=lambda h: getattr(h, "amenity_match_ratio", 0),
            reverse=True,
        )

    def filter_hotels_by_budget(
        self, hotels: List[HotelInfo], budget: float
    ) -> List[HotelInfo]:
        """Filter hotels by budget"""
        if budget <= 0:
            return hotels

        filtered_hotels = []

        for hotel in hotels:
            # Get price if available
            price = hotel.price.get("total", "")

            try:
                # Parse price
                if isinstance(price, str) and price:
                    price = float(price.replace("$", "").replace(",", ""))
                elif isinstance(price, (int, float)):
                    price = float(price)
                else:
                    # No price info, keep hotel
                    filtered_hotels.append(hotel)
                    continue

                # Compare with budget
                if price <= budget:
                    filtered_hotels.append(hotel)
            except Exception as e:
                logger.warning(f"Error parsing hotel price '{price}': {e}")
                # Unable to parse price, keep hotel
                filtered_hotels.append(hotel)

        return filtered_hotels

    def search_hotels(
        self,
        destination: str,
        date_range: str,
        preferences: str,
        budget: str = "$3000",
        travelers: str = "2 adults",
        use_rag: bool = True,
        use_api: bool = True,
        min_results: int = 3,
        max_results: int = 5,
    ) -> List[HotelInfo]:
        """
        Search for hotels using RAG and Amadeus APIs

        Args:
            destination: City or destination
            date_range: Date range (e.g., "2025-06-01 to 2025-06-05")
            preferences: Hotel preferences (e.g., "luxury hotel with spa")
            budget: Budget per night (e.g., "$3000")
            travelers: Number of travelers (e.g., "2 adults")
            use_rag: Whether to use RAG
            use_api: Whether to use Amadeus API
            min_results: Minimum number of results
            max_results: Maximum number of results

        Returns:
            List of HotelInfo objects
        """
        logger.info(
            f"Searching for hotels in {destination} from {date_range} for {travelers}..."
        )
        logger.info(f"Preferences: {preferences}")
        logger.info(f"Budget: {budget}")

        # Parse budget
        try:
            budget_value = float(budget.replace("$", "").replace(",", ""))
        except Exception as e:
            logger.warning(f"Error parsing budget '{budget}': {e}")
            budget_value = 300.0
            logger.info(f"Using default budget: ${budget_value}")

        # Parse dates
        check_in_date, check_out_date = self.parse_date_range(date_range)

        # Parse travelers
        adults = self.parse_travelers(travelers)

        # Extract amenities from preferences
        required_amenities = self.extract_amenities_from_preferences(preferences)
        logger.info(f"Looking for amenities: {required_amenities}")

        all_hotels = []

        # Step 1: Try RAG first with improved city matching
        if use_rag:
            logger.info("Searching with RAG system...")
            rag_query = f"{preferences} in {destination}"

            # Use the improved query_rag method that filters by city
            rag_results = self.rag_system.query_rag(
                rag_query, destination, k=max_results * 2
            )

            if rag_results:
                # Use the improved method that adds price information
                rag_hotels = self.rag_system.rag_results_to_hotel_info(
                    rag_results, HotelInfo
                )
                logger.info(f"Found {len(rag_hotels)} hotels from RAG")

                # Filter by amenities
                rag_hotels = self.filter_hotels_by_amenities(
                    rag_hotels, required_amenities
                )

                # Filter by budget (now we can since we've added pricing)
                rag_hotels = self.filter_hotels_by_budget(rag_hotels, budget_value)

                all_hotels.extend(rag_hotels)

        # Step 2: If RAG results are insufficient, try Amadeus API
        if use_api and (not all_hotels or len(all_hotels) < min_results):
            logger.info("Searching with Amadeus APIs...")

            # Convert destination to city code using our enhanced converter
            city_code = self.city_converter.get_code(destination)
            logger.info(f"Using city code: {city_code} for {destination}")

            # Get hotel list
            logger.info("Searching hotels with Hotel List API...")
            hotel_list = self.amadeus_client.search_hotel_list(city_code)

            if hotel_list:
                logger.info(f"Found {len(hotel_list)} hotels from Hotel List API")

                # Get hotel IDs for offers search
                hotel_ids = [h.get("hotelId") for h in hotel_list if "hotelId" in h]

                # Get hotel offers
                logger.info("Searching hotel offers...")
                offers = self.amadeus_client.search_hotel_offers(
                    city_code=city_code,
                    check_in_date=check_in_date,
                    check_out_date=check_out_date,
                    adults=adults,
                    hotel_ids=hotel_ids[:20],  # API limit
                )

                if offers:
                    logger.info(f"Found {len(offers)} hotel offers")

                    # Convert API results to HotelInfo
                    api_hotels = self.convert_api_results_to_hotel_info(
                        hotel_list, offers
                    )

                    # Filter by amenities and budget
                    api_hotels = self.filter_hotels_by_amenities(
                        api_hotels, required_amenities
                    )
                    api_hotels = self.filter_hotels_by_budget(api_hotels, budget_value)

                    all_hotels.extend(api_hotels)

        # Deduplicate hotels by ID
        unique_hotels = {}
        for hotel in all_hotels:
            if hotel.id not in unique_hotels:
                unique_hotels[hotel.id] = hotel

        # Sort and limit results
        sorted_hotels = list(unique_hotels.values())
        sorted_hotels.sort(
            key=lambda h: getattr(h, "amenity_match_ratio", 0), reverse=True
        )

        # Return limited number of results
        return sorted_hotels[:max_results]

    def execute(
        self,
        destination: str,
        date_range: str,
        preferences: str,
        budget: str = "$300",
        travelers: str = "2 adults",
    ) -> Dict[str, Any]:
        """
        Execute hotel search and return formatted results

        Args:
            destination: City or destination
            date_range: Date range (e.g., "2025-06-01 to 2025-06-05")
            preferences: Hotel preferences (e.g., "luxury hotel with spa")
            budget: Budget per night (e.g., "$300")
            travelers: Number of travelers (e.g., "2 adults")

        Returns:
            Dictionary with search results
        """
        try:
            # Search for hotels
            hotels = self.search_hotels(
                destination=destination,
                date_range=date_range,
                preferences=preferences,
                budget=budget,
                travelers=travelers,
            )

            if not hotels:
                return {
                    "success": False,
                    "message": "No hotels found matching your criteria.",
                    "hotels": [],
                }

            # Format results
            hotel_results = [hotel.to_dict() for hotel in hotels]

            # Parse dates for response
            check_in_date, check_out_date = self.parse_date_range(date_range)

            return {
                "success": True,
                "message": f"Found {len(hotels)} hotels matching your criteria.",
                "search_criteria": {
                    "destination": destination,
                    "check_in_date": check_in_date,
                    "check_out_date": check_out_date,
                    "preferences": preferences,
                    "budget": budget,
                    "travelers": travelers,
                },
                "hotels": hotel_results,
            }
        except Exception as e:
            logger.error(f"Error executing hotel search: {str(e)}")

            return {
                "success": False,
                "message": f"Error searching for hotels: {str(e)}",
                "hotels": [],
            }
