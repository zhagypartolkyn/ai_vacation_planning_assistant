"""
Amadeus Client module for the AI Travel Assistant.
Handles authentication and API calls to the Amadeus travel APIs.
"""

import os
import requests
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AmadeusClient:
    """Client for Amadeus Travel APIs"""

    def __init__(self, client_id=None, client_secret=None):
        """
        Initialize Amadeus API client

        Args:
            client_id: Amadeus API key (optional, can use env var)
            client_secret: Amadeus API secret (optional, can use env var)
        """
        self.api_key = client_id or os.getenv("AMADEUS_API_PROD_KEY")
        self.api_secret = client_secret or os.getenv("AMADEUS_SECRET_PROD_KEY")

        if not self.api_key or not self.api_secret:
            logger.error("Amadeus API credentials not provided")

        self.token = None
        self.token_expiry = None
        self.base_url = "https://api.amadeus.com/v1"
        self.flight_base_url = "https://api.amadeus.com/v2"

    def get_auth_token(self) -> str:
        """
        Get OAuth token for Amadeus API

        Returns:
            OAuth access token

        Raises:
            Exception: If authentication fails
        """
        # Check if we have a valid token
        if self.token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.token

        # Get new token
        url = f"{self.base_url}/security/oauth2/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.api_secret,
        }

        try:
            logger.info("Authenticating with Amadeus API...")
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            result = response.json()

            self.token = result.get("access_token")
            # Set expiry slightly before actual expiry to be safe
            expires_in = result.get("expires_in", 1800)  # Default to 30 min
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)

            logger.info("Authentication successful.")
            return self.token
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting auth token: {str(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response: {e.response.text}")
            raise Exception(f"Amadeus authentication failed: {str(e)}")

    def get_airport_code(self, city_name: str) -> str:
        """
        Get airport/city code using Amadeus Airport & City Search API

        Args:
            city_name: City name to search for

        Returns:
            IATA code for the city/airport
        """
        token = self.get_auth_token()
        url = f"{self.base_url}/reference-data/locations"

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        params = {
            "subType": "CITY,AIRPORT",
            "keyword": city_name,
            "page[limit]": 10,
            "sort": "analytics.travelers.score,desc",
        }

        try:
            logger.info(f"Looking up code for {city_name}...")
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                results = response.json().get("data", [])
                if results:
                    # Return the highest ranked result's code
                    code = results[0].get("iataCode")
                    logger.info(f"Found code {code} for {city_name}")
                    return code
                else:
                    logger.warning(f"No code found for {city_name}")
                    # Fallback: return first 3 letters of city name as uppercase
                    return city_name[:3].upper()
            else:
                logger.error(f"Error looking up code: {response.status_code}")
                # Fallback: return first 3 letters of city name as uppercase
                return city_name[:3].upper()
        except Exception as e:
            logger.error(f"Exception while looking up code: {str(e)}")
            # Fallback: return first 3 letters of city name as uppercase
            return city_name[:3].upper()

    def get_city_code(self, city_name: str) -> str:
        """
        Get city code using Amadeus API

        Args:
            city_name: City name to search for

        Returns:
            IATA code for the city
        """
        # For now, we'll use the same implementation as airport codes
        # In a production system, you might want to filter for CITY type only
        return self.get_airport_code(city_name)

    def search_flight_offers(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        travel_class: str = "ECONOMY",
        currency: str = "USD",
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search flight offers using Amadeus Flight Offers Search API

        Args:
            origin: Origin city or airport code (e.g., "PAR", "NYC")
            destination: Destination city or airport code (e.g., "LON", "SYD")
            departure_date: Departure date in YYYY-MM-DD format
            return_date: Optional return date for round trips in YYYY-MM-DD format
            adults: Number of adults
            children: Number of children
            infants: Number of infants
            travel_class: Travel class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
            currency: Currency code for prices
            max_results: Maximum number of results to return

        Returns:
            List of flight offers
        """
        token = self.get_auth_token()
        url = f"{self.flight_base_url}/shopping/flight-offers"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Build request body
        request_body = {
            "currencyCode": currency,
            "originDestinations": [
                {
                    "id": "1",
                    "originLocationCode": origin,
                    "destinationLocationCode": destination,
                    "departureDateTimeRange": {"date": departure_date},
                }
            ],
            "travelers": [],
            "sources": ["GDS"],
            "searchCriteria": {
                "maxFlightOffers": max_results,
                "flightFilters": {
                    "cabinRestrictions": [
                        {
                            "cabin": travel_class,
                            "coverage": "MOST_SEGMENTS",
                            "originDestinationIds": ["1"],
                        }
                    ]
                },
            },
        }

        # Add return flight if provided
        if return_date:
            request_body["originDestinations"].append(
                {
                    "id": "2",
                    "originLocationCode": destination,
                    "destinationLocationCode": origin,
                    "departureDateTimeRange": {"date": return_date},
                }
            )
            # Update cabin restrictions for return flight
            request_body["searchCriteria"]["flightFilters"]["cabinRestrictions"][0][
                "originDestinationIds"
            ].append("2")

        # Add travelers
        traveler_id = 1
        # Add adults
        for i in range(adults):
            request_body["travelers"].append(
                {"id": str(traveler_id), "travelerType": "ADULT"}
            )
            traveler_id += 1

        # Add children if any
        for i in range(children):
            request_body["travelers"].append(
                {"id": str(traveler_id), "travelerType": "CHILD"}
            )
            traveler_id += 1

        # Add infants if any
        for i in range(infants):
            request_body["travelers"].append(
                {"id": str(traveler_id), "travelerType": "INFANT"}
            )
            traveler_id += 1

        try:
            logger.info(
                f"Searching flights from {origin} to {destination} on {departure_date}..."
            )
            response = requests.post(url, headers=headers, json=request_body)

            logger.info(f"API Response: {response.status_code}")
            response.raise_for_status()

            result = response.json()
            logger.info(f"Found {len(result.get('data', []))} flight offers")

            return result.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching flight offers: {str(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response: {e.response.text}")
            return []

    def search_hotel_list(self, city_code: str) -> List[Dict[str, Any]]:
        """
        Search hotels using Amadeus Hotel List API

        Args:
            city_code: IATA city code

        Returns:
            List of hotels in the specified city
        """
        token = self.get_auth_token()
        url = f"{self.base_url}/reference-data/locations/hotels/by-city"

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        params = {
            "cityCode": city_code,
            "radius": 20,  # 20 km radius
            "radiusUnit": "KM",
            "hotelSource": "ALL",
        }

        try:
            logger.info(f"Searching hotels in {city_code}...")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Found {len(result.get('data', []))} hotels in {city_code}")

            return result.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching hotels with Hotel List API: {str(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response: {e.response.text}")
            return []

    def search_hotel_offers(
        self,
        city_code: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 1,
        hotel_ids: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search hotel offers using Amadeus Hotel Search API

        Args:
            city_code: IATA city code
            check_in_date: Check-in date in YYYY-MM-DD format
            check_out_date: Check-out date in YYYY-MM-DD format
            adults: Number of adults
            hotel_ids: Optional list of specific hotel IDs to search

        Returns:
            List of hotel offers with pricing and availability
        """
        token = self.get_auth_token()
        # Note the v3 endpoint for shopping/hotel-offers
        url = "https://api.amadeus.com/v3/shopping/hotel-offers"

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        params = {
            "cityCode": city_code,
            "checkInDate": check_in_date,
            "checkOutDate": check_out_date,
            "adults": adults,
            "roomQuantity": 1,
            "bestRateOnly": "true",
            "currency": "USD",
        }

        # Add hotel IDs if provided
        if hotel_ids:
            params["hotelIds"] = ",".join(hotel_ids[:20])  # API limit

        try:
            logger.info(
                f"Searching hotel offers in {city_code} for {check_in_date} to {check_out_date}..."
            )
            response = requests.get(url, headers=headers, params=params)

            logger.info(f"API Response: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Error response: {response.text[:500]}...")

            response.raise_for_status()

            result = response.json()
            logger.info(f"Found {len(result.get('data', []))} hotel offers")

            return result.get("data", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching hotel offers: {str(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response: {e.response.text[:500]}...")
            return []

    def get_hotel_offers_by_id(
        self, hotel_id: str, check_in_date: str, check_out_date: str, adults: int = 1
    ) -> Dict[str, Any]:
        """
        Get offers for a specific hotel

        Args:
            hotel_id: Hotel ID
            check_in_date: Check-in date in YYYY-MM-DD format
            check_out_date: Check-out date in YYYY-MM-DD format
            adults: Number of adults

        Returns:
            Hotel offer details with pricing and availability
        """
        token = self.get_auth_token()
        url = f"{self.base_url}/shopping/hotel-offers/by-hotel"

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        params = {
            "hotelId": hotel_id,
            "checkInDate": check_in_date,
            "checkOutDate": check_out_date,
            "adults": adults,
            "roomQuantity": 1,
            "currency": "USD",
        }

        try:
            logger.info(f"Getting offers for hotel {hotel_id}...")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting hotel offers for hotel {hotel_id}: {str(e)}")
            if hasattr(e, "response") and e.response:
                logger.error(f"Response: {e.response.text}")
            return {}

    def handle_retries(
        self, func, *args, max_retries: int = 3, initial_delay: float = 1.0, **kwargs
    ):
        """
        Helper function to handle retries with exponential backoff

        Args:
            func: Function to call
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function call

        Raises:
            Exception: If all retries fail
        """
        retries = 0
        delay = initial_delay

        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                retries += 1

                if retries >= max_retries:
                    logger.error(f"Max retries reached, giving up: {str(e)}")
                    raise

                # Check if it's a rate limiting error (429)
                if (
                    hasattr(e, "response")
                    and e.response
                    and e.response.status_code == 429
                ):
                    logger.warning(f"Rate limit hit, retrying in {delay} seconds")
                else:
                    logger.warning(
                        f"Request failed, retrying in {delay} seconds: {str(e)}"
                    )

                # Sleep with exponential backoff
                time.sleep(delay)
                delay *= 2  # Double the delay for next retry

    def get_city_name_from_code(self, code: str) -> str:
        """
        Get city name from IATA code using Amadeus API

        Args:
            code: IATA code to lookup

        Returns:
            City name if found, the code itself otherwise
        """
        token = self.get_auth_token()
        url = f"{self.base_url}/reference-data/locations/{code}"

        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        try:
            logger.info(f"Looking up city name for code {code}...")
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                result = response.json()

                # Extract the city or airport name
                if "data" in result:
                    location_data = result.get("data", {})

                    # First try to get the city name
                    city_name = location_data.get("cityName")
                    if city_name:
                        logger.info(f"Found city name {city_name} for {code}")
                        return city_name

                    # Otherwise use the location name
                    name = location_data.get("name")
                    if name:
                        logger.info(f"Found location name {name} for {code}")
                        return name

                logger.warning(f"No name found for {code} in Amadeus API response")
            else:
                logger.error(f"Error looking up code: {response.status_code}")
                # Log the error response for debugging
                if hasattr(response, "text"):
                    logger.error(f"Response: {response.text[:500]}")
        except Exception as e:
            logger.error(f"Exception while looking up code: {str(e)}")

        # Return the code as a fallback
        return code


if __name__ == "__main__":
    # Test the Amadeus Client
    client = AmadeusClient()

    # Test authentication
    try:
        token = client.get_auth_token()
        print(f"Authentication successful, token: {token[:10]}...")
    except Exception as e:
        print(f"Authentication failed: {e}")
        exit(1)

    # Test airport code lookup
    test_cities = ["Paris", "New York", "Tokyo", "Sydney"]

    print("\nAIRPORT CODES:")
    for city in test_cities:
        code = client.get_airport_code(city)
        print(f"Airport code for {city}: {code}")

    # Test flight search
    print("\nFLIGHT SEARCH:")
    origin = "NYC"
    destination = "LON"
    departure_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    return_date = (datetime.now() + timedelta(days=37)).strftime("%Y-%m-%d")

    print(f"Searching flights from {origin} to {destination} on {departure_date}...")
    flights = client.search_flight_offers(
        origin=origin,
        destination=destination,
        departure_date=departure_date,
        return_date=return_date,
        max_results=5,
    )

    print(f"Found {len(flights)} flights")

    # Test hotel search
    print("\nHOTEL SEARCH:")
    city_code = "LON"
    check_in = departure_date
    check_out = return_date

    print(f"Searching hotels in {city_code} from {check_in} to {check_out}...")
    hotels = client.search_hotel_list(city_code=city_code)

    print(f"Found {len(hotels)} hotels")

    if hotels:
        # Get first 5 hotel IDs
        hotel_ids = [h.get("hotelId") for h in hotels[:5] if "hotelId" in h]

        if hotel_ids:
            print(f"Searching offers for {len(hotel_ids)} hotels...")
            offers = client.search_hotel_offers(
                city_code=city_code,
                check_in_date=check_in,
                check_out_date=check_out,
                hotel_ids=hotel_ids,
            )

            print(f"Found {len(offers)} hotel offers")
