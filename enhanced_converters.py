"""
Enhanced Converters module for the AI Travel Assistant.
Provides robust conversion of city names to IATA codes with multiple fallback mechanisms.
"""

import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAirportCodeConverter:
    """Enhanced converter for airport codes with multiple fallbacks"""

    def __init__(self, amadeus_client=None, openai_connector=None):
        """
        Initialize with optional clients for lookups

        Args:
            amadeus_client: AmadeusClient instance for API lookups
            openai_connector: OpenAIConnector instance for AI-powered lookups
        """
        self.amadeus_client = amadeus_client
        self.openai_connector = openai_connector

        # Existing airport codes dictionary - comprehensive list of major airports
        self.AIRPORT_CODES = {
            "paris": "PAR",
            "london": "LON",
            "new york": "NYC",
            "tokyo": "TYO",
            "rome": "ROM",
            "barcelona": "BCN",
            "amsterdam": "AMS",
            "berlin": "BER",
            "madrid": "MAD",
            "singapore": "SIN",
            "dubai": "DXB",
            "hong kong": "HKG",
            "bangkok": "BKK",
            "sydney": "SYD",
            "los angeles": "LAX",
            "san francisco": "SFO",
            "chicago": "ORD",
            "miami": "MIA",
            "toronto": "YTO",
            "montreal": "YMQ",
            "delhi": "DEL",
            "mumbai": "BOM",
            "shanghai": "SHA",
            "beijing": "BJS",
            "seoul": "SEL",
            "athens": "ATH",
            "cairo": "CAI",
            "johannesburg": "JNB",
            "rio de janeiro": "RIO",
            "sao paulo": "SAO",
            "mexico city": "MEX",
            "vancouver": "YVR",
            "boston": "BOS",
            "washington": "WAS",
            "dallas": "DFW",
            "houston": "HOU",
            "atlanta": "ATL",
            "denver": "DEN",
            "seattle": "SEA",
            "las vegas": "LAS",
            "phoenix": "PHX",
            "austin": "AUS",
            "philadelphia": "PHL",
            "san diego": "SAN",
            "portland": "PDX",
            "new orleans": "MSY",
            "orlando": "MCO",
            "tampa": "TPA",
            "nashville": "BNA",
            "sydney": "SYD",
            "melbourne": "MEL",
            "brisbane": "BNE",
            "perth": "PER",
            "auckland": "AKL",
            "wellington": "WLG",
            "christchurch": "CHC",
            "honolulu": "HNL",
            "vienna": "VIE",
            "brussels": "BRU",
            "prague": "PRG",
            "copenhagen": "CPH",
            "helsinki": "HEL",
            "nice": "NCE",
            "frankfurt": "FRA",
            "munich": "MUC",
            "hamburg": "HAM",
            "athens": "ATH",
            "budapest": "BUD",
            "dublin": "DUB",
            "milan": "MIL",
            "venice": "VCE",
            "florence": "FLR",
            "naples": "NAP",
            "oslo": "OSL",
            "warsaw": "WAW",
            "lisbon": "LIS",
            "porto": "OPO",
            "bucharest": "BUH",
            "stockholm": "STO",
            "geneva": "GVA",
            "zurich": "ZRH",
            "istanbul": "IST",
        }

    def get_code(self, city: str) -> str:
        """
        Get airport code with fallback chain:
        1. Static dictionary lookup
        2. Amadeus API lookup (if client available)
        3. OpenAI API lookup (if connector available)
        4. Fallback to first 3 letters

        Args:
            city: City or airport name

        Returns:
            3-letter IATA code
        """
        if not city:
            logger.warning("Empty city name provided")
            return "XXX"  # Return placeholder for empty input

        # Normalize city name
        city_normalized = city.strip().lower()

        # Log the lookup attempt
        logger.info(f"Looking up airport code for: '{city}'")

        # 1. Direct lookup from static dictionary
        if city_normalized in self.AIRPORT_CODES:
            code = self.AIRPORT_CODES[city_normalized]
            logger.info(f"Found code '{code}' for '{city}' in static dictionary")
            return code

        # 2. Try partial match
        for known_city, code in self.AIRPORT_CODES.items():
            if city_normalized in known_city or known_city in city_normalized:
                logger.info(
                    f"Found partial match '{code}' for '{city}' in static dictionary"
                )
                return code

        # 3. Try Amadeus lookup if client is available
        if self.amadeus_client:
            try:
                logger.info(f"Trying Amadeus API lookup for '{city}'")
                code = self.amadeus_client.get_airport_code(city)
                if code and len(code) == 3 and code.isalpha():
                    logger.info(f"Found code '{code}' for '{city}' via Amadeus API")
                    return code
            except Exception as e:
                logger.warning(f"Amadeus lookup failed for '{city}': {e}")

        # 4. Try OpenAI lookup if connector is available
        if self.openai_connector:
            try:
                logger.info(f"Trying OpenAI lookup for '{city}'")
                code = self.openai_connector.get_airport_code(city)
                if code:
                    logger.info(f"Found code '{code}' for '{city}' via OpenAI")
                    return code
            except Exception as e:
                logger.warning(f"OpenAI lookup failed for '{city}': {e}")

        # 5. Fallback to first 3 letters uppercase
        fallback_code = city_normalized[:3].upper()
        if len(fallback_code) < 3:
            # Pad with X if too short
            fallback_code = fallback_code.ljust(3, "X")

        logger.warning(f"No code found for '{city}', using fallback: '{fallback_code}'")
        return fallback_code

    def is_valid_code(self, code: str) -> bool:
        """
        Check if a code is a valid IATA airport code format

        Args:
            code: Code to check

        Returns:
            True if valid format, False otherwise
        """
        return code and len(code) == 3 and code.isalpha() and code.isupper()

    def get_city_name(self, code: str) -> str:
        """
        Get city name from IATA airport code with fallback chain:
        1. Static dictionary lookup
        2. Amadeus API lookup (if client available)
        3. OpenAI API lookup (if connector available)
        4. Return the code itself as fallback

        Args:
            code: 3-letter IATA code

        Returns:
            City or airport name
        """
        if not code or len(code) != 3:
            logger.warning(f"Invalid airport code provided: {code}")
            return "Unknown"

        # Normalize code
        code_normalized = code.strip().upper()

        # Log the lookup attempt
        logger.info(f"Looking up city name for airport code: '{code_normalized}'")

        # 1. Create reverse mapping from AIRPORT_CODES dictionary
        reverse_mapping = {v: k for k, v in self.AIRPORT_CODES.items()}

        # 2. Direct lookup from reverse mapping
        if code_normalized in reverse_mapping:
            city = reverse_mapping[code_normalized]
            logger.info(
                f"Found city '{city}' for code '{code_normalized}' in static dictionary"
            )
            return city.title()  # Convert to title case

        # 3. Try Amadeus lookup if client is available
        if self.amadeus_client:
            try:
                logger.info(f"Trying Amadeus API lookup for code '{code_normalized}'")
                # Note: Amadeus doesn't have a direct endpoint for this
                # This is a placeholder - you would need to implement this
                # using an appropriate Amadeus endpoint
                city = self.amadeus_client.get_city_name_from_code(code_normalized)
                if city:
                    logger.info(
                        f"Found city '{city}' for code '{code_normalized}' via Amadeus API"
                    )
                    return city
            except Exception as e:
                logger.warning(
                    f"Amadeus lookup failed for code '{code_normalized}': {e}"
                )

        # 4. Try OpenAI lookup if connector is available
        if self.openai_connector:
            try:
                logger.info(f"Trying OpenAI lookup for code '{code_normalized}'")
                city = self.openai_connector.get_city_name_from_code(code_normalized)
                if city:
                    logger.info(
                        f"Found city '{city}' for code '{code_normalized}' via OpenAI"
                    )
                    return city
            except Exception as e:
                logger.warning(
                    f"OpenAI lookup failed for code '{code_normalized}': {e}"
                )

        # 5. Return the code itself as fallback
        logger.warning(
            f"No city found for code '{code_normalized}', returning code as is"
        )
        return code_normalized


class EnhancedCityCodeConverter:
    """Enhanced converter for city codes with multiple fallbacks"""

    def __init__(self, amadeus_client=None, openai_connector=None):
        """
        Initialize with optional clients for lookups

        Args:
            amadeus_client: AmadeusClient instance for API lookups
            openai_connector: OpenAIConnector instance for AI-powered lookups
        """
        self.amadeus_client = amadeus_client
        self.openai_connector = openai_connector

        # City codes dictionary - comprehensive list of major cities
        self.CITY_CODES = {
            "paris": "PAR",
            "london": "LON",
            "new york": "NYC",
            "tokyo": "TYO",
            "rome": "ROM",
            "barcelona": "BCN",
            "amsterdam": "AMS",
            "berlin": "BER",
            "madrid": "MAD",
            "singapore": "SIN",
            "dubai": "DXB",
            "hong kong": "HKG",
            "bangkok": "BKK",
            "sydney": "SYD",
            "los angeles": "LAX",
            "san francisco": "SFO",
            "chicago": "CHI",
            "miami": "MIA",
            "toronto": "YTO",
            "montreal": "YMQ",
            "delhi": "DEL",
            "mumbai": "BOM",
            "shanghai": "SHA",
            "beijing": "BJS",
            "seoul": "SEL",
            "athens": "ATH",
            "cairo": "CAI",
            "johannesburg": "JNB",
            "rio de janeiro": "RIO",
            "sao paulo": "SAO",
            "mexico city": "MEX",
            "vancouver": "YVR",
            "boston": "BOS",
            "washington": "WAS",
            "dallas": "DFW",
            "houston": "HOU",
            "atlanta": "ATL",
            "denver": "DEN",
            "seattle": "SEA",
            "las vegas": "LAS",
            "phoenix": "PHX",
            "austin": "AUS",
            "philadelphia": "PHL",
            "san diego": "SAN",
            "portland": "PDX",
            "new orleans": "MSY",
            "orlando": "MCO",
            "tampa": "TPA",
            "nashville": "BNA",
            "sydney": "SYD",
            "melbourne": "MEL",
            "brisbane": "BNE",
            "perth": "PER",
            "auckland": "AKL",
            "wellington": "WLG",
            "christchurch": "CHC",
            "honolulu": "HNL",
            "vienna": "VIE",
            "brussels": "BRU",
            "prague": "PRG",
            "copenhagen": "CPH",
            "helsinki": "HEL",
            "nice": "NCE",
            "frankfurt": "FRA",
            "munich": "MUC",
            "hamburg": "HAM",
            "athens": "ATH",
            "budapest": "BUD",
            "dublin": "DUB",
            "milan": "MIL",
            "venice": "VCE",
            "florence": "FLR",
            "naples": "NAP",
            "oslo": "OSL",
            "warsaw": "WAW",
            "lisbon": "LIS",
            "porto": "OPO",
            "bucharest": "BUH",
            "stockholm": "STO",
            "geneva": "GVA",
            "zurich": "ZRH",
            "istanbul": "IST",
        }

    def get_code(self, city: str) -> str:
        """
        Get city code with fallback chain:
        1. Static dictionary lookup
        2. Amadeus API lookup (if client available)
        3. OpenAI API lookup (if connector available)
        4. Fallback to first 3 letters

        Args:
            city: City name

        Returns:
            3-letter city code
        """
        if not city:
            logger.warning("Empty city name provided")
            return "XXX"  # Return placeholder for empty input

        # Normalize city name
        city_normalized = city.strip().lower()

        # Log the lookup attempt
        logger.info(f"Looking up city code for: '{city}'")

        # 1. Direct lookup from static dictionary
        if city_normalized in self.CITY_CODES:
            code = self.CITY_CODES[city_normalized]
            logger.info(f"Found code '{code}' for '{city}' in static dictionary")
            return code

        # 2. Try partial match
        for known_city, code in self.CITY_CODES.items():
            if city_normalized in known_city or known_city in city_normalized:
                logger.info(
                    f"Found partial match '{code}' for '{city}' in static dictionary"
                )
                return code

        # 3. Try Amadeus lookup if client is available
        if self.amadeus_client:
            try:
                logger.info(f"Trying Amadeus API lookup for '{city}'")
                code = self.amadeus_client.get_city_code(city)
                if code and len(code) == 3 and code.isalpha():
                    logger.info(f"Found code '{code}' for '{city}' via Amadeus API")
                    return code
            except Exception as e:
                logger.warning(f"Amadeus lookup failed for '{city}': {e}")

        # 4. Try OpenAI lookup if connector is available
        if self.openai_connector:
            try:
                logger.info(f"Trying OpenAI lookup for '{city}'")
                code = self.openai_connector.get_city_code(city)
                if code:
                    logger.info(f"Found code '{code}' for '{city}' via OpenAI")
                    return code
            except Exception as e:
                logger.warning(f"OpenAI lookup failed for '{city}': {e}")

        # 5. Fallback to first 3 letters uppercase
        fallback_code = city_normalized[:3].upper()
        if len(fallback_code) < 3:
            # Pad with X if too short
            fallback_code = fallback_code.ljust(3, "X")

        logger.warning(f"No code found for '{city}', using fallback: '{fallback_code}'")
        return fallback_code

    def is_valid_code(self, code: str) -> bool:
        """
        Check if a code is a valid city code format

        Args:
            code: Code to check

        Returns:
            True if valid format, False otherwise
        """
        return code and len(code) == 3 and code.isalpha() and code.isupper()

    def get_city_name(self, code: str) -> str:
        """
        Get city name from city code with fallback chain:
        1. Static dictionary lookup
        2. Amadeus API lookup (if client available)
        3. OpenAI API lookup (if connector available)
        4. Return the code itself as fallback

        Args:
            code: 3-letter city code

        Returns:
            City name
        """
        if not code or len(code) != 3:
            logger.warning(f"Invalid city code provided: {code}")
            return "Unknown"

        # Normalize code
        code_normalized = code.strip().upper()

        # Log the lookup attempt
        logger.info(f"Looking up city name for city code: '{code_normalized}'")

        # 1. Create reverse mapping from CITY_CODES dictionary
        reverse_mapping = {v: k for k, v in self.CITY_CODES.items()}

        # 2. Direct lookup from reverse mapping
        if code_normalized in reverse_mapping:
            city = reverse_mapping[code_normalized]
            logger.info(
                f"Found city '{city}' for code '{code_normalized}' in static dictionary"
            )
            return city.title()  # Convert to title case

        # 3. Try Amadeus lookup if client is available
        if self.amadeus_client:
            try:
                logger.info(f"Trying Amadeus API lookup for code '{code_normalized}'")
                # Note: Amadeus doesn't have a direct endpoint for this
                # This is a placeholder - you would need to implement this
                # using an appropriate Amadeus endpoint
                city = self.amadeus_client.get_city_name_from_code(code_normalized)
                if city:
                    logger.info(
                        f"Found city '{city}' for code '{code_normalized}' via Amadeus API"
                    )
                    return city
            except Exception as e:
                logger.warning(
                    f"Amadeus lookup failed for code '{code_normalized}': {e}"
                )

        # 4. Try OpenAI lookup if connector is available
        if self.openai_connector:
            try:
                logger.info(f"Trying OpenAI lookup for code '{code_normalized}'")
                city = self.openai_connector.get_city_name_from_code(code_normalized)
                if city:
                    logger.info(
                        f"Found city '{city}' for code '{code_normalized}' via OpenAI"
                    )
                    return city
            except Exception as e:
                logger.warning(
                    f"OpenAI lookup failed for code '{code_normalized}': {e}"
                )

        # 5. Return the code itself as fallback
        logger.warning(
            f"No city found for code '{code_normalized}', returning code as is"
        )
        return code_normalized


if __name__ == "__main__":
    # Test the converters
    from openai_connector import OpenAIConnector

    openai_connector = OpenAIConnector()

    # Test airport code converter
    airport_converter = EnhancedAirportCodeConverter(openai_connector=openai_connector)

    test_cities = [
        "Paris, France",
        "New York, USA",
        "Tokyo, Japan",
        "Sydney, Australia",
        "Unknown City",
    ]

    print("AIRPORT CODES:")
    for city in test_cities:
        code = airport_converter.get_code(city)
        print(f"Airport code for {city}: {code}")

    # Test city code converter
    city_converter = EnhancedCityCodeConverter(openai_connector=openai_connector)

    print("\nCITY CODES:")
    for city in test_cities:
        code = city_converter.get_code(city)
        print(f"City code for {city}: {code}")
