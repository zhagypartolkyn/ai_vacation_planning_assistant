"""
OpenAI Connector module for the AI Travel Assistant.
Provides AI-powered lookups for travel codes when traditional methods fail.
"""

import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIConnector:
    """Connector for OpenAI API calls"""

    def __init__(self, api_key=None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        if not self.api_key:
            logger.warning("No OpenAI API key provided. AI fallbacks will not work.")
        else:
            logger.info("OpenAI API key available")

            # Import here to avoid import errors if openai is not installed
            try:
                import openai

                # Set up the client with API key
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.error(
                    "OpenAI package not installed. Please install with 'pip install openai'"
                )
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")

    def get_airport_code(self, city_name: str) -> Optional[str]:
        """
        Get IATA airport code using OpenAI

        Args:
            city_name: Name of the city/airport to look up

        Returns:
            3-letter IATA code if found, None otherwise
        """
        if not self.api_key or not self.client:
            logger.warning(
                "OpenAI API key not set or client not initialized, cannot get airport code"
            )
            return None

        try:
            logger.info(f"Looking up airport code for '{city_name}' using OpenAI")

            prompt = f"""
            What is the 3-letter IATA airport code for {city_name}?
            Please respond with ONLY the code in uppercase. If there are multiple airports, 
            provide the code for the main international airport or the city code.
            For major cities that have multiple airports, use the metropolitan area code.
            Examples:
            - For "New York" respond with "NYC" (not JFK, LGA, or EWR)
            - For "London" respond with "LON" (not LHR, LGW, or STN)
            - For "Tokyo" respond with "TYO" (not NRT or HND)
            - For smaller cities with just one airport, provide that airport's code.
            
            Only respond with the 3-letter code and nothing else.
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise assistant that only returns IATA airport codes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0,
            )

            # Extract the code - the response should just be 3 letters
            code = response.choices[0].message.content.strip().upper()

            # Validate code format (3 uppercase letters)
            if len(code) == 3 and code.isalpha() and code.isupper():
                logger.info(
                    f"Successfully obtained airport code '{code}' for '{city_name}'"
                )
                return code

            logger.warning(
                f"OpenAI returned invalid airport code format: '{code}' for '{city_name}'"
            )
            return None

        except Exception as e:
            logger.error(f"Error getting airport code from OpenAI: {str(e)}")
            return None

    def get_city_code(self, city_name: str) -> Optional[str]:
        """
        Get city code using OpenAI

        Args:
            city_name: Name of the city to look up

        Returns:
            3-letter city code if found, None otherwise
        """
        return self.get_airport_code(city_name)  # Uses same implementation for now

    def get_city_name_from_code(self, code: str) -> Optional[str]:
        """
        Get city name from IATA code using OpenAI

        Args:
            code: 3-letter IATA code to look up

        Returns:
            City name if found, None otherwise
        """
        if not self.api_key or not self.client:
            logger.warning(
                "OpenAI API key not set or client not initialized, cannot get city name"
            )
            return None

        try:
            logger.info(f"Looking up city name for code '{code}' using OpenAI")

            prompt = f"""
            What city or airport does the 3-letter IATA code '{code}' represent?
            Please respond with ONLY the full city or airport name and nothing else.
            For airport codes that represent a specific airport, give me the city name.
            For example:
            - For "NYC" respond with "New York City"
            - For "LAX" respond with "Los Angeles"
            - For "LHR" respond with "London"
            
            Only respond with the city name and nothing else.
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise assistant that only returns city names.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=20,
                temperature=0,
            )

            # Extract the city name
            city_name = response.choices[0].message.content.strip()

            logger.info(
                f"Successfully obtained city name '{city_name}' for code '{code}'"
            )
            return city_name

        except Exception as e:
            logger.error(f"Error getting city name from OpenAI: {str(e)}")
            return None

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text using OpenAI

        Args:
            text: Text to translate
            target_language: Target language code (e.g., "es" for Spanish)

        Returns:
            Translated text
        """
        if not self.api_key or not self.client:
            logger.warning(
                "OpenAI API key not set or client not initialized, cannot translate text"
            )
            return text

        try:
            logger.info(f"Translating text to {target_language}")

            prompt = f"""
            Translate the following text to {target_language}:
            
            {text}
            
            Provide only the translated text without explanations.
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise translator."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.2,
            )

            translated_text = response.choices[0].message.content.strip()
            return translated_text

        except Exception as e:
            logger.error(f"Error translating text with OpenAI: {str(e)}")
            return text  # Return original text if translation fails


if __name__ == "__main__":
    # Test the OpenAI connector
    connector = OpenAIConnector()

    if connector.api_key:
        print(f"API key found: {connector.api_key[:4]}{'*' * 20}")
    else:
        print(
            "No API key found. This is expected if you're using the app.py integration."
        )
        print(
            "When running directly, you can set the OPENAI_API_KEY environment variable for testing."
        )

        # For direct testing only - if you want to test without environment variables
        should_test = input(
            "Would you like to input an API key for testing? (yes/no): "
        )
        if should_test.lower() == "yes":
            api_key = input("Enter your OpenAI API key: ")
            if api_key:
                connector = OpenAIConnector(api_key=api_key)
                print("Connector initialized with provided API key.")
            else:
                print("No API key provided. Exiting test mode.")
                exit(1)
        else:
            print("Exiting test mode.")
            exit(1)

    # Test airport code lookup
    test_cities = [
        "Paris, France",
        "New York, USA",
        "Tokyo, Japan",
        "Sydney, Australia",
    ]

    print("\nTesting airport code lookup:")
    for city in test_cities:
        code = connector.get_airport_code(city)
        print(f"Airport code for {city}: {code}")

    # Test reverse lookup
    if connector.api_key and connector.client:
        print("\nTesting city name lookup:")
        test_codes = ["CDG", "JFK", "NRT", "SYD"]
        for code in test_codes:
            city = connector.get_city_name_from_code(code)
            print(f"City for code {code}: {city}")
