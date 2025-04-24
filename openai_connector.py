"""
OpenAI Connector module for the AI Travel Assistant.
Provides AI-powered lookups for travel codes when traditional methods fail.
"""

import os
import openai
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIConnector:
    """Connector for OpenAI API calls"""

    def __init__(self, api_key=None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        else:
            logger.warning("No OpenAI API key provided. AI fallbacks will not work.")

    def get_airport_code(self, city_name: str) -> Optional[str]:
        """
        Get IATA airport code using OpenAI

        Args:
            city_name: Name of the city/airport to look up

        Returns:
            3-letter IATA code if found, None otherwise
        """
        if not self.api_key:
            logger.warning("OpenAI API key not set, cannot get airport code")
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

            response = openai.ChatCompletion.create(
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
        if not self.api_key:
            logger.warning("OpenAI API key not set, cannot get city name")
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

            response = openai.ChatCompletion.create(
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
        if not self.api_key:
            logger.warning("OpenAI API key not set, cannot translate text")
            return text

        try:
            logger.info(f"Translating text to {target_language}")

            prompt = f"""
            Translate the following text to {target_language}:
            
            {text}
            
            Provide only the translated text without explanations.
            """

            response = openai.ChatCompletion.create(
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

    # Test airport code lookup
    test_cities = [
        "Paris, France",
        "New York, USA",
        "Tokyo, Japan",
        "Sydney, Australia",
    ]

    for city in test_cities:
        code = connector.get_airport_code(city)
        print(f"Airport code for {city}: {code}")
