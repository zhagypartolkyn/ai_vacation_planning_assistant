from IPython.display import Markdown
import os
import pandas as pd
from crewai.tools import BaseTool
import requests
import logging

from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from pydantic import BaseModel, Field
from typing import List, Optional, Any

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Updated path to the RAG vector store
vectorstore_base = r"C:\LabGit\Capstone\restaurant-chunks"
logger.info(f"Loading restaurant vector store from: {vectorstore_base}")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize vector stores
all_vectorstores = []

try:
    # Check if directory exists
    if os.path.exists(vectorstore_base) and os.path.isdir(vectorstore_base):
        # Get all subdirectories
        chunk_dirs = sorted(
            [
                os.path.join(vectorstore_base, d)
                for d in os.listdir(vectorstore_base)
                if os.path.isdir(os.path.join(vectorstore_base, d))
            ]
        )

        # Load each vector store
        for i, path in enumerate(chunk_dirs):
            try:
                db = Chroma(persist_directory=path, embedding_function=embedding_model)
                all_vectorstores.append(db)
                logger.info(f"✅ Loaded vectorstore: {path}")
            except Exception as e:
                logger.error(f"❌ Failed to load vectorstore at {path}: {str(e)}")
    else:
        logger.warning(f"Vector store directory not found: {vectorstore_base}")

except Exception as e:
    logger.error(f"Error initializing vector stores: {str(e)}")


# Define MultiRetriever class
class MultiRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def invoke(self, query):
        # Initialize empty docs list
        docs = []

        # If no retrievers, return empty list
        if not self.retrievers:
            logger.warning("No retrievers available")
            return docs

        # Aggregate results from all retrievers
        for i, retriever in enumerate(self.retrievers):
            try:
                result = retriever.invoke(query)
                logger.info(f"📦 Chunk {i}: {len(result)} docs")
                docs.extend(result)
            except Exception as e:
                logger.error(f"Error from retriever {i}: {str(e)}")
        return docs


# Initialize the retriever if we have vector stores
retriever = None
if all_vectorstores:
    retriever = MultiRetriever([db.as_retriever() for db in all_vectorstores])
    logger.info(f"MultiRetriever initialized with {len(all_vectorstores)} retrievers")
else:
    logger.warning("No vector stores available, MultiRetriever will not be initialized")


# Define RAG tool
class SearchRestaurantsRagTool(BaseTool):
    name: str = "Search restaurant with RAG"
    description: str = "Extracts data with RAG to send restaurant recommendations"
    retriever: Any = Field(...)

    def _run(self, query: Any) -> str:
        if isinstance(query, dict):
            query = query.get("query") or query.get("description") or str(query)

        try:
            logger.info(f"Querying restaurant RAG with: {query}")
            results = self.retriever.invoke(query)

            if not results:
                logger.warning("No relevant restaurants found in RAG")
                return "😕 Sorry! No relevant restaurants found."

            top_results = results[:10]
            response = "🍽️ Top restaurants based on your query:\n\n"

            for idx, result in enumerate(top_results, 1):
                metadata = result.metadata
                dietary_options = []
                if metadata.get("vegetarian_friendly"):
                    dietary_options.append("🥗 Vegetarian Friendly")
                if metadata.get("vegan_options"):
                    dietary_options.append("🌱 Vegan Options")
                if metadata.get("gluten_free"):
                    dietary_options.append("🌾 Gluten Free")

                dietary_str = (
                    " | ".join(dietary_options)
                    if dietary_options
                    else "❌ No special dietary options"
                )

                response += (
                    f"{idx}. 🍝 *{metadata.get('restaurant_name', 'Unknown')}* \n\n "
                    f"   🍜 Cuisine: {metadata.get('cuisine', 'Not specified')}\n\n"
                    f"   ⭐ Rating: {metadata.get('rating', 'N/A')} | 💰 Price: {metadata.get('price_category', 'Unknown')}\n\n"
                    f"   🥕 Dietary: {dietary_str}\n\n"
                    f"   📍 Address: {metadata.get('address', 'Not provided')}\n"
                )

            logger.info(f"Found {len(top_results)} restaurants from RAG")
            return response
        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}")
            return f"Error searching for restaurants: {str(e)}"


# Define Google Places API tool as fallback
class SearchGooglePlacesApiTool(BaseTool):
    name: str = "Search Google Places API"
    description: str = "Uses Google Places API to find restaurants if local data fails."

    def _run(self, query: str) -> str:
        try:
            logger.info(f"Searching Google Places API with: {query}")
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {
                "query": query,
                "type": "restaurant",
                "key": GOOGLE_PLACES_API_KEY,
            }

            response = requests.get(url, params=params)
            data = response.json()

            if "results" not in data or not data["results"]:
                logger.warning("No restaurants found via Google Places API")
                return "😔 No restaurants found via Google Places."

            top_results = data["results"][:10]
            output = "🌍 Top restaurants found for you:\n\n"

            for i, result in enumerate(top_results, 1):
                name = result.get("name", "Unknown")
                address = result.get("formatted_address", "No address provided")
                rating = result.get("rating", "N/A")
                price_level = result.get("price_level", 0)
                price_str = (
                    "💰" * price_level if isinstance(price_level, int) else "N/A"
                )

                output += (
                    f"{i}. 🍝 *{name}*\n\n"
                    f"   📍 Address: {address}\n\n"
                    f"   ⭐ Rating: {rating}\n\n"
                    f"   💸 Price category: {price_str}\n\n"
                )

            logger.info(f"Found {len(top_results)} restaurants from Google Places API")
            return output
        except Exception as e:
            logger.error(f"Error calling Google API: {str(e)}")
            return f"🚨 Error calling Google API: {str(e)}"


# Initialize tools
rag_tool = None
if retriever:
    rag_tool = SearchRestaurantsRagTool(retriever=retriever)
    logger.info("RAG tool initialized")
else:
    logger.warning("RAG tool not initialized due to missing retriever")

google_tool = SearchGooglePlacesApiTool()
logger.info("Google Places API tool initialized")


def search_restaurants(destination: str, prefs: str) -> str:
    """
    Search for restaurants based on destination and preferences.
    Uses RAG first, then falls back to Google Places API if needed.

    Args:
        destination: The destination city
        prefs: Food preferences or cuisine types

    Returns:
        Formatted string with restaurant recommendations
    """
    query = f"{prefs} restaurants in {destination}"
    logger.info(f"Searching restaurants with query: {query}")

    # Try RAG first if available
    if rag_tool:
        try:
            logger.info("Attempting RAG search")
            rag_result = rag_tool._run(query)

            # If RAG found results, return them
            if (
                rag_result
                and "Sorry! No relevant restaurants found" not in rag_result
                and "Error" not in rag_result
            ):
                logger.info("Using results from RAG")
                return rag_result
            else:
                logger.info("No results from RAG, falling back to Google Places API")
        except Exception as e:
            logger.error(
                f"Error in RAG search, falling back to Google Places API: {str(e)}"
            )
    else:
        logger.info("RAG tool not available, using Google Places API directly")

    # Fall back to Google Places API
    try:
        google_result = google_tool._run(query)
        return google_result
    except Exception as e:
        logger.error(f"Error in Google Places API search: {str(e)}")
        return f"Error searching for restaurants: {str(e)}"
