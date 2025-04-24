import os
import torch
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Enhanced RAG system for hotel search with better city matching and budget handling"""

    def __init__(self, vectorstore_path=None):
        """
        Initialize RAG system

        Args:
            vectorstore_path: Path to the FAISS vector store
        """
        # Use provided path or default
        if not vectorstore_path:
            self.vectorstore_path = "./hotel_vectorstore"
        else:
            self.vectorstore_path = vectorstore_path

        self.vector_store = None

    def load_rag_system(self):
        """Load the RAG system"""
        logger.info("Loading RAG system...")

        if not os.path.exists(self.vectorstore_path):
            logger.error(f"Vector store not found at {self.vectorstore_path}")
            return False

        try:
            # Load the embedding model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            use_gpu = True
            device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
            logger.info(f"Using device: {device}")

            embeddings = HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs={"device": device}
            )

            # Load vector store
            self.vector_store = FAISS.load_local(
                self.vectorstore_path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info("RAG system loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading RAG system: {str(e)}")
            return False

    def query_rag(self, query: str, city: str, k: int = 10) -> List[Document]:
        """
        Query the RAG system with improved city filtering

        Args:
            query: User query including preferences
            city: Specific city to filter for
            k: Number of results to retrieve (pre-filtering)

        Returns:
            List of Document objects filtered by city
        """
        if not self.vector_store and not self.load_rag_system():
            logger.error("RAG system not available")
            return []

        try:
            # Extract just the city name without country information
            city_name = city.split(",")[0].strip().lower()
            logger.info(f"Querying RAG system with: '{query}' for city: '{city_name}'")

            # Retrieve more results initially to have enough after filtering
            results = self.vector_store.similarity_search(query, k=k * 2)

            if not results:
                logger.warning("No results from initial RAG query")
                return []

            # Filter results to prioritize the specific city
            city_results = []
            other_results = []

            for doc in results:
                doc_city = doc.metadata.get("city_name", "").lower()

                # Check if the document city matches the requested city
                # Use fuzzy matching to handle slight variations in city names
                if self._is_city_match(doc_city, city_name):
                    city_results.append(doc)
                else:
                    other_results.append(doc)

            logger.info(
                f"Found {len(city_results)} results matching city '{city_name}'"
            )

            # If we have enough city-specific results, use those
            # Otherwise, supplement with other results
            final_results = city_results[:k]
            if len(final_results) < k:
                remaining_slots = k - len(final_results)
                final_results.extend(other_results[:remaining_slots])

            logger.info(f"Returning {len(final_results)} total results")
            return final_results

        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            return []

    def _is_city_match(self, doc_city: str, target_city: str) -> bool:
        """
        Check if the document city matches the target city using fuzzy matching

        Args:
            doc_city: City name from document metadata
            target_city: Target city name from user query

        Returns:
            True if cities match, False otherwise
        """
        # Exact match
        if doc_city == target_city:
            return True

        # Handle common city name variations (spaces, hyphens, etc.)
        doc_city_clean = re.sub(r"[^a-z]", "", doc_city)
        target_city_clean = re.sub(r"[^a-z]", "", target_city)

        if doc_city_clean == target_city_clean:
            return True

        # Check if one contains the other
        if (
            len(doc_city) > 3
            and len(target_city) > 3
            and (doc_city in target_city or target_city in doc_city)
        ):
            return True

        return False

    def _assign_price_based_on_sentiment(self, hotel_info) -> Dict[str, Any]:
        """
        Assign price to RAG results based on sentiment score

        Args:
            hotel_info: Hotel information object

        Returns:
            Price dictionary with appropriate pricing
        """
        # Extract sentiment score from metadata if available
        sentiment_score = hotel_info.metadata.get("description_sentiment_category", 0)
        if not isinstance(sentiment_score, (int, float)):
            try:
                sentiment_score = float(sentiment_score)
            except (ValueError, TypeError):
                sentiment_score = 0

        # Assign base rate based on sentiment score
        if sentiment_score >= 90:
            base_rate = 200.0
        elif 85 <= sentiment_score < 90:
            base_rate = 150.0
        else:
            # Random value between 85-100 for lower sentiment scores
            import random

            base_rate = random.uniform(85.0, 100.0)

        # Create price dictionary
        return {
            "currency": "USD",
            "total": str(base_rate),
            "base": str(base_rate),
            "taxes": [],
            "variations": {},
        }

    def rag_results_to_hotel_info(
        self, results: List[Document], from_class=None
    ) -> List:
        """
        Convert RAG results to HotelInfo objects with pricing information

        Args:
            results: List of Document objects from RAG search
            from_class: Optional HotelInfo class to use for instantiation

        Returns:
            List of HotelInfo objects with price information
        """
        hotels = []

        for doc in results:
            # Extract metadata
            hotel_id = doc.metadata.get("hotel_id", "Unknown")
            hotel_name = doc.metadata.get("hotel_name", "Unknown")

            # Extract content
            content_lines = doc.page_content.split("\n")
            description = ""
            facilities = ""

            for line in content_lines:
                if line.startswith("Description:"):
                    description = line[12:].strip()
                elif line.startswith("Facilities:"):
                    facilities = line[11:].strip()

            # Parse facilities into list
            amenities = [f.strip() for f in facilities.split(",") if f.strip()]

            # Create location dict
            location = {
                "city": doc.metadata.get("city_name", ""),
                "country": doc.metadata.get("country_code", ""),
                "latitude": doc.metadata.get("latitude", None),
                "longitude": doc.metadata.get("longitude", None),
            }

            # Generate price based on sentiment
            price = self._assign_price_based_on_sentiment(doc)

            # Get sentiment category for display
            sentiment_category = doc.metadata.get("description_sentiment_category", "")

            # Create hotel info object
            if from_class:
                hotel = from_class(
                    id=hotel_id,
                    name=hotel_name,
                    description=description,
                    amenities=amenities,
                    location=location,
                    price=price,
                    source="RAG",
                )
                # Add sentiment category as an attribute (this will be preserved in to_dict())
                hotel.description_sentiment_category = sentiment_category
            else:
                # Create a dictionary if no class provided
                hotel = {
                    "id": hotel_id,
                    "name": hotel_name,
                    "description": description,
                    "amenities": amenities,
                    "location": location,
                    "price": price,
                    "source": "RAG",
                    "description_sentiment_category": sentiment_category,
                }

            hotels.append(hotel)

        return hotels
