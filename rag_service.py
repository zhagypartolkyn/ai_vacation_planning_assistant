"""
Enhanced RAG system for hotel search with better city matching and budget handling
"""

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
        logger.info(f"Loading RAG system from {self.vectorstore_path}...")

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
            results = self.vector_store.similarity_search(
                query, k=k * 3
            )  # Get more results for better filtering

            if not results:
                logger.warning("No results from initial RAG query")
                return []

            # Debug log showing all found cities for troubleshooting
            doc_cities = [doc.metadata.get("city_name", "").lower() for doc in results]
            logger.info(f"Found cities in RAG results: {set(doc_cities)}")

            # Debug log of document content for the first few results
            for i, doc in enumerate(results[:3]):
                logger.info(
                    f"Document {i+1} content sample: {doc.page_content[:100]}..."
                )
                logger.info(f"Document {i+1} metadata: {doc.metadata}")

            # Filter results to prioritize the specific city - IMPROVED matching
            city_results = []
            other_results = []

            for doc in results:
                doc_city = doc.metadata.get("city_name", "").lower()

                # Improved city matching
                if self._is_city_match_strict(doc_city, city_name):
                    city_results.append(doc)
                else:
                    other_results.append(doc)

            logger.info(
                f"Found {len(city_results)} results strictly matching city '{city_name}'"
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

    def _is_city_match_strict(self, doc_city: str, target_city: str) -> bool:
        """
        Check if the document city matches the target city using stricter matching

        Args:
            doc_city: City name from document metadata
            target_city: Target city name from user query

        Returns:
            True if cities match, False otherwise
        """
        # Handle empty strings
        if not doc_city or not target_city:
            return False

        # Exact match
        if doc_city.lower() == target_city.lower():
            return True

        # Normalize both strings: lowercase, remove spaces and punctuation
        doc_city_norm = re.sub(r"[^a-z0-9]", "", doc_city.lower())
        target_city_norm = re.sub(r"[^a-z0-9]", "", target_city.lower())

        # Exact match on normalized strings
        if doc_city_norm == target_city_norm:
            return True

        # Check for a more strict containment (to avoid false positives like "London" and "Orlando")
        # Only match if the city name is a whole word within the other
        doc_words = set(re.findall(r"\b\w+\b", doc_city.lower()))
        target_words = set(re.findall(r"\b\w+\b", target_city.lower()))

        # Check for word-level matches
        common_words = doc_words.intersection(target_words)

        # Return True if there's at least one significant word in common
        # and it's at least 4 characters long (to avoid matching on short words like "the", "in", etc.)
        return any(len(word) >= 4 for word in common_words)

    def _assign_price_based_on_sentiment(self, hotel_info) -> Dict[str, Any]:
        import random

        """
        Assign price to RAG results based on quality category
        """
        # Extract sentiment category
        quality = hotel_info.metadata.get("description_sentiment_category", "")

        # Assign base rate based on quality category
        if quality == "Amazing":
            base_rate = 150.0 + (10.0 * random.random())  # 150-160 price range
        elif quality == "Great":
            base_rate = 130.0 + (10.0 * random.random())  # 130-140 price range
        elif quality == "Good":
            base_rate = 110.0 + (10.0 * random.random())  # 110-120 price range
        else:
            base_rate = 90.0 + (10.0 * random.random())  # 90-100 price range

        # Create price dictionary
        return {
            "currency": "USD",
            "total": str(base_rate),
            "base": str(base_rate),
            "taxes": [],
            "variations": {},
        }

    def extract_description_from_html(self, html_text):
        """Extract clean text from HTML formatted description"""
        # Replace HTML tags with proper formatting
        description = html_text.replace("<p>", "").replace("</p>", "\n")
        description = description.replace("<HeadLine>", "").replace("</HeadLine>", "")
        # Remove any remaining HTML tags
        import re

        description = re.sub(r"<[^>]+>", "", description)
        return description.strip()

    def rag_results_to_hotel_info(
        self, results: List[Document], from_class=None
    ) -> List:
        """
        Convert RAG results to HotelInfo objects with pricing information
        """
        hotels = []

        for doc in results:
            try:
                # Extract metadata
                hotel_id = doc.metadata.get("hotel_id", "Unknown")
                hotel_name = doc.metadata.get("hotel_name", "Unknown")

                # Log the raw document content for debugging
                logger.info(f"Processing RAG document for hotel: {hotel_name}")
                logger.info(f"Document content: {doc.page_content[:200]}...")

                # Extract content with improved parsing for different formats
                content_lines = doc.page_content.split("\n")
                description = ""
                facilities = ""

                # Enhanced extraction logic to handle various formats
                for line in content_lines:
                    line_lower = line.lower()

                    # Extract description
                    if line.startswith("Description:"):
                        raw_description = line.split(":", 1)[1].strip()
                        description = self.extract_description_from_html(
                            raw_description
                        )
                    elif "description:" in line_lower and not description:
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            raw_description = parts[1].strip()
                            description = self.extract_description_from_html(
                                raw_description
                            )

                    # Extract facilities/amenities with multiple format support
                    if line.startswith("Facilities:"):
                        facilities = line.split(":", 1)[1].strip()
                    elif "facilities:" in line_lower and not facilities:
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            facilities = parts[1].strip()
                    elif line.startswith("Amenities:") or "amenities:" in line_lower:
                        parts = line.split(":", 1)
                        if len(parts) > 1 and not facilities:
                            facilities = parts[1].strip()
                    elif "hotel facilities:" in line_lower and not facilities:
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            facilities = parts[1].strip()
                    elif "hotelfacilities:" in line_lower and not facilities:
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            facilities = parts[1].strip()

                # Log extracted facilities
                logger.info(f"Extracted raw facilities: {facilities}")

                # Parse facilities into list
                amenities = []
                if facilities:
                    # Handle different separator formats (comma, semicolon, pipe)
                    if "," in facilities:
                        amenities = [
                            f.strip() for f in facilities.split(",") if f.strip()
                        ]
                    elif ";" in facilities:
                        amenities = [
                            f.strip() for f in facilities.split(";") if f.strip()
                        ]
                    elif "|" in facilities:
                        amenities = [
                            f.strip() for f in facilities.split("|") if f.strip()
                        ]
                    else:
                        # Single item or space-separated
                        amenities = [facilities.strip()]

                # Check if amenities might be in a different field
                if not amenities:
                    for line in content_lines:
                        if "HotelFacilities:" in line or "Hotel Facilities:" in line:
                            facilities_part = line.split(":", 1)[1].strip()
                            if facilities_part:
                                if "," in facilities_part:
                                    amenities = [
                                        f.strip()
                                        for f in facilities_part.split(",")
                                        if f.strip()
                                    ]
                                elif ";" in facilities_part:
                                    amenities = [
                                        f.strip()
                                        for f in facilities_part.split(";")
                                        if f.strip()
                                    ]
                                else:
                                    amenities = [facilities_part.strip()]

                # Try to directly get facilities from metadata if available
                if not amenities and "facilities" in doc.metadata:
                    facilities_meta = doc.metadata.get("facilities", "")
                    if isinstance(facilities_meta, list):
                        amenities = facilities_meta
                    elif isinstance(facilities_meta, str) and facilities_meta.strip():
                        amenities = [
                            f.strip() for f in facilities_meta.split(",") if f.strip()
                        ]

                # If no amenities found, try to extract from description
                if not amenities and description:
                    common_amenities = [
                        "pool",
                        "wifi",
                        "breakfast",
                        "gym",
                        "spa",
                        "restaurant",
                        "bar",
                        "parking",
                        "air conditioning",
                        "fitness",
                        "sauna",
                        "beach",
                        "room service",
                        "concierge",
                        "free wifi",
                    ]
                    desc_lower = description.lower()
                    for amenity in common_amenities:
                        if amenity.lower() in desc_lower:
                            amenities.append(amenity.capitalize())

                # Also check the hotel name for obvious amenities
                hotel_name_lower = hotel_name.lower()
                for amenity in ["spa", "resort", "beach"]:
                    if (
                        amenity in hotel_name_lower
                        and amenity.capitalize() not in amenities
                    ):
                        amenities.append(amenity.capitalize())

                # Add some default amenities if none were found (for better UX)
                if not amenities:
                    amenities = ["WiFi", "Air conditioning"]
                    logger.info(f"No amenities found, adding defaults for {hotel_name}")

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
                sentiment_category = doc.metadata.get(
                    "description_sentiment_category", ""
                )

                # Log the final amenities list
                logger.info(f"Final amenities list for {hotel_name}: {amenities}")

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
                    # Add sentiment category as an attribute
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
            except Exception as e:
                logger.error(f"Error processing RAG result: {str(e)}")
                continue

        return hotels

    def inspect_vectorstore(self, limit=5):
        """Inspect the content of the vectorstore for debugging"""
        if not self.vector_store and not self.load_rag_system():
            logger.error("RAG system not available")
            return "Vector store not loaded"

        try:
            # Get documents from the vector store
            # In FAISS, we need to use a dummy query to get documents
            documents = self.vector_store.similarity_search("hotel", k=limit)

            results = []
            for i, doc in enumerate(documents):
                results.append(f"Document {i+1}:")
                results.append(f"Content: {doc.page_content}")
                results.append(f"Metadata: {doc.metadata}")
                results.append("---")

            return "\n".join(results)
        except Exception as e:
            logger.error(f"Error inspecting vector store: {str(e)}")
            return f"Error inspecting vector store: {str(e)}"
