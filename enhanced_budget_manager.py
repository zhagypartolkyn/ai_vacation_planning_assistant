"""
Enhanced Budget Manager module for the AI Travel Assistant.
Handles budget allocation and filtering of travel options.
Integrates with Budget Agent for detailed budget allocation.
"""

import logging
from typing import List, Dict, Any, Tuple
from budget_agent import allocate_budget

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BudgetManager:
    """Manager for trip budget allocation and filtering"""

    def __init__(
        self,
        total_budget: float,
        flight_percent: float = 0.35,
        hotel_percent: float = 0.35,
    ):
        """
        Initialize budget manager

        Args:
            total_budget: Total trip budget in USD
            flight_percent: Percentage of total budget to allocate to flights (default: 35%)
            hotel_percent: Percentage of total budget to allocate to hotels (default: 35%)
        """
        self.total_budget = total_budget
        self.flight_percent = flight_percent
        self.hotel_percent = hotel_percent

        # Calculate allocated budgets
        self.travel_budget = total_budget * (flight_percent + hotel_percent)
        self.flight_budget = total_budget * flight_percent
        self.hotel_budget = total_budget * hotel_percent
        self.remaining_budget = total_budget * (1 - flight_percent - hotel_percent)

        # Store detailed budget allocation
        self.detailed_allocation = None

        logger.info(f"Budget Manager initialized with total budget: ${total_budget}")
        logger.info(
            f"Flight budget: ${self.flight_budget}, Hotel budget: ${self.hotel_budget}"
        )

    def get_flight_budget(self) -> float:
        """Get flight budget"""
        return self.flight_budget

    def get_hotel_budget(self) -> float:
        """Get hotel budget"""
        return self.hotel_budget

    def get_remaining_budget(self) -> float:
        """Get remaining budget (for other expenses)"""
        return self.remaining_budget

    def get_detailed_allocation(
        self, min_budget: float = None, max_budget: float = None
    ) -> str:
        """
        Get detailed budget allocation using the budget_agent

        Args:
            min_budget: Optional minimum budget
            max_budget: Optional maximum budget

        Returns:
            Markdown-formatted detailed budget allocation
        """
        try:
            # Use provided budget range or default to current total budget
            min_budget = min_budget or self.total_budget * 0.85
            max_budget = max_budget or self.total_budget * 1.15

            # Call budget_agent's allocate_budget function
            allocation = allocate_budget(min_budget=min_budget, max_budget=max_budget)

            # Store the allocation for later reference
            self.detailed_allocation = allocation

            return allocation
        except Exception as e:
            logger.error(f"Error getting detailed budget allocation: {str(e)}")
            # Return basic allocation if the detailed one fails
            return self._get_basic_allocation_string()

    def _get_basic_allocation_string(self) -> str:
        """Generate basic budget allocation string"""
        result = "Here is a budget allocation for your trip: \n\n"
        result += f"Total budget: ${self.total_budget} \n\n"
        result += (
            f"- ✈️ Flights: ${self.flight_budget:.2f} ({self.flight_percent*100:.0f}%)\n"
        )
        result += (
            f"- 🏨 Hotels: ${self.hotel_budget:.2f} ({self.hotel_percent*100:.0f}%)\n"
        )
        result += f"- 🚗 Transport, 🍽️ Food & 🎫 Activities: ${self.remaining_budget:.2f} ({(1-self.flight_percent-self.hotel_percent)*100:.0f}%)\n"

        return result

    def extract_flight_price(self, flight: Dict[str, Any]) -> float:
        """Extract price from flight object"""
        try:
            # Handle different price formats
            price = flight.get("price", {}).get("total", 0)
            if isinstance(price, str):
                # Remove currency symbols and commas
                price = price.replace("$", "").replace(",", "")
                return float(price)
            return float(price)
        except (ValueError, TypeError):
            logger.warning(
                f"Could not parse flight price: {flight.get('price', 'Unknown')}"
            )
            return float("inf")  # Return infinity if price can't be parsed

    def extract_hotel_price(self, hotel: Dict[str, Any]) -> float:
        """
        Extract price from hotel object, handling both API and RAG hotels consistently
        """
        try:
            # Get price data
            price_data = hotel.get("price", {})

            # Handle different price formats
            if isinstance(price_data, dict):
                price = price_data.get("total", 0)
            else:
                price = price_data

            # Parse string prices
            if isinstance(price, str):
                # Remove currency symbols and commas
                price = price.replace("$", "").replace(",", "")
                return float(price)
            elif isinstance(price, (int, float)):
                return float(price)
            else:
                logger.warning(
                    f"Unknown price format for hotel {hotel.get('id')}: {price}"
                )
                return float("inf")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing hotel price for {hotel.get('id')}: {e}")
            return float("inf")  # Return infinity if price can't be parsed

    def calculate_total_hotel_cost(
        self, hotel: Dict[str, Any], num_nights: int
    ) -> float:
        """
        Calculate total hotel cost for the entire stay

        Args:
            hotel: Hotel object
            num_nights: Number of nights for the stay

        Returns:
            Total cost for the entire stay
        """
        nightly_rate = self.extract_hotel_price(hotel)
        if nightly_rate == float("inf"):
            return float("inf")

        return nightly_rate * num_nights

    def filter_flights_by_budget(
        self, flights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter flights by budget"""
        filtered = [
            f for f in flights if self.extract_flight_price(f) <= self.flight_budget
        ]
        logger.info(
            f"Filtered flights by budget: {len(filtered)} of {len(flights)} remain"
        )
        return filtered

    def filter_hotels_by_budget(
        self, hotels: List[Dict[str, Any]], nights: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Filter hotels by budget, accounting for total stay duration

        Args:
            hotels: List of hotel objects
            nights: Number of nights for the stay

        Returns:
            Filtered list of hotels within budget
        """
        # Calculate budget per night
        budget_per_night = self.hotel_budget / max(1, nights)
        logger.info(f"Hotel budget per night: ${budget_per_night} for {nights} nights")

        filtered = []
        for hotel in hotels:
            price = self.extract_hotel_price(hotel)
            if price <= budget_per_night:
                filtered.append(hotel)

        logger.info(
            f"Filtered hotels by budget: {len(filtered)} of {len(hotels)} remain"
        )
        return filtered

    def find_best_combinations(
        self,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]],
        max_combinations: int = 5,
        nights: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find best combinations of flights and hotels within budget

        Args:
            flights: List of flight options
            hotels: List of hotel options
            max_combinations: Maximum number of combinations to return
            nights: Number of nights for the stay

        Returns:
            List of combinations, each with flight, hotel, and total price
        """
        if not flights or not hotels:
            logger.warning("No flights or hotels to combine")
            return []

        combinations = []

        # Extract prices for all options
        flight_prices = [(f, self.extract_flight_price(f)) for f in flights]
        hotel_prices = [(h, self.extract_hotel_price(h)) for h in hotels]

        # Sort by price
        flight_prices.sort(key=lambda x: x[1])
        hotel_prices.sort(key=lambda x: x[1])

        # Log some debugging info
        logger.info(
            f"Finding combinations with {len(flight_prices)} flights and {len(hotel_prices)} hotels"
        )
        logger.info(
            f"Cheapest flight: ${flight_prices[0][1] if flight_prices else 'N/A'}"
        )
        logger.info(
            f"Cheapest hotel: ${hotel_prices[0][1] if hotel_prices else 'N/A'} per night"
        )

        # Generate combinations
        for flight, flight_price in flight_prices:
            if flight_price == float("inf"):
                continue

            for hotel, hotel_price in hotel_prices:
                if hotel_price == float("inf"):
                    continue

                # Calculate total price (hotel price is per night)
                total_hotel_price = hotel_price * nights
                total_price = flight_price + total_hotel_price

                # Check if within budget
                within_budget = total_price <= self.travel_budget

                # Create combination entry
                combination = {
                    "flight": flight,
                    "hotel": hotel,
                    "flight_price": flight_price,
                    "hotel_price": hotel_price,  # This is per night
                    "total_hotel_price": total_hotel_price,  # This is for the entire stay
                    "total_price": total_price,
                    "within_budget": within_budget,
                    "nights": nights,
                }

                combinations.append(combination)

                # Sort combinations by price if we have too many
                if len(combinations) > max_combinations * 2:
                    combinations.sort(key=lambda x: x["total_price"])
                    combinations = combinations[: max_combinations * 2]

        # Sort combinations by within_budget (True first) and then by price
        combinations.sort(key=lambda x: (not x["within_budget"], x["total_price"]))

        # Return the best combinations
        return combinations[:max_combinations]

    def get_budget_statistics(
        self,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]],
        nights: int = 5,
    ) -> Dict[str, Any]:
        """
        Get budget statistics to help user understand budget constraints

        Args:
            flights: List of flight options
            hotels: List of hotel options
            nights: Number of nights for the stay

        Returns:
            Dictionary with min/max/avg prices and budget recommendations
        """
        # Handle empty lists
        if not flights or not hotels:
            return {
                "min_flight_price": 0,
                "min_hotel_price": 0,
                "min_total_price": 0,
                "avg_flight_price": 0,
                "avg_hotel_price": 0,
                "budget_sufficient": False,
                "recommended_budget": self.total_budget,
                "current_budget": self.total_budget,
            }

        flight_prices = [self.extract_flight_price(f) for f in flights]
        hotel_prices = [self.extract_hotel_price(h) for h in hotels]

        # Remove any infinite values
        flight_prices = [p for p in flight_prices if p != float("inf")]
        hotel_prices = [p for p in hotel_prices if p != float("inf")]

        if not flight_prices or not hotel_prices:
            return {
                "min_flight_price": 0,
                "min_hotel_price": 0,
                "min_total_price": 0,
                "avg_flight_price": 0,
                "avg_hotel_price": 0,
                "budget_sufficient": False,
                "recommended_budget": self.total_budget,
                "current_budget": self.total_budget,
            }

        # Calculate statistics
        min_flight = min(flight_prices)
        min_hotel = min(hotel_prices)
        min_hotel_total = min_hotel * nights
        min_total = min_flight + min_hotel_total

        # Calculate averages
        avg_flight = sum(flight_prices) / len(flight_prices)
        avg_hotel = sum(hotel_prices) / len(hotel_prices)
        avg_hotel_total = avg_hotel * nights
        avg_total = avg_flight + avg_hotel_total

        # Provide budget recommendations and sufficiency check
        budget_sufficient = min_total <= self.travel_budget
        recommended_budget = max(self.total_budget, min_total * 1.2)  # 20% buffer

        logger.info(
            f"Budget statistics: min flight ${min_flight}, min hotel ${min_hotel}/night"
        )
        logger.info(f"Total minimum cost: ${min_total} for {nights} nights")
        logger.info(
            f"Budget sufficient: {budget_sufficient}, recommended: ${recommended_budget}"
        )

        return {
            "min_flight_price": min_flight,
            "min_hotel_price": min_hotel,
            "min_hotel_total": min_hotel_total,
            "min_total_price": min_total,
            "avg_flight_price": avg_flight,
            "avg_hotel_price": avg_hotel,
            "avg_hotel_total": avg_hotel_total,
            "avg_total_price": avg_total,
            "nights": nights,
            "budget_sufficient": budget_sufficient,
            "recommended_budget": recommended_budget,
            "current_budget": self.total_budget,
            "travel_budget": self.travel_budget,
        }

    def update_budget_allocation(
        self,
        new_total_budget: float = None,
        flight_percent: float = None,
        hotel_percent: float = None,
    ):
        """
        Update budget allocation percentages and/or total budget

        Args:
            new_total_budget: New total budget (optional)
            flight_percent: New flight budget percentage (optional)
            hotel_percent: New hotel budget percentage (optional)
        """
        # Update total budget if provided
        if new_total_budget is not None:
            self.total_budget = new_total_budget

        # Update percentages if provided
        if flight_percent is not None:
            self.flight_percent = flight_percent
        if hotel_percent is not None:
            self.hotel_percent = hotel_percent

        # Make sure percentages don't exceed 1.0
        total_percent = self.flight_percent + self.hotel_percent
        if total_percent > 1.0:
            # Scale down to fit within 1.0
            scale_factor = 1.0 / total_percent
            self.flight_percent *= scale_factor
            self.hotel_percent *= scale_factor
            logger.warning(
                f"Budget percentages exceeded 1.0, scaled down to flight: {self.flight_percent}, hotel: {self.hotel_percent}"
            )

        # Recalculate budgets
        self.travel_budget = self.total_budget * (
            self.flight_percent + self.hotel_percent
        )
        self.flight_budget = self.total_budget * self.flight_percent
        self.hotel_budget = self.total_budget * self.hotel_percent
        self.remaining_budget = self.total_budget * (
            1 - self.flight_percent - self.hotel_percent
        )

        logger.info(
            f"Budget updated: total ${self.total_budget}, flight ${self.flight_budget}, hotel ${self.hotel_budget}"
        )

    def allocate_budget_from_agent(
        self, min_budget: float = None, max_budget: float = None
    ):
        """
        Use the budget_agent to suggest budget allocation and update this manager

        Args:
            min_budget: Minimum budget to consider
            max_budget: Maximum budget to consider

        Returns:
            String representation of the allocated budget
        """
        try:
            if min_budget is None:
                min_budget = self.total_budget * 0.85
            if max_budget is None:
                max_budget = self.total_budget * 1.15

            # Get allocation from budget agent
            allocation_text = allocate_budget(
                min_budget=min_budget, max_budget=max_budget
            )

            # Parse the allocation to extract percentages
            try:
                # Extract flight percentage
                flight_line = [
                    line for line in allocation_text.split("\n") if "✈️ Flights" in line
                ]
                if flight_line:
                    flight_amount = float(
                        flight_line[0].split(":")[1].split("–")[0].strip()
                    )
                    flight_percent = flight_amount / min_budget
                else:
                    flight_percent = None

                # Extract hotel percentage
                hotel_line = [
                    line for line in allocation_text.split("\n") if "🏨 Hotels" in line
                ]
                if hotel_line:
                    hotel_amount = float(
                        hotel_line[0].split(":")[1].split("–")[0].strip()
                    )
                    hotel_percent = hotel_amount / min_budget
                else:
                    hotel_percent = None

                # Update budget manager if we extracted values
                if flight_percent and hotel_percent:
                    self.update_budget_allocation(
                        new_total_budget=max_budget,
                        flight_percent=flight_percent,
                        hotel_percent=hotel_percent,
                    )
                    logger.info(
                        f"Updated budget allocation based on budget agent: flight {flight_percent:.2f}, hotel {hotel_percent:.2f}"
                    )
            except Exception as e:
                logger.error(f"Failed to parse budget agent allocation: {e}")

            return allocation_text

        except Exception as e:
            logger.error(f"Error using budget agent: {str(e)}")
            return self._get_basic_allocation_string()
