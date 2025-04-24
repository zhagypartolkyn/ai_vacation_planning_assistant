AI Travel Assistant
An intelligent travel planning application that helps users find flights, hotels, and combined options within their budget.

Features
✈️ Flight Search: Find the best flights between your origin and destination
🏨 Hotel Search: Find hotels that match your preferences and budget
💰 Budget Management: Smart allocation of your total budget between flights and hotels
🔍 Intelligent Location Resolution: Advanced city code lookup with multiple fallbacks
📋 Combined Itineraries: View options that include both flights and hotels
🧠 AI-Powered Fallbacks: Uses OpenAI when traditional lookups fail
Architecture
The application is built with the following components:

Streamlit UI: User-friendly interface for trip planning
Flight Agent: Searches for flights using Amadeus API
Hotel Agent: Searches for hotels using Amadeus API
Enhanced Code Converters: Robust city/airport code lookup with multiple fallbacks
Budget Manager: Manages budget allocation and filtering
OpenAI Connector: AI-powered fallback for code lookups
Setup
Prerequisites
Python 3.8+
Amadeus API credentials
OpenAI API key (optional, for advanced fallbacks)
Installation
Clone the repository:
bash
git clone https://github.com/yourusername/ai-travel-assistant.git
cd ai-travel-assistant
Create and activate a virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
bash
pip install -r requirements.txt
Create a .env file with your API keys:
AMADEUS_API_PROD_KEY=your_amadeus_api_key
AMADEUS_SECRET_PROD_KEY=your_amadeus_secret_key
OPENAI_API_KEY=your_openai_api_key
Running the Application
Start the Streamlit app:

bash
streamlit run app.py
The application will be available at http://localhost:8501


ai-travel-assistant/
├── app.py                 # Main Streamlit application
├── amadeus_client.py      # Amadeus API client
├── openai_connector.py    # OpenAI API connector for fallbacks
├── enhanced_converters.py # Code converters with fallbacks
├── budget_manager.py      # Budget allocation and filtering
├── rag_service.py         # Vector store for hotel information
├── flight_agent.py        # Flight search agent
├── hotel_agent.py         # Hotel search agent with RAG capabilities
├── hotel_vectorstore/     # Directory containing FAISS index for hotel RAG
├── requirements.txt       # Dependencies
└── README.md              # DocumentationHow It Works

Budget Allocation
70% of your total budget is allocated to flights and hotels
The application tries to find combinations that fit within this allocation
If no options fit the budget, it shows the cheapest combination with a warning
Code Conversion Process
The application converts city names to IATA codes using the following fallback chain:

Static dictionary lookup (fast, no API calls)
Amadeus API lookup (official source, requires API call)
OpenAI API fallback (AI-generated prediction, requires API call)
First 3 letters of city name (fallback of last resort)
Error Handling
API calls include retry logic with exponential backoff
If a service fails, the application continues with partial results when possible
User-friendly error messages provide guidance on how to resolve issues
Customization
You can adjust the following settings:

Budget allocation percentages in budget_manager.py
Static city/airport code dictionaries in enhanced_converters.py
API retry settings in amadeus_client.py
UI components and layout in app.py
Future Enhancements
Planned features for future development:

🗺️ Itinerary planning with day-by-day activities
🍽️ Restaurant recommendations at the destination
🎟️ Attraction and experience booking
📱 Mobile-optimized interface
🌐 Multi-language support
License
This project is licensed under the MIT License - see the LICENSE file for details.

