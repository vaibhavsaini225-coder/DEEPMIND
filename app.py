"""
LangChain Agent with Tools and Memory (LangChain v1 - Groq Version)
A beginner-friendly agent using Tavily search, datetime, weather tools, and chat history
"""

# CORRECTED IMPORTS FOR LANGCHAIN V1
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq  # Changed from langchain_openai
from langchain_community.tools.tavily_search import TavilySearchResults
from datetime import datetime, timedelta
import requests
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# STEP 1: Define Custom Tools
# =============================================================================

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None 

@tool
def get_current_datetime() -> str:
    """Return current date & time in Indian Standard Time (IST)."""
    try:
        if ZoneInfo is not None:
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
        else:
            # Fallback if zoneinfo not available
            now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        # Human friendly format, e.g., "2025-12-13 16:37:45 (IST)"
        return now.strftime("%Y-%m-%d %H:%M:%S (IST)")
    except Exception as e:
        return f"Error getting current time: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    
    Args:
        city: Name of the city to get weather for
    """
    # Using wttr.in API (free, no key required)
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            weather_desc = current['weatherDesc'][0]['value']
            temp_c = current['temp_C']
            feels_like = current['FeelsLikeC']
            humidity = current['humidity']
            
            return f"Weather in {city}: {weather_desc}, Temperature: {temp_c}°C (feels like {feels_like}°C), Humidity: {humidity}%"
        else:
            return f"Could not fetch weather for {city}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

# =============================================================================
# STEP 2: Initialize Chat History (Short-term Memory)
# =============================================================================

# Initialize chat_history as a global variable
chat_history = []

# =============================================================================
# STEP 3: Create the Agent
# =============================================================================

def create_agent():
    """Initialize and return the agent executor."""
    
    # Initialize the Groq LLM with tool calling support
    llm = ChatGroq(
       model_name="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
        model_kwargs={
            "tool_choice": "auto",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_datetime",
                        "description": "Get the current date and time."
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string", "description": "The city name"}
                            },
                            "required": ["city"]
                        }
                    }
                }
            ]
        }
    )
    
    # Initialize Tavily search tool (LangChain built-in)
    tavily_tool = TavilySearchResults(
        max_results=3,
        search_depth="basic",  # or "advanced" for more detailed results
        include_answer=True,
        include_raw_content=False
    )
    
    # Define all tools
    tools = [get_current_datetime, get_weather, tavily_tool]
    
    # Create prompt template with memory placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to tools for:
        - Getting current date and time (use tool: get_current_datetime)
        - Checking weather for any city (use tool: get_weather)
        - Searching the web for information (use tool: tavily_search_results_json)
        - Remembering chat history to provide context for future interactions
        
        Use these tools when needed to provide accurate and helpful responses.
        Time and weather information should be current.
        Use Indian Standard Time (IST) for all time-related queries.
        Be conversational and remember the context from previous messages."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=60
    )
    
    return agent_executor

# =============================================================================
# STEP 4: Run the Agent with Memory
# =============================================================================

def chat(user_input: str, agent_executor):
    """
    Process user input and maintain chat history.
    
    Args:
        user_input: The user's message
        agent_executor: The agent executor instance
    
    Returns:
        The agent's response
    """
    global chat_history
    
    try:
        # Initialize chat_history if it's None
        if chat_history is None:
            chat_history = []
            
        # Convert chat history to proper message format for LangChain v1
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role == "human":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "assistant" and content:  # Only add non-empty messages
                    if isinstance(content, str):
                        formatted_history.append(AIMessage(content=content))
        
        # Ensure the agent_executor is initialized
        if agent_executor is None:
            agent_executor = create_agent()
        
        # Prepare the input for the agent
        input_data = {
            "input": user_input,
            "chat_history": formatted_history or []
        }
        
        # Run the agent with current chat history
        try:
            response = agent_executor.invoke(input_data)
            
            # Get the output safely with more robust error handling
            if response is None:
                output = "No response was generated. Please try again."
            elif isinstance(response, dict):
                output = response.get('output', '')
                if not output:  # If output is empty or None
                    output = "I didn't get a proper response. Could you rephrase your question?"
            elif hasattr(response, 'output') and response.output is not None:
                output = str(response.output)
            else:
                output = str(response) if response is not None else "No response was generated."
                
            # Ensure output is a non-empty string
            if not output or not isinstance(output, str):
                output = "I'm having trouble understanding. Could you rephrase your question?"
                
        except Exception as e:
            output = f"I encountered an error: {str(e)}. Could you please rephrase your question?"
        
        # Update chat history with the response
        if output and output != 'No response generated':
            chat_history.append(("human", user_input))
            chat_history.append(("assistant", output))
        
        # Keep only last 10 exchanges (20 messages) to prevent context from growing too large
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        return output if output else "I'm not sure how to respond to that. Could you rephrase?"
        
    except Exception as e:
        error_msg = f"Error in chat function: {str(e)}"
        print(error_msg)  # Log the error for debugging
        return "I'm sorry, I encountered an error processing your request. Please try again."

# =============================================================================
# STEP 5: Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Agent with Tools and Memory (Groq Version)")
    print("=" * 60)
    print("\n📋 Loading API keys from .env file...")
    print("\nRequired API Keys:")
    print("- GROQ_API_KEY (for LLM)")
    print("- TAVILY_API_KEY (for web search)")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not found in .env file!")
        print("\nPlease create a .env file with:")
        print("GROQ_API_KEY=gsk-your-groq-key-here")
        print("TAVILY_API_KEY=tvly-your-tavily-key-here")
        sys.exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠️  TAVILY_API_KEY not found in .env file!")
        print("\nPlease add to your .env file:")
        print("TAVILY_API_KEY=tvly-your-tavily-key-here")
        sys.exit(1)
    
    print("✅ API keys loaded successfully!")
    
    # Initialize agent
    print("\n🤖 Initializing agent...")
    try:
        agent_executor = create_agent()
        print("✅ Agent ready!\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {str(e)}")
        sys.exit(1)
    
    # Interactive chat loop
    print("Chat with the agent (type 'quit' to exit, 'history' to see chat history):\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() == 'history':
            print("\n--- Chat History ---")
            if not chat_history:
                print("No chat history yet.")
            else:
                for role, message in chat_history:
                    preview = message[:100] + "..." if len(message) > 100 else message
                    print(f"{role}: {preview}")
            print("--- End History ---\n")
            continue
        
        if user_input.lower() == 'clear':
            chat_history.clear()
            print("\n✅ Chat history cleared!\n")
            continue
        
        try:
            response = chat(user_input, agent_executor)
            print(f"\n🤖 Agent: {response}\n")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

# =============================================================================
# Installation & Example Usage:
# =============================================================================
"""
INSTALLATION (LangChain v1 with Groq):
--------------------------------------
pip install langchain langchain-groq langchain-core langchain-community tavily-python requests python-dotenv

SETUP .ENV FILE:
----------------
Create a file named .env in the same directory as your script:

GROQ_API_KEY=gsk-your-groq-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here

Example .env file content:
--------------------------
GROQ_API_KEY=gsk_abc123xyz...
TAVILY_API_KEY=tvly-abc123xyz...

RUN:
----
python agent.py

EXAMPLE CONVERSATION:
--------------------
You: What time is it?
Agent: [Uses get_current_datetime tool]

You: What's the weather in Mumbai?
Agent: [Uses get_weather tool]

You: Search for recent AI news
Agent: [Uses TavilySearchResults tool]

You: What was the first thing I asked?
Agent: [Uses chat_history memory to recall]

COMMANDS:
---------
- 'quit' or 'exit' - Exit the program
- 'history' - View chat history
- 'clear' - Clear chat history

GROQ MODELS AVAILABLE:
---------------------
- llama-3.3-70b-versatile (Best for tool calling - RECOMMENDED)
- llama-3.1-70b-versatile
- mixtral-8x7b-32768
- gemma2-9b-it

KEY FEATURES:
-------------
✅ Uses Groq API instead of OpenAI (faster and often free)
✅ Loads API keys from .env file (using python-dotenv)
✅ LangChain v1 compatible imports
✅ Proper message formatting with HumanMessage/AIMessage
✅ Simple array-based chat history (last 10 exchanges)
✅ Three useful tools: datetime, weather, web search
✅ Error handling and user-friendly messages
✅ Indian Standard Time (IST) support

GET API KEYS:
-------------
- Groq: https://console.groq.com/keys (Free tier available!)
- Tavily: https://app.tavily.com/

NOTES:
------
- Groq provides very fast inference
- llama-3.3-70b-versatile has excellent tool calling capabilities
- Free tier includes generous rate limits
"""