from agents import Agent, function_tool
from duckduckgo_search import DDGS

from agents import function_tool
from duckduckgo_search import DDGS

@function_tool
def search_web(query: str) -> dict:
    """
    Searches the web using DuckDuckGo and returns the top results
    in a structured format suitable for agent reasoning.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as e:
        return {
            "error": "search_failed",
            "message": str(e),
            "query": query,
        }

    if not results:
        return {
            "query": query,
            "results": []
        }

    return {
        "query": query,
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            }
            for r in results
        ]
    }


INSTRUCTIONS = (
    "You are a research assistant. "
    "When given a query, you MUST call the search_web tool to gather information. "
    "Use the returned results to write a concise summary (2–3 paragraphs, under 300 words). "
    "Focus on key facts and trends. Do not mention sources explicitly unless necessary."
)


search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    tools=[search_web],
    model="gemini-2.5-flash-lite", 
)