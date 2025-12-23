from pydantic import BaseModel, Field
from agents import Agent

HOW_MANY_SEARCHES = 3

INSTRUCTIONS = (
    f"You are a research planning agent. Given a user query, generate exactly "
    f"{HOW_MANY_SEARCHES} high-quality and non-overlapping web search queries "
    "that together would best answer the question.\n\n"
    "Each search should have a clear and distinct purpose (e.g., background, "
    "recent developments, expert analysis).\n\n"
    "Output MUST be a single valid JSON object with one key: 'searches'.\n"
    "The value of 'searches' MUST be a list of objects, each containing:\n"
    "- 'reason': why this search is necessary\n"
    "- 'query': the exact search string to use\n\n"
    "Do NOT include markdown, explanations, or any extra text. "
    "Do NOT include more or fewer than the required number of searches."
)

class WebSearchItem(BaseModel):
    reason: str = Field(
        description="Why this specific search is important for answering the query."
    )
    query: str = Field(
        description="A precise and effective web search query."
    )

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(
        description="Exactly the required number of web searches to perform."
    )

planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model="gemini-2.5-flash-lite",
    output_type=WebSearchPlan,
)
