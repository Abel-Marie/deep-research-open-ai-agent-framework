from pydantic import BaseModel, Field
from agents import Agent

INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive, high-quality report for a research query. "
    "You will be provided with the original query and supporting research material gathered by a research assistant.\n\n"
    "First, internally plan a clear outline that defines the structure and logical flow of the report. "
    "Then, write the report following that structure.\n\n"
    "The report should synthesize the provided information, resolve inconsistencies where possible, "
    "and focus on clarity, accuracy, and insight rather than copying source text.\n\n"
    "Formatting requirements:\n"
    "- The report MUST be written in Markdown\n"
    "- Use clear section headings (##, ###) to reflect the planned outline\n"
    "- Use bullet points or tables where they improve readability\n\n"
    "Your response MUST be a single valid JSON object with exactly the following fields:\n"
    "- 'short_summary': a concise 2–3 sentence summary of the key findings\n"
    "- 'markdown_report': the full Markdown report\n"
    "- 'follow_up_questions': a list of specific, research-oriented questions for further exploration\n\n"
    "Do NOT include any preamble, explanations, or conversational text outside the JSON object."
)

class ReportData(BaseModel):
    short_summary: str = Field(
        description="A concise 2–3 sentence summary of the key findings."
    )
    markdown_report: str = Field(
        description="The final, well-structured research report written in Markdown."
    )
    follow_up_questions: list[str] = Field(
        description="Specific, actionable research questions to explore next."
    )

writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model="gemini-2.5-flash-lite",
    output_type=ReportData,
)
