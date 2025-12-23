import asyncio
from typing import AsyncGenerator, Union
from agents import Runner
from planner_agent import planner_agent
from search_agent import search_agent
from writer_agent import writer_agent, ReportData
from email_agent import email_agent


class ResearchManager:
    def __init__(self):
        self.runner = Runner()

    async def run(self, query: str) -> AsyncGenerator[Union[str, ReportData], None]:
        """
        Orchestrates the research process:
        1. Plan searches
        2. Execute searches
        3. Write final report
        """

        # 1. Planning phase
        yield "Creating research plan..."
        plan_result = await self.runner.run(
            agent=planner_agent,
            messages=[{
                "role": "user",
                "content": f"Create a research plan for: {query}"
            }]
        )

        plan = plan_result.output  # WebSearchPlan

        # 2. Research/Search phase
        all_results = []

        for i, item in enumerate(plan.searches):
            yield f"Searching ({i + 1}/{len(plan.searches)}): {item.query}..."

            search_result = await self.runner.run(
                agent=search_agent,
                messages=[{
                    "role": "user",
                    "content": item.query
                }]
            )

            search_data = search_result.output

            # Normalize structured search output into readable context
            if isinstance(search_data, dict) and "results" in search_data:
                formatted = [
                    f"- {r['title']}\n  {r['snippet']}\n  Source: {r['url']}"
                    for r in search_data["results"]
                ]
                block = f"Search query: {search_data.get('query')}\n" + "\n".join(formatted)
                all_results.append(block)

        research_context = "\n\n".join(all_results)

        # 3. Writing phase
        yield "Writing final report..."
        report_result = await self.runner.run(
            agent=writer_agent,
            messages=[{
                "role": "user",
                "content": (
                    f"Original research query:\n{query}\n\n"
                    f"Collected research findings:\n{research_context}"
                )
            }]
        )

        report: ReportData = report_result.output

        yield "Finalizing report display..."
        yield report

    async def send_report_email(self, report: ReportData, email: str, query: str):
        """
        Sends the report via the email agent.
        The agent is responsible for converting Markdown to HTML.
        """
        return await self.runner.run(
            agent=email_agent,
            messages=[{
                "role": "user",
                "content": (
                    f"Send the following research report via email.\n\n"
                    f"Email subject: Deep Research Report – {query}\n\n"
                    f"Report content (Markdown):\n{report.markdown_report}"
                )
            }]
        )
