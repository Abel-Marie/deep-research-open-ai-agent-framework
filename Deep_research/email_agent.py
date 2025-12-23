import os
from typing import Dict
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from agents import Agent, function_tool


@function_tool
def send_email(
    subject: str,
    html_body: str,
    recipient_email: str = "abelmarie49@gmail.com"
) -> Dict[str, str]:
    """
    Send an HTML email using BREVO (Sendinblue).
    Returns a structured status object suitable for agent reasoning.
    """

    # Mock mode (no API key)
    if not os.environ.get("BREVO_API_KEY"):
        return {
            "status": "success",
            "mode": "mock",
            "recipient": recipient_email,
            "subject": subject,
            "message_id": None,
        }

    try:
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key["api-key"] = os.environ.get("BREVO_API_KEY")

        api_client = sib_api_v3_sdk.ApiClient(configuration)
        api_instance = sib_api_v3_sdk.TransactionalEmailsApi(api_client)

        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
            to=[{"email": recipient_email}],
            sender={"email": "abelmarie49@gmail.com", "name": "Deep Research AI"},
            subject=subject,
            html_content=html_body,
        )

        response = api_instance.send_transac_email(send_smtp_email)

        return {
            "status": "success",
            "mode": "live",
            "recipient": recipient_email,
            "subject": subject,
            "message_id": str(response.message_id),
        }

    except ApiException as e:
        return {
            "status": "error",
            "mode": "live",
            "recipient": recipient_email,
            "subject": subject,
            "error": str(e),
        }


INSTRUCTIONS = (
    "You are an email delivery agent.\n"
    "You will be provided with a detailed research report.\n\n"
    "Your task is to:\n"
    "- Write a clear, professional email subject\n"
    "- Convert the report into a well-formatted HTML email body\n"
    "- Send exactly ONE email using the send_email tool\n\n"
    "Requirements:\n"
    "- Use clean HTML (not Markdown)\n"
    "- Do not include explanations or commentary\n"
    "- Call the tool once and only once"
)

email_agent = Agent(
    name="EmailAgent",
    instructions=INSTRUCTIONS,
    tools=[send_email],
    model="gemini-2.5-flash-lite",
)
