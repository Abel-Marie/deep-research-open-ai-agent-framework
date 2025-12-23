import streamlit as st
import asyncio
from dotenv import load_dotenv
from research_manager import ResearchManager
from writer_agent import ReportData

load_dotenv(override=True)

st.set_page_config(page_title="Deep Research AI", page_icon="🕵️‍♂️")

st.title("🕵️‍♂️ Free Deep Research Agent")
st.caption("Powered by Gemini 2.5 Flash Lite & DuckDuckGo")

# -----------------------------
# Session State Initialization
# -----------------------------
if "report_content" not in st.session_state:
    st.session_state.report_content = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "manager" not in st.session_state:
    st.session_state.manager = ResearchManager()

if "status_log" not in st.session_state:
    st.session_state.status_log = []


# -----------------------------
# Async helpers
# -----------------------------
def run_async(coro):
    """Safely run async code in Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.create_task(coro)
    else:
        return asyncio.run(coro)


async def run_research(user_query: str):
    manager = st.session_state.manager
    status_placeholder = st.empty()

    async for update in manager.run(user_query):
        if isinstance(update, str):
            st.session_state.status_log.append(update)
            status_placeholder.info("\n".join(st.session_state.status_log))
        elif isinstance(update, ReportData):
            return update


# -----------------------------
# UI: Research Input
# -----------------------------
query = st.text_input("What topic would you like to research?")
start_btn = st.button("Start Research", type="primary")

if start_btn and query:
    st.session_state.last_query = query
    st.session_state.report_content = None
    st.session_state.status_log = []

    with st.spinner("Initializing agents..."):
        try:
            final_report = run_async(run_research(query))
            st.session_state.report_content = final_report
        except Exception as e:
            st.error(f"An error occurred: {e}")


# -----------------------------
# Display Report
# -----------------------------
if st.session_state.report_content:
    r = st.session_state.report_content

    st.success("Research Complete!")

    # -----------------------------
    # Email Section
    # -----------------------------
    st.divider()
    st.subheader("📬 Send Report")

    email_input = st.text_input(
        "Enter recipient email address:",
        value="exampleemail.com",
        key="email_recipient"
    )

    send_email_btn = st.button(
        "Send Report",
        type="secondary",
        disabled=not bool(email_input)
    )

    if send_email_btn:
        with st.spinner("Sending email..."):
            try:
                run_async(
                    st.session_state.manager.send_report_email(
                        r,
                        email_input,
                        st.session_state.last_query
                    )
                )
                st.toast("Email sent successfully! ✅")
            except Exception as e:
                st.error(f"Failed to send email: {e}")

    # -----------------------------
    # Report Content
    # -----------------------------
    st.divider()
    st.subheader("Summary")
    st.write(r.short_summary)

    st.divider()
    st.markdown(r.markdown_report)

    st.divider()
    st.subheader("Follow-up Questions")
    for q in r.follow_up_questions:
        st.markdown(f"- {q}")
