import os
from typing import Optional

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import ListSortOrder


load_dotenv()


def get_client() -> AIProjectClient:
    """
    Build the AIProjectClient. Prefer connection string if provided; otherwise use endpoint + DefaultAzureCredential.
    """
    conn = os.getenv("PROJECT_CONNECTION_STRING")
    if conn:
        return AIProjectClient.from_connection_string(conn, credential=DefaultAzureCredential(exclude_environment_credential=True,  
        exclude_managed_identity_credential=True))
    endpoint = os.environ["PROJECT_ENDPOINT"]
    return AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential(exclude_environment_credential=True,  
        exclude_managed_identity_credential=True))


def run_query(project: AIProjectClient, orchestrator_agent_id: str, thread_id: str, user_query: str) -> None:
    # Post the user message
    project.agents.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_query,
    )

    # Kick off the run (blocking until done)
    run = project.agents.runs.create_and_process(
        thread_id=thread_id,
        agent_id=orchestrator_agent_id,
    )

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")
        return

    # List messages in ascending order and print the latest agent reply
    messages = list(project.agents.messages.list(thread_id=thread_id, order=ListSortOrder.ASCENDING))

    def extract_text(msg) -> str:
        # Prefer text_messages if present
        if getattr(msg, "text_messages", None):
            parts = []
            for tm in msg.text_messages:
                if getattr(tm, "text", None):
                    val = getattr(tm.text, "value", None)
                    if val:
                        parts.append(val)
            if parts:
                return parts[-1]
        # Fallback to content parts with .text or .value
        if getattr(msg, "content", None):
            parts = []
            for c in msg.content:
                if hasattr(c, "text") and c.text:
                    # text might be a string or have .value
                    val = c.text if isinstance(c.text, str) else getattr(c.text, "value", None)
                    if val:
                        parts.append(val)
                if hasattr(c, "value") and c.value:
                    parts.append(c.value)
            if parts:
                return parts[-1]
        # Last resort: string repr
        return ""

    for message in reversed(messages):
        role = str(getattr(message, "role", "")).lower()
        if "agent" in role or "assistant" in role:
            text = extract_text(message)
            if text:
                print(f"Agent: {text}")
                return
            # If no text extracted, show the raw message once for debugging
            print(f"Agent: (no parsed text) raw={message}")
            return

    print("Agent: (no reply found)")


def main() -> None:
    orchestrator_agent_id = os.environ["ORCHESTRATOR_AGENT_ID"]
    project = get_client()
    # Create a single thread for the interactive session.
    thread = project.agents.threads.create()
    print(f"Interactive mode. Thread: {thread.id}. Type your question, or 'quit' to exit.")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"quit", "exit"}:
            print("Exiting.")
            break
        if not user_query:
            continue
        run_query(project, orchestrator_agent_id, thread.id, user_query)


if __name__ == "__main__":
    main()
