
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """
You are SmartHire AI Career Mentor.

Help students with:
- Career selection and career roadmaps
- Resume improvement
- Technical skills and project ideas
- Interview preparation
- Job descriptions and skill gaps

Use the supplied SmartHire context when relevant.
Never invent details about the user's resume or job listings.
If information is missing, explain that clearly.
Give practical, beginner-friendly advice.
Ask follow-up questions when necessary.
Never promise that the user will get a job.
"""


def ask_mentor(messages, context=""):
    system_message = SYSTEM_PROMPT

    if context.strip():
        system_message += (
            "\n\nRelevant SmartHire context:\n" + context
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            *messages[-12:]
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running "
            "and llama3.2:3b has been downloaded."
        ) from exc

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The AI model took too long to respond. Please try again."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        try:
            detail = response.json().get("error", "")
        except Exception:
            detail = ""

        raise RuntimeError(
            "Ollama returned an error. Check the model installation. "
            + detail
        ) from exc

    answer = response.json().get(
        "message", {}
    ).get("content", "").strip()

    if not answer:
        raise RuntimeError(
            "The AI model returned an empty response."
        )

    return answer