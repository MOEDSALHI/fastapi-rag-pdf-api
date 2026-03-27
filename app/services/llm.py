from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import LLMResponseGenerationError


def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
    )


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)

    return (
        "You are a helpful assistant answering questions strictly based on the provided document context.\n"
        "If the answer is not present in the context, say that the information is not available in the document.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def generate_rag_answer(question: str, context_chunks: list[str]) -> str:
    if not question.strip():
        raise ValueError("question must not be empty.")

    if not context_chunks:
        raise ValueError("context_chunks must not be empty.")

    prompt = build_rag_prompt(question, context_chunks)

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the provided document context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMResponseGenerationError(
                "The chat model returned an empty response."
            )

        return content.strip()

    except LLMResponseGenerationError:
        raise
    except Exception as exc:
        raise LLMResponseGenerationError(
            "An error occurred while generating the answer."
        ) from exc