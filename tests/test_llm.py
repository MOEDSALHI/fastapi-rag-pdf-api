import pytest
from unittest.mock import Mock, patch

from app.core.exceptions import LLMResponseGenerationError
from app.services.llm import build_rag_prompt, generate_rag_answer


def test_build_rag_prompt_contains_question_and_context() -> None:
    prompt = build_rag_prompt(
        question="Quel est le loyer ?",
        context_chunks=[
            "Le loyer mensuel est de 950 euros.",
            "Le bail dure 3 ans.",
        ],
    )

    assert "Quel est le loyer ?" in prompt
    assert "Le loyer mensuel est de 950 euros." in prompt
    assert "Le bail dure 3 ans." in prompt


def test_generate_rag_answer_raises_error_for_empty_question() -> None:
    with pytest.raises(ValueError, match="question must not be empty."):
        generate_rag_answer("   ", ["chunk 1"])


def test_generate_rag_answer_raises_error_for_empty_context() -> None:
    with pytest.raises(ValueError, match="context_chunks must not be empty."):
        generate_rag_answer("Quel est le loyer ?", [])


def test_generate_rag_answer_returns_content() -> None:
    with patch("app.services.llm.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Le loyer mensuel est de 950 euros."))
        ]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_rag_answer(
            "Quel est le loyer ?",
            ["Le loyer mensuel est de 950 euros."],
        )

        assert result == "Le loyer mensuel est de 950 euros."


def test_generate_rag_answer_raises_error_on_empty_model_response() -> None:
    with patch("app.services.llm.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="   "))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        with pytest.raises(
            LLMResponseGenerationError,
            match="The chat model returned an empty response.",
        ):
            generate_rag_answer(
                "Quel est le loyer ?",
                ["Le loyer mensuel est de 950 euros."],
            )