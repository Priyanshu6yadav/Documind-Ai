from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import get_llm


def create_query_rewriter():
    """
    Build an LCEL chain that rewrites follow-up questions.
    """

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a query rewriting assistant for DocuMind AI.
            Your task is to rewrite the latest user question
            into a standalone search query using conversation history.
            Rules:
                1. Preserve the original meaning.
                2. Resolve references such as "it", "they", and "that method".
                3. Do not answer the question.
                4. Do not invent facts or document details.
                5. If the question is already standalone, return it unchanged.
                6. Return only the rewritten question.
            """
        ),
        (
            "placeholder",
            "{chat_history}"
        ),
        (
            "human",
            "{question}"
        )
    ])

    chain = prompt | get_llm() | StrOutputParser()

    return chain


def rewrite_question(
    question: str,
    chat_history: list[BaseMessage],
) -> str:
    """
    Rewrite a question using previous conversation messages.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # No history means the question is already our search input.
    if not chat_history:
        return question.strip()

    chain = create_query_rewriter()

    rewritten_question = chain.invoke({
        "chat_history": chat_history,
        "question": question,
    }).strip()

    return rewritten_question or question.strip()


if __name__ == "__main__":

    history = [
        HumanMessage(
            content="What is the main objective of the cloud-removal paper?"
        ),
        AIMessage(
            content=(
                "The paper investigates removing clouds "
                "from satellite imagery."
            )
        ),
    ]

    question = "Which dataset did they use?"

    rewritten = rewrite_question(
        question=question,
        chat_history=history,
    )

    print("Original Question:")
    print(question)

    print("\nRewritten Question:")
    print(rewritten)