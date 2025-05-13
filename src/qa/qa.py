import os
from typing import Tuple, List

from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import Document


# Load environment variables from .env file
load_dotenv()

# Constants / defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-3.5-turbo")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant that answers questions using the provided context snippets."
)

# Instantiate OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def answer_question(
    query: str,
    docs: List[Document],
) -> Tuple[str, List[Document]]:
    """
    Call the OpenAI chat API with provided document chunks as context
    and return the generated answer along with the list of Documents used.

    Args:
        query:  The user's natural-language question.
        docs:   List of Document objects (with metadata['chunk_id'],
                metadata['page_content'] or doc.page_content).

    Returns:
        A tuple (answer_text, docs) where answer_text is the LLM's response,
        and docs is unchanged list of Document objects.
    """
    if not docs:
        # no context provided
        return "I’m sorry, I don’t have any context to answer that.", []

    # Build context snippets, preferring .page_content but falling back to metadata
    context_snippets = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "<unknown_chunk>")
        # prefer direct page_content, else metadata entry
        snippet = doc.page_content 
        if not snippet:
            raise ValueError(f"Missing text for chunk {chunk_id}")
        context_snippets.append(f"[{chunk_id}]\n{snippet}")

    # Assemble prompt
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context snippets (count={len(context_snippets)}):\n\n"
        + "\n\n".join(context_snippets)
        + f"\n\nQuestion: {query}"
    )

    # Call the ChatCompletion API
    chat_resp = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
    )

    answer = chat_resp.choices[0].message.content.strip()
    return answer, docs

def main():
    """
    Demo of answer_question with an extended family/business story:

    - Mark is a plumber who owns a business in Springfield and has five kids.
    - Gigio is Mark's first kid and loves computers.
    - Mark's plumbing business is called "Mark's Pipes" and has been serving Springfield for 10 years.
    - His wife Lisa helps manage the office and schedules appointments.
    - Their second child, Anna, enjoys painting and art.
    - They also have a golden retriever named Buddy that often rides along to job sites.
    """
    from langchain.schema import Document

    # Build example Document chunks
    docs = [
        Document(
            page_content="Mark is a plumber who owns a business in Springfield and has five kids.",
            metadata={"chunk_id": "fact_chunk_0", "source": "fact_0"}
        ),
        Document(
            page_content="Gigio is Mark's first kid and loves computers.",
            metadata={"chunk_id": "fact_chunk_1", "source": "fact_1"}
        ),
        Document(
            page_content="Mark's plumbing business is called \"Mark's Pipes\" and has been serving Springfield for 10 years.",
            metadata={"chunk_id": "fact_chunk_2", "source": "fact_2"}
        ),
        Document(
            page_content="His wife Lisa helps manage the office and schedules appointments.",
            metadata={"chunk_id": "fact_chunk_3", "source": "fact_3"}
        ),
        Document(
            page_content="One of the other kids, Anna, enjoys painting and art.",
            metadata={"chunk_id": "fact_chunk_4", "source": "fact_4"}
        ),
        Document(
            page_content="The family has a golden retriever named Buddy who often rides along to job sites.",
            metadata={"chunk_id": "fact_chunk_5", "source": "fact_5"}
        ),
    ]

    # Example questions to ask
    questions = [
        "How many children does Mark have?",
        "What is the name of Mark's plumbing business?",
        "Who helps Mark manage the office?",
        "Which of Mark's children loves computers?",
        "What is Anna's favorite hobby?",
        "What's the name of the family dog?",
        "How long has Mark's Pipes been serving Springfield?",
        "Do we know anything about the children other than Anna e Gigio?",
        "Who's the youngest of Mark's children?",
    ]

    for question in questions:
        print(f"\n=== QUESTION ===\n{question}\n")
        answer, used = answer_question(question, docs)
        print("=== ANSWER ===")
        print(answer)
        print("\n" + "="*40)

if __name__ == "__main__":
    main()
