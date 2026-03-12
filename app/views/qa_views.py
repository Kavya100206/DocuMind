"""
QA Views (Response Schemas for Answer Generation)

These Pydantic schemas define the shape of the /api/ask response.
"""

from typing import List, Optional
from pydantic import BaseModel


class Citation(BaseModel):
    """
    One source citation — where a piece of the answer came from.

    document_name: Filename of the source document
    page_number:   Page where the relevant text was found
    text_snippet:  First 150 chars of the source chunk (preview)
    """
    document_name: str
    page_number: int
    text_snippet: str


class QAResponse(BaseModel):
    """
    Full response for a question-answering request.

    Fields:
      question:        The original question (echoed back)
      answer:          The LLM-generated answer (grounded in documents)
      citations:       List of sources used to generate the answer
      confidence:      0.0–1.0 score based on retrieval similarity
      has_answer:      False if LLM said it doesn't have enough info
      model_used:      Which Groq model generated the answer
      rewritten_query: The keyword-rich query used for retrieval (if rewriting occurred)
    """
    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    has_answer: bool
    model_used: str
    rewritten_query: Optional[str] = None
