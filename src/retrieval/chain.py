from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from ..utils.config import config
from ..utils.logger import logger
from typing import List


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("file_name", "unknown")
        parts.append("[" + str(i+1) + "] (" + source + ")" + chr(10) + doc.page_content)
    return (chr(10) * 2).join(parts)


def get_llm():
    return ChatOllama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL,
        temperature=0.1,
    )


def ask(question: str, docs: List[Document]) -> str:
    context = format_context(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer based on context only. If not in context, say so. Answer in the same language as the question."),
        ("human", "Context:" + chr(10) + "{context}" + chr(10) + chr(10) + "Question: {question}"),
    ])
    chain = prompt | get_llm() | StrOutputParser()
    logger.info("Sending to LLM: " + question[:50])
    return chain.invoke({"context": context, "question": question})
