
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # reads OPENROUTER_API_KEY, MODEL, etc. from a .env file

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, autodetect_encoding=True),
    ".md": lambda p: TextLoader(p, autodetect_encoding=True),
    ".docx": Docx2txtLoader,
}

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question as a standalone search query, using the "
     "chat history to resolve pronouns and references. Return only the query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using ONLY the context excerpts below, taken from the "
     "user's uploaded documents. If the context does not contain the answer, say "
     "you couldn't find it in the documents; do not guess or use outside knowledge. "
     "Be concise, and mention the source file names you relied on.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def _to_messages(history: list[tuple[str, str]]) -> list[BaseMessage]:
    return [HumanMessage(c) if role == "user" else AIMessage(c) for role, c in history]


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        loc = f"{d.metadata.get('source_name')}" + (f", p.{page + 1}" if isinstance(page, int) else "")
        parts.append(f"[{loc}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGAssistant:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        chat_model: str | None = None,
        embedding_model: str | None = None,
        k: int = 4,
        llm=None,
        embeddings=None,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key and not (llm and embeddings):
            raise RuntimeError("OPENROUTER_API_KEY is not set (add it to your .env file).")
        base_url = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)

        self.llm = llm or ChatOpenAI(
            model=chat_model or os.getenv("MODEL", "openai/gpt-5.6-luna"),
            api_key=api_key,
            base_url=base_url,
        )
        self._embeddings = embeddings or OpenAIEmbeddings(
            model=embedding_model or os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            api_key=api_key,
            base_url=base_url,
            check_embedding_ctx_length=False,  # send plain text, not OpenAI-specific token IDs
        )
        self._persist_dir = persist_dir
        self._k = k
        self._connect()
        parser = StrOutputParser()
        self._condense = CONDENSE_PROMPT | self.llm | parser
        self._answer = ANSWER_PROMPT | self.llm | parser
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # ---- ingestion -------------------------------------------------------
    def add_file(self, path: str | Path, display_name: str | None = None) -> int:
        """Load, chunk, embed and store one file. Re-adding the same file is idempotent."""
        path = Path(path)
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            raise ValueError(f"Unsupported file type: {path.suffix} (use {', '.join(LOADERS)})")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        pages = loader_cls(str(path)).load()
        chunks = self._splitter.split_documents(pages)
        for d in chunks:
            d.metadata["source_name"] = display_name or path.name
        ids = [f"{digest}-{i}" for i in range(len(chunks))]  # stable ids -> upsert, no duplicates
        if chunks:
            self.store.add_documents(chunks, ids=ids)
        return len(chunks)

    def count(self) -> int:
        return self.store._collection.count()

    def _connect(self) -> None:
        self.store = Chroma(
            collection_name="documents",
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )
        self.retriever = self.store.as_retriever(
            search_type="mmr", search_kwargs={"k": self._k, "fetch_k": self._k * 5}
        )

    def clear(self) -> None:
        self.store.delete_collection()
        self._connect()

    # ---- question answering ---------------------------------------------
    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> dict:
        """history is a list of ("user" | "assistant", text). Returns {answer, sources}."""
        messages = _to_messages(history or [])
        query = self._condense.invoke({"question": question, "history": messages}) if messages else question

        docs = self.retriever.invoke(query)
        if not docs:
            return {"answer": "I don't have any documents to search yet. Please upload some first.", "sources": []}

        answer = self._answer.invoke(
            {"context": _format_docs(docs), "question": question, "history": messages}
        )
        seen, sources = set(), []
        for d in docs:
            page = d.metadata.get("page")
            key = (d.metadata.get("source_name"), page)
            if key not in seen:
                seen.add(key)
                label = d.metadata.get("source_name", "unknown") + (f" (p.{page + 1})" if isinstance(page, int) else "")
                sources.append({"label": label, "snippet": d.page_content[:300]})
        return {"answer": answer, "sources": sources}


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your documents.")
    ap.add_argument("files", nargs="*", help="PDF / DOCX / TXT / MD files to ingest")
    ap.add_argument("--db", default="./chroma_db", help="vector store directory")
    args = ap.parse_args()

    bot = RAGAssistant(persist_dir=args.db)
    for f in args.files:
        print(f"Ingested {bot.add_file(f)} chunks from {f}")
    print(f"{bot.count()} chunks indexed. Ask a question (Ctrl-C to quit).\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            q = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        res = bot.ask(q, history)
        print(f"\nassistant> {res['answer']}")
        if res["sources"]:
            print("sources: " + "; ".join(s["label"] for s in res["sources"]))
        print()
        history += [("user", q), ("assistant", res["answer"])]


if __name__ == "__main__":
    main()