import tempfile
from pathlib import Path

import streamlit as st

from assistant import LOADERS, RAGAssistant

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Ask your documents")


@st.cache_resource
def get_bot() -> RAGAssistant:
    return RAGAssistant()


bot = get_bot()
st.session_state.setdefault("history", [])
st.session_state.setdefault("ingested", set())

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload files", type=[e.lstrip(".") for e in LOADERS], accept_multiple_files=True
    )
    for up in uploads or []:
        key = (up.name, up.size)
        if key in st.session_state.ingested:
            continue
        with st.spinner(f"Indexing {up.name}…"):
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / up.name
                p.write_bytes(up.getvalue())
                n = bot.add_file(p, display_name=up.name)
        st.session_state.ingested.add(key)
        st.success(f"{up.name}: {n} chunks")
    st.caption(f"{bot.count()} chunks indexed")
    if st.button("Clear index & chat"):
        bot.clear()
        st.session_state.history, st.session_state.ingested = [], set()
        st.rerun()

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if q := st.chat_input("Ask a question about your documents"):
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            res = bot.ask(q, st.session_state.history)
        st.write(res["answer"])
        for s in res["sources"]:
            with st.expander(f"📎 {s['label']}"):
                st.caption(s["snippet"] + "…")
    st.session_state.history += [("user", q), ("assistant", res["answer"])]