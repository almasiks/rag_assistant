import streamlit as st
import requests
import sqlite3
import pandas as pd
from pathlib import Path
import tempfile
import os
import subprocess
import sys

API_URL = "http://localhost:8000"

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

tab1, tab2, tab3 = st.tabs(["💬 Чат", "📄 Загрузка документов", "📊 Статистика"])

# --- TAB 1: CHAT ---
with tab1:
    st.header("RAG Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Источники"):
                    for s in msg["sources"]:
                        st.write("- " + s.get("file_name", "unknown") + " (chunk " + str(s.get("chunk_index", "?")) + ")")
            if msg.get("latency_ms"):
                st.caption("Latency: " + str(msg["latency_ms"]) + "ms")

    question = st.chat_input("Задай вопрос по документам...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    resp = requests.post(
                        API_URL + "/query",
                        json={"question": question},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.write(data["answer"])
                        with st.expander("Источники"):
                            for s in data["sources"]:
                                st.write("- " + s.get("file_name", "unknown") + " (chunk " + str(s.get("chunk_index", "?")) + ")")
                        st.caption("Latency: " + str(data["latency_ms"]) + "ms")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["answer"],
                            "sources": data["sources"],
                            "latency_ms": data["latency_ms"],
                        })
                    else:
                        st.error("Ошибка API: " + str(resp.status_code))
                except Exception as e:
                    st.error("Не удалось подключиться к API: " + str(e))

# --- TAB 2: UPLOAD ---
with tab2:
    st.header("Загрузка документов")
    st.info("Загрузи PDF или TXT файлы — они будут добавлены в базу знаний")

    uploaded = st.file_uploader("Выбери файлы", type=["pdf", "txt", "md"], accept_multiple_files=True)

    if uploaded and st.button("Загрузить и индексировать", type="primary"):
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)

        for f in uploaded:
            dest = docs_dir / f.name
            dest.write_bytes(f.read())
            st.success("Сохранён: " + f.name)

        with st.spinner("Индексирование..."):
            result = subprocess.run(
                [sys.executable, "-m", "src.ingestion.ingest_hybrid", "docs/"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                st.success("Индексирование завершено!")
                st.code(result.stdout)
            else:
                st.error("Ошибка индексирования")
                st.code(result.stderr)

# --- TAB 3: DASHBOARD ---
with tab3:
    st.header("Статистика запросов")

    if st.button("Обновить"):
        st.rerun()

    db_path = Path("data/logs.db")
    if not db_path.exists():
        st.warning("Нет данных. Задай несколько вопросов в чате.")
    else:
        conn = sqlite3.connect(db_path)

        total = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
        avg_lat = conn.execute("SELECT AVG(latency_ms) FROM query_logs").fetchone()[0]

        col1, col2 = st.columns(2)
        col1.metric("Всего запросов", total)
        col2.metric("Средняя latency", str(round(avg_lat or 0)) + " ms")

        st.subheader("Последние запросы")
        rows = conn.execute(
            "SELECT question, latency_ms, created_at FROM query_logs ORDER BY created_at DESC LIMIT 20"
        ).fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["Вопрос", "Latency (ms)", "Время"])
            st.dataframe(df, use_container_width=True)

        st.subheader("Latency по запросам")
        lat_rows = conn.execute(
            "SELECT created_at, latency_ms FROM query_logs ORDER BY created_at"
        ).fetchall()
        if lat_rows:
            df2 = pd.DataFrame(lat_rows, columns=["Время", "Latency (ms)"])
            st.line_chart(df2.set_index("Время"))

        conn.close()
