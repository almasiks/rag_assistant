import sqlite3
import aiosqlite
from pathlib import Path
from ..utils.logger import logger

DB_PATH = Path("data/logs.db")


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "CREATE TABLE IF NOT EXISTS query_logs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "question TEXT NOT NULL,"
            "answer TEXT NOT NULL,"
            "sources TEXT,"
            "latency_ms INTEGER,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        await db.commit()
    logger.info("SQLite DB initialized at " + str(DB_PATH))


async def log_query(question: str, answer: str, sources: list, latency_ms: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO query_logs (question, answer, sources, latency_ms) VALUES (?, ?, ?, ?)",
            (question, answer, str(sources), latency_ms)
        )
        await db.commit()


def get_stats():
    if not DB_PATH.exists():
        return {"total": 0, "avg_latency": 0, "recent": []}
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
    avg_lat = conn.execute("SELECT AVG(latency_ms) FROM query_logs").fetchone()[0]
    recent = conn.execute(
        "SELECT question, answer, sources, latency_ms, created_at FROM query_logs ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return {
        "total": total or 0,
        "avg_latency": round(avg_lat or 0),
        "recent": [{"question": r[0], "answer": r[1], "sources": r[2], "latency_ms": r[3], "created_at": r[4]} for r in recent],
    }
