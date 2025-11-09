from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

DB_PATH = Path("live_signals.db")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS control (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()


def load_map() -> Dict[str, str]:
    if not DB_PATH.exists():
        return {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            _ensure_table(conn)
            cur = conn.execute("SELECT key, value FROM control")
            return {k: (v if v is not None else "") for k, v in cur.fetchall()}
    except Exception:
        return {}


def set_items(items: Dict[str, Any]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_table(conn)
        for k, v in items.items():
            conn.execute(
                "INSERT INTO control(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(k), "" if v is None else str(v)),
            )
        conn.commit()


def get_bool(d: Dict[str, str], key: str, default: bool) -> bool:
    v = d.get(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def get_int(d: Dict[str, str], key: str, default: int) -> int:
    v = d.get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def get_list(d: Dict[str, str], key: str, default: list[str]) -> list[str]:
    v = d.get(key)
    if v is None or str(v).strip() == "":
        return list(default)
    raw = str(v)
    out: list[str] = []
    for tok in raw.replace(";", ",").split(","):
        s = tok.strip()
        if s:
            out.append(s)
    return out or list(default)


