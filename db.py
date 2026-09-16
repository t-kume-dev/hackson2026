"""
SQLite DB層（T5・T6が使う「外側の箱」）

テーブル定義は仕様書（おまかせ整理Bot_仕様書.md 8章）に準拠。
接続・スキーマ管理・BLOB変換はここに閉じ込め、T5/T6は下記の関数を呼ぶだけでよい。

- init_db()                                  : DBファイルとテーブルを用意する（初回起動時に呼ぶ）
- get_category_names() / get_category_embeddings() / insert_category()
  : T5（カテゴリ重複防止）が使う
- insert_file() / insert_trash_log()
  : T6（ファイル移動＋DB保存）が使う

embeddingはnumpy配列（float32）としてやり取りし、DBにはBLOBとして保存する。

このファイルの中身（テーブル定義・関数シグネチャ）は変更しないこと。
中の実装（SQL文など）を直す分にはT5/T6担当の裁量でOK。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

DB_PATH = Path(__file__).parent / "omakase.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    filename TEXT NOT NULL,
    filetype TEXT NOT NULL,
    category TEXT NOT NULL,
    subtags TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding BLOB NOT NULL,
    hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    created_at DATETIME NOT NULL,
    last_accessed_at DATETIME,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS trash_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id),
    original_path TEXT NOT NULL,
    action_type TEXT NOT NULL,
    acted_at DATETIME NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    """DB接続を1つ返す。呼び出し側で `with get_connection() as conn:` として使う想定。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """テーブルが無ければ作成する。アプリ起動時に一度呼べばよい（何度呼んでも安全）。"""
    with get_connection() as conn:
        conn.executescript(_SCHEMA)


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# T5: カテゴリ重複防止
# ---------------------------------------------------------------------------

def get_category_names() -> list[str]:
    """既存カテゴリ名の一覧（T4のプロンプトに埋め込む用）。"""
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM categories").fetchall()
    return [row["name"] for row in rows]


def get_category_embeddings() -> list[tuple[str, np.ndarray]]:
    """既存カテゴリの (名前, embedding) 一覧。コサイン類似度計算に使う。"""
    with get_connection() as conn:
        rows = conn.execute("SELECT name, embedding FROM categories").fetchall()
    return [(row["name"], _blob_to_embedding(row["embedding"])) for row in rows]


def insert_category(name: str, embedding: np.ndarray) -> None:
    """新規カテゴリを登録する。同名が既にあれば何もしない。"""
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO categories (name, embedding) VALUES (?, ?)",
            (name, _embedding_to_blob(embedding)),
        )


# ---------------------------------------------------------------------------
# T6: ファイル移動＋DB保存
# ---------------------------------------------------------------------------

def insert_file(
    path: str,
    filetype: str,
    category: str,
    subtags: list[str],
    summary: str,
    embedding: np.ndarray,
    file_hash: str,
    file_size: int,
) -> int:
    """
    移動後のファイル情報をfilesテーブルに保存する。

    Returns:
        挿入した行のid（trash_log等から参照する際に使う）
    """
    filename = Path(path).name
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO files
                (path, filename, filetype, category, subtags, summary,
                 embedding, hash, file_size, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                path, filename, filetype, category,
                json.dumps(subtags, ensure_ascii=False), summary,
                _embedding_to_blob(embedding), file_hash, file_size,
                datetime.now().isoformat(),
            ),
        )
        return cursor.lastrowid


def insert_trash_log(file_id: int, original_path: str, action_type: str) -> None:
    """ファイル移動・削除操作をtrash_logに記録する（Undo用）。action_type: 'move' | 'delete'"""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO trash_log (file_id, original_path, action_type, acted_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, original_path, action_type, datetime.now().isoformat()),
        )


if __name__ == "__main__":
    init_db()
    print(f"DB初期化完了: {DB_PATH}")
