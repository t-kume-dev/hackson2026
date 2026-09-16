"""
T6. ファイル移動＋メタデータ保存

【変更点】db.py は使わない。T5(t5_dedupe.py)が実際に使っている categories.db と
同じファイルに、T6が自分で files / trash_log テーブルを作って書き込む。
categories テーブルには一切触れない（T5の領域）。

db.py / t5_dedupe.py はどちらも変更しない前提。
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# T5(t5_dedupe.py)のDB_PATHデフォルト値と同じファイルを指す。
# T5側のDB_PATHを変更する場合は、必ずこちらも同じ値に合わせること。
DB_PATH = "categories.db"

ORGANIZED_ROOT = Path("organized")

EXTENSION_TYPE_MAP = {
    ".txt": "text",
    ".md": "text",
    ".csv": "text",
    ".pdf": "pdf",
    ".png": "photo",
    ".jpg": "photo",
    ".jpeg": "photo",
    ".gif": "photo",
}

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model: Optional[SentenceTransformer] = None

# T6が責任を持つテーブルだけを作る。categoriesテーブルには一切触れない。
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

CREATE TABLE IF NOT EXISTS trash_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id),
    original_path TEXT NOT NULL,
    action_type TEXT NOT NULL,
    acted_at DATETIME NOT NULL
);
"""


def _get_connection() -> sqlite3.Connection:
    """
    categories.db への接続を返す。files/trash_logテーブルが無ければここで作成する
    （毎回チェックするだけなので、既にあれば何もしない＝安全に何度呼んでもよい）。
    categoriesテーブルには一切関与しない（T5が管理するため）。
    """
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def _get_embedding_model() -> SentenceTransformer:
    """embeddingモデルを初回だけ読み込む"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _guess_filetype(file_path: str) -> str:
    """拡張子からfiletypeを判定する"""
    extension = Path(file_path).suffix.lower()
    return EXTENSION_TYPE_MAP.get(extension, "unknown")


def _calculate_hash(file_path: Path) -> str:
    """ファイルのSHA-256ハッシュを計算する"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _move_to_category_folder(file_path: Path, category: str) -> Path:
    """カテゴリフォルダへファイルを移動する"""
    safe_category = "".join(
        c for c in category
        if c not in '\\/:*?"<>|'
    ).strip() or "未分類"

    destination_dir = ORGANIZED_ROOT / safe_category
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / file_path.name
    counter = 1

    while destination.exists():
        destination = (
            destination_dir
            / f"{file_path.stem}_{counter}{file_path.suffix}"
        )
        counter += 1

    file_path.rename(destination)
    return destination


def _create_embedding(category: str, subtags: list[str], summary: str) -> np.ndarray:
    """カテゴリ・サブタグ・要約からembeddingを作る"""
    text = " ".join([category, summary, " ".join(subtags)]).strip()
    model = _get_embedding_model()
    embedding = model.encode(text)
    return np.asarray(embedding, dtype=np.float32)


def _insert_file(
    conn: sqlite3.Connection,
    path: str,
    filetype: str,
    category: str,
    subtags: list[str],
    summary: str,
    embedding: np.ndarray,
    file_hash: str,
    file_size: int,
) -> int:
    filename = Path(path).name
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
            np.asarray(embedding, dtype=np.float32).tobytes(), file_hash, file_size,
            datetime.now().isoformat(),
        ),
    )
    return cursor.lastrowid


def _insert_trash_log(conn: sqlite3.Connection, file_id: int, original_path: str, action_type: str) -> None:
    conn.execute(
        """
        INSERT INTO trash_log (file_id, original_path, action_type, acted_at)
        VALUES (?, ?, ?, ?)
        """,
        (file_id, original_path, action_type, datetime.now().isoformat()),
    )


def save_result(file_path: str, classification: dict) -> dict:
    """
    ファイルをカテゴリフォルダへ移動し、categories.db の files/trash_log に保存する。

    classification:
        {"category": str, "subtags": list[str], "summary": str}

    Returns:
        {"moved_path": str, "db_saved": bool}
    """
    source = Path(file_path)

    if not source.exists():
        logger.error(f"[T6] ファイルが存在しません: {file_path}")
        return {"moved_path": file_path, "db_saved": False}

    category = classification.get("category")
    subtags = classification.get("subtags", [])
    summary = classification.get("summary", "")

    if not category:
        logger.error(f"[T6] categoryがありません: {classification}")
        return {"moved_path": file_path, "db_saved": False}

    original_path = str(source)
    filetype = _guess_filetype(file_path)
    file_hash = _calculate_hash(source)
    file_size = source.stat().st_size

    try:
        embedding = _create_embedding(category, subtags, summary)
    except Exception as e:
        logger.error(f"[T6] embedding生成に失敗しました: {e}")
        return {"moved_path": file_path, "db_saved": False}

    try:
        moved_path = _move_to_category_folder(source, category)
    except OSError as e:
        logger.error(f"[T6] ファイル移動に失敗しました: {e}")
        return {"moved_path": file_path, "db_saved": False}

    try:
        conn = _get_connection()
        try:
            file_id = _insert_file(
                conn,
                path=str(moved_path),
                filetype=filetype,
                category=category,
                subtags=subtags,
                summary=summary,
                embedding=embedding,
                file_hash=file_hash,
                file_size=file_size,
            )
            _insert_trash_log(conn, file_id=file_id, original_path=original_path, action_type="move")
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.error(f"[T6] DB保存に失敗しました: {e}")
        return {"moved_path": str(moved_path), "db_saved": False}

    logger.info(f"[T6] 保存完了: {original_path} -> {moved_path}")
    return {"moved_path": str(moved_path), "db_saved": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("T6 t6_filemanager.py")
    print(f"DB: {DB_PATH}（categoriesテーブル以外はT6が自分で管理）")

    # 動作確認用サンプル（実際に試す場合はテスト用ファイルを用意してパスを書き換える）
    # result = save_result(
    #     "C:/sample/memo.txt",
    #     {"category": "レポート", "subtags": ["ハッカソン"], "summary": "テスト用の要約"},
    # )
    # print(result)