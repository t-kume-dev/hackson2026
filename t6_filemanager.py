"""
T6. ファイル移動＋メタデータ保存

【前提】DBアクセスは全て db.py に集約されている。
    このファイルはSQLを直接書かない。DBファイルのパスも db.DB_PATH ただ1つが正。
    担当テーブルは files / trash_log（categoriesはT5の領域なので触らない）。

入力: ファイルパス + 分類結果一式（category / subtags / summary）
出力: {"moved_path": 移動後のパス, "db_saved": DB保存に成功したか}

なお files.embedding はここで sentence-transformers を使って作る。
T5が categories.embedding に入れるGemini embeddingとは次元もモデルも別物なので、
T7の意味検索で files.embedding と比較する際は必ずこのファイルと同じモデル
（EMBEDDING_MODEL_NAME）でクエリをembedding化すること。
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import db

logger = logging.getLogger(__name__)

# 整理後のファイルを置くルート。実行時のカレントディレクトリに依存しないよう
# このファイルの場所を基準にする。
ORGANIZED_ROOT = Path(__file__).parent / "organized"

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
        if c not in r'\/:*?"<>|'
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


def save_result(file_path: str, classification: dict) -> dict:
    """
    ファイルをカテゴリフォルダへ移動し、files/trash_logテーブルに保存する。

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

    # filesへのINSERTとtrash_logへのINSERTは1つのトランザクションにまとめる
    # （片方だけ残ってUndoできなくなるのを防ぐ）
    try:
        conn = db.get_connection()
        try:
            file_id = db.insert_file(
                path=str(moved_path),
                filetype=filetype,
                category=category,
                subtags=subtags,
                summary=summary,
                embedding=embedding,
                file_hash=file_hash,
                file_size=file_size,
                conn=conn,
            )
            db.insert_trash_log(
                file_id=file_id,
                original_path=original_path,
                action_type="move",
                conn=conn,
            )
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
    print(f"DB: {db.DB_PATH}（files / trash_log を担当。categoriesはT5の領域）")
    print(f"整理先ルート: {ORGANIZED_ROOT}")
