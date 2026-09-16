"""
T6. ファイル移動＋メタデータ保存

外から使われる関数は2つ:
  - get_existing_categories() -> list[str]
      DBに登録済みの既存カテゴリ名一覧を返す。
      T4のプロンプト作成・T5の重複判定にそのまま渡せる形（文字列のリスト）。

  - save_result(file_path: str, classification: dict) -> dict
      ファイルを最終的な置き場所（カテゴリ名フォルダ）へ移動し、分類結果をDBに保存する。
      Args:
          file_path: 元のファイルパス（T1〜T3で扱っていたもの。移動前）
          classification: {"category": str, "subtags": list[str], "summary": str}
                           （T5で確定した最終カテゴリを反映済みのもの）
      Returns:
          {"moved_path": str, "db_saved": bool}

【T5担当からの申し送り事項の反映】
- DBファイルは categories.db（T5と共通のSQLiteファイル）を既定値として使う
- categories テーブル（id, name UNIQUE）は T5 側が作成・書き込みを行うため、
  T6側では CREATE TABLE も INSERT も行わない。get_existing_categories() は
  「SELECT name FROM categories」で読み出すだけ。
- 本番でDBファイルパスを categories.db 以外にする場合は、T5の t5_dedupe.py の
  DB_PATH も合わせて変更する必要がある（このファイルの DB_PATH と揃えること）。

【Epic4 受け入れ条件との対応】
- カテゴリ名のフォルダが存在しない場合は自動作成される     -> _move_to_category_folder()
- ファイルが対象フォルダへ移動される                        -> _move_to_category_folder()
- 移動前の元パスが trash_log に記録される                    -> save_result() 内でINSERT
- files テーブルに全メタデータが漏れなく保存される            -> save_result() 内でINSERT
    (path, filetype, category, subtags, summary, embedding, hash, file_size, created_at)

【実装上の注意（申し送り事項）】
- save_result() の契約には filetype が含まれていないため、T2と同じ拡張子マップを使って
  T6内で再判定している。判定がずれる場合は、main.py側でfile_typeも一緒に渡す契約に
  変更することを検討してください。
- files.embedding は、元のテキスト内容を受け取らない契約のため
  category + summary + subtags を結合した文字列から生成している
  （categoriesテーブルのembeddingとは無関係。files検索用）。

事前準備:
  pip install sentence-transformers numpy
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# DBファイルの場所。T5の t5_dedupe.py の DB_PATH と揃えること（既定: categories.db を共有）。
DB_PATH = os.environ.get("APP_DB_PATH", "categories.db")

# 分類後のファイルを置くルートフォルダ（この下に category 名のフォルダが作られる）
ORGANIZED_ROOT = os.environ.get("ORGANIZED_ROOT", "organized")

# 日本語カテゴリ・要約を扱うため多言語対応モデルを使用
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# T2と同じ拡張子->filetype対応表（T6単体でもfiletypeを復元できるようにするため）
EXTENSION_TYPE_MAP = {
    ".txt": "text", ".md": "text", ".csv": "text",
    ".pdf": "pdf",
    ".png": "photo", ".jpg": "photo", ".jpeg": "photo", ".gif": "photo",
}

_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """埋め込みモデルを取得する（初回のみロードしてキャッシュ）"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _guess_filetype(file_path: str) -> str:
    """拡張子からfiletypeを推定する（T2と同じロジック）"""
    ext = Path(file_path).suffix.lower()
    return EXTENSION_TYPE_MAP.get(ext, "unknown")


def _sha256_of_file(path: Path) -> str:
    """重複ファイル検出用のハッシュ値を計算する"""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _init_db(conn: sqlite3.Connection) -> None:
    """
    files / trash_log テーブルが無ければ作成する（T6が責任を持つテーブルのみ）。
    categories テーブルはT5側が作成するため、ここでは一切触らない。
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            filename TEXT NOT NULL,
            filetype TEXT NOT NULL,
            category TEXT NOT NULL,
            subtags TEXT,
            summary TEXT,
            embedding BLOB,
            hash TEXT,
            file_size INTEGER,
            created_at TEXT NOT NULL,
            last_accessed_at TEXT,
            status TEXT NOT NULL DEFAULT 'active'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trash_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER,
            original_path TEXT NOT NULL,
            action_type TEXT NOT NULL,
            acted_at TEXT NOT NULL,
            FOREIGN KEY (file_id) REFERENCES files(id)
        )
        """
    )
    conn.commit()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    _init_db(conn)
    return conn


def get_existing_categories() -> List[str]:
    """
    DBに登録済みの既存カテゴリ名一覧を返す。
    T5が管理する categories テーブル（id, name）から読み出すだけで、
    T6側でテーブル作成や書き込みは行わない。

    categories テーブルがまだ存在しない場合（T5がまだ一度も実行されていない等）は
    空リストを返す。
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            rows = conn.execute("SELECT name FROM categories ORDER BY name").fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        # categories テーブルがまだ無い場合など
        logger.warning(f"[T6] categoriesテーブルが見つかりません（未作成の可能性）: {e}")
        return []
    except sqlite3.Error as e:
        logger.error(f"[T6] 既存カテゴリの取得に失敗: {e}")
        return []


def _move_to_category_folder(file_path: Path, category: str) -> Path:
    """カテゴリ名のフォルダへファイルを移動する。フォルダが無ければ自動作成する。"""
    # OSのフォルダ名として使えない文字を簡易的に除去
    safe_category = "".join(c for c in category if c not in '\\/:*?"<>|').strip() or "未分類"
    dest_dir = Path(ORGANIZED_ROOT) / safe_category
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / file_path.name
    # 同名ファイルがすでにある場合は連番を付けて上書きを防ぐ
    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
        counter += 1

    shutil.move(str(file_path), str(dest_path))
    return dest_path


def save_result(file_path: str, classification: dict) -> dict:
    """
    ファイルを最終的な置き場所（カテゴリ名フォルダ）へ移動し、
    分類結果を files / trash_log テーブルに保存する。
    （categories テーブルへの書き込みは行わない。T5側の責務）

    Args:
        file_path: 元のファイルパス（T1〜T3で扱っていたもの。移動前）
        classification: {"category": str, "subtags": list[str], "summary": str}
                         （T5で確定した最終カテゴリを反映済みのもの）

    Returns:
        {"moved_path": str, "db_saved": bool}
        失敗時: moved_pathは可能な範囲で正確な値（移動前 or 移動後）、db_saved=False
    """
    src_path = Path(file_path)

    if not src_path.exists():
        logger.error(f"[T6] ファイルが存在しません: {file_path}")
        return {"moved_path": file_path, "db_saved": False}

    category = classification.get("category")
    subtags = classification.get("subtags", [])
    summary = classification.get("summary", "")

    if not category:
        logger.error(f"[T6] classificationにcategoryがありません: {classification}")
        return {"moved_path": file_path, "db_saved": False}

    # 移動するとパスが変わってしまう情報は、移動前に取得しておく
    filetype = _guess_filetype(str(src_path))
    file_hash = _sha256_of_file(src_path)
    file_size = src_path.stat().st_size
    original_path = str(src_path)
    filename = src_path.name

    try:
        moved_path = _move_to_category_folder(src_path, category)
    except OSError as e:
        logger.error(f"[T6] ファイル移動に失敗: {file_path} ({e})")
        return {"moved_path": file_path, "db_saved": False}

    # files検索用embedding: カテゴリ・要約・サブタグをまとめてベクトル化
    embed_source = " ".join([category, summary, " ".join(subtags)]).strip()
    try:
        embedding_bytes = _get_embedding_model().encode(embed_source).astype(np.float32).tobytes()
    except Exception as e:
        logger.warning(f"[T6] embedding生成に失敗（DB保存は続行、embeddingはNoneにする）: {e}")
        embedding_bytes = None

    created_at = datetime.now(timezone.utc).isoformat()

    try:
        conn = _get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO files (
                    path, filename, filetype, category, subtags, summary,
                    embedding, hash, file_size, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                """,
                (
                    str(moved_path), filename, filetype, category,
                    json.dumps(subtags, ensure_ascii=False), summary,
                    embedding_bytes, file_hash, file_size, created_at,
                ),
            )
            file_id = cursor.lastrowid

            # 移動前の元パスをUndo用に記録
            conn.execute(
                """
                INSERT INTO trash_log (file_id, original_path, action_type, acted_at)
                VALUES (?, ?, 'move', ?)
                """,
                (file_id, original_path, created_at),
            )

            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.error(f"[T6] DB保存に失敗: {file_path} ({e})")
        # ファイル自体はすでに移動済みなので、moved_pathは移動後のものを返す
        return {"moved_path": str(moved_path), "db_saved": False}

    logger.info(f"[T6] 保存完了: {original_path} -> {moved_path} (category={category})")
    return {"moved_path": str(moved_path), "db_saved": True}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("既存カテゴリ:", get_existing_categories())

    # 動作確認用のサンプル（実際に試す場合はテスト用のファイルを用意してパスを書き換える）
    # result = save_result(
    #     "C:/sample/memo.txt",
    #     {"category": "レポート", "subtags": ["ハッカソン"], "summary": "テスト用の要約"},
    # )
    # print(result)