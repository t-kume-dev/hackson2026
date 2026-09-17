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

【T7担当へ】検索クエリのembeddingは自前で作らず、このファイルの embed_query() を
呼ぶこと。使用モデルは intfloat/multilingual-e5-base（768次元）で、e5系は
文書側に "passage: "、クエリ側に "query: " のプレフィックスが必須。付け忘れても
例外は出ず静かに精度が落ちるだけなので、両方ともこのファイルに閉じ込めてある。
embeddingはL2正規化済みなので、コサイン類似度は単なる内積で計算できる。
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
from t4_classify import CATEGORY_SEPARATOR

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

# 検索精度の実測比較（現実的なノイズ入り300件）で、旧モデル
# paraphrase-multilingual-MiniLM-L12-v2 は 83.3%、e5-base は 94.4% だった。
# MiniLMは正解が20位まで落ちるような外し方をするため e5-base を採用している。
# e5-large(2.2GB) は e5-base と同点だったので大きくする意味はない。
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768

# e5系モデルはプレフィックス必須。文書側とクエリ側で別の語を付ける。
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

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


def _sanitize_folder_name(part: str) -> str:
    """フォルダ名として使えない文字を除去する（階層の各パーツごとに個別に行う）"""
    return "".join(c for c in part if c not in r'\/:*?"<>|').strip() or "未分類"


def _move_to_category_folder(file_path: Path, category: str) -> Path:
    """カテゴリフォルダへファイルを移動する（CATEGORY_SEPARATOR区切りで入れ子フォルダを作る）"""
    # "アニメ・ゲーム／鬼滅の刃／グッズ写真" -> ["アニメ・ゲーム", "鬼滅の刃", "グッズ写真"]
    parts = [_sanitize_folder_name(p) for p in category.split(CATEGORY_SEPARATOR) if p.strip()]
    if not parts:
        parts = ["未分類"]

    destination_dir = ORGANIZED_ROOT
    for part in parts:
        destination_dir = destination_dir / part
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


def build_index_text(category: str, subtags: list[str], summary: str) -> str:
    """files.embedding の元になる文書テキストを組み立てる（プレフィックスは付けない）"""
    return " ".join([category, summary, " ".join(subtags)]).strip()


def _encode(text: str) -> np.ndarray:
    """L2正規化済みのembeddingを返す（コサイン類似度を内積で計算できるようにする）"""
    model = _get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embedding, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    検索クエリ（自然文）をembedding化する。T7の意味検索はこの関数を使うこと。

    files.embedding と同じモデル・同じ正規化で、e5系に必要な "query: "
    プレフィックスを内側で付ける。
    """
    return _encode(QUERY_PREFIX + query)


def _create_embedding(category: str, subtags: list[str], summary: str) -> np.ndarray:
    """カテゴリ・サブタグ・要約からembeddingを作る（文書側なので "passage: " を付ける）"""
    return _encode(PASSAGE_PREFIX + build_index_text(category, subtags, summary))


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
        return {"moved_path": file_path, "db_saved": False, "file_id": None}

    category = classification.get("category")
    subtags = classification.get("subtags", [])
    summary = classification.get("summary", "")

    if not category:
        logger.error(f"[T6] categoryがありません: {classification}")
        return {"moved_path": file_path, "db_saved": False, "file_id": None}

    original_path = str(source)
    filetype = _guess_filetype(file_path)
    file_hash = _calculate_hash(source)
    file_size = source.stat().st_size

    try:
        embedding = _create_embedding(category, subtags, summary)
    except Exception as e:
        logger.error(f"[T6] embedding生成に失敗しました: {e}")
        return {"moved_path": file_path, "db_saved": False, "file_id": None}

    try:
        moved_path = _move_to_category_folder(source, category)
    except OSError as e:
        logger.error(f"[T6] ファイル移動に失敗しました: {e}")
        return {"moved_path": file_path, "db_saved": False, "file_id": None}

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
        return {"moved_path": str(moved_path), "db_saved": False, "file_id": None}

    logger.info(f"[T6] 保存完了: {original_path} -> {moved_path}")
    return {"moved_path": str(moved_path), "db_saved": True, "file_id": file_id}


def recategorize(file_id: int, new_category: str) -> Optional[str]:
    """
    AIの分類をユーザーが訂正したときに、別カテゴリのフォルダへ移し直す。

    分類は必ず外れるので、手で直せる逃げ道がないとその場で詰む。
    移動前のパスは trash_log に残すので、Undoの対象にもなる。

    Returns:
        移動後のパス。失敗した場合はNone。
    """
    record = db.get_file(file_id)
    if record is None:
        logger.error(f"[T6] ファイルが見つかりません: id={file_id}")
        return None

    source = Path(record["path"])
    if not source.exists():
        logger.error(f"[T6] 実ファイルがありません: {source}")
        return None

    original_path = str(source)

    try:
        moved_path = _move_to_category_folder(source, new_category)
    except OSError as e:
        logger.error(f"[T6] 移動に失敗しました: {e}")
        return None

    try:
        db.update_file_location(file_id, str(moved_path), new_category)
        db.insert_trash_log(
            file_id=file_id,
            original_path=original_path,
            action_type="move",
        )
    except sqlite3.Error as e:
        logger.error(f"[T6] 訂正のDB更新に失敗しました: {e}")
        return None

    logger.info(f"[T6] カテゴリを訂正: {original_path} -> {moved_path}")
    return str(moved_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("T6 t6_filemanager.py")
    print(f"DB: {db.DB_PATH}（files / trash_log を担当。categoriesはT5の領域）")
    print(f"整理先ルート: {ORGANIZED_ROOT}")
    print(f"embeddingモデル: {EMBEDDING_MODEL_NAME}（{EMBEDDING_DIM}次元）")