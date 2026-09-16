"""
T5. カテゴリ重複防止（embeddingベース + db.py の categories テーブルを利用）

【前提】DBアクセスは全て db.py に集約されている。
    このファイルはSQLを直接書かない。DBファイルのパスも db.DB_PATH ただ1つが正。
    categories.embedding はNULL許容（embedding取得に失敗したカテゴリは名前だけ先に
    登録し、次回のresolve_category呼び出しでバックフィルする）。

入力: 新しいカテゴリ名（T4の出力の category）
出力: 最終的に使うカテゴリ名（文字列）。既存流用 or 新規のどちらか。

- 提案されたカテゴリ名をembedding化する（task_type=SEMANTIC_SIMILARITYを指定し、
  意味的な類似度判定に最適化されたembeddingを取得する）
- categoriesテーブルの既存カテゴリと比較する。
  embeddingが未計算の既存カテゴリは提案カテゴリと一緒にまとめてembedding化し、
  DBにキャッシュとして書き戻す（2回目以降はAPI呼び出し不要になる）
- 最も類似度が高い既存カテゴリが閾値以上なら、そのカテゴリ名に置き換える
- 閾値未満なら、新規カテゴリとしてembeddingと一緒にcategoriesテーブルへ登録して返す

このファイルの中身（embeddingモデルの選定・閾値等）はT5担当の裁量。
外から使われるのは resolve_category() と get_category_names() の2つ。

事前準備:
- pip install google-genai python-dotenv
- 環境変数 GEMINI_API_KEY にAPIキーを設定
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

import db

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"

# 類似度の判定しきい値（0.0〜1.0）。task_type=SEMANTIC_SIMILARITY採用に伴い0.93に調整。
# 高いほど厳しく（似ていないと同一視しない）
SIMILARITY_THRESHOLD = 0.93

# DBファイルのパスは db.py が唯一の正。ここでは別名を張るだけ。
DB_PATH = db.DB_PATH

# カテゴリ名/ベクトルの型（DBから来るものはnumpy配列、APIから来るものはfloatのリスト）
Vector = Union[np.ndarray, list]
DbPath = Optional[Union[str, Path]]


def _get_embeddings(texts: list[str]) -> Optional[list[list[float]]]:
    """
    複数のテキストをまとめて1回のAPI呼び出しでembedding化する。
    戻り値は texts と同じ順番・同じ件数のベクトルのリスト。失敗時はNone。
    """
    if not texts:
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("[T5] GEMINI_API_KEY が設定されていません")
        return None

    try:
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        embeddings = [list(e.values) for e in result.embeddings]
    except Exception as e:
        logger.error(f"[T5] embedding取得に失敗: {texts!r} ({e})")
        return None

    if len(embeddings) != len(texts):
        logger.error(
            f"[T5] embedding件数が入力件数と一致しません（入力{len(texts)}件 / 結果{len(embeddings)}件）"
        )
        return None

    return embeddings


def _cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """2つのベクトルのコサイン類似度を計算する。"""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def _register_new_category(
    category: str,
    embedding_vec: Optional[Vector],
    db_path: DbPath = None,
) -> bool:
    """
    新規カテゴリをembeddingと一緒にcategoriesテーブルへ登録する。既に同名があれば何もしない。
    DBエラーはここで握り潰す（カテゴリ登録に失敗してもパイプライン全体は止めない）。
    """
    try:
        db.insert_category(category, embedding_vec, db_path)
    except sqlite3.Error as e:
        logger.error(f"[T5] DB登録に失敗: {category} ({e})")
        return False

    logger.info(f"[T5] 新規カテゴリをDBに登録: {category}")
    return True


def _update_embedding(name: str, vector: Vector, db_path: DbPath = None) -> None:
    """既存カテゴリ行のembeddingカラムを埋める（初回アクセス時のキャッシュ書き戻し用）"""
    try:
        db.update_category_embedding(name, vector, db_path)
    except sqlite3.Error as e:
        logger.error(f"[T5] embeddingのキャッシュ書き戻しに失敗: {name} ({e})")
        return
    logger.info(f"[T5] 既存カテゴリのembeddingをキャッシュに書き戻し: {name}")


def get_category_names(db_path: DbPath = None) -> list[str]:
    """既存カテゴリ名だけの一覧を返す（T4がプロンプトに埋め込む既存カテゴリ一覧として使う）"""
    return db.get_category_names(db_path)


def resolve_category(
    proposed_category: str,
    db_path: DbPath = None,
    threshold: float = SIMILARITY_THRESHOLD,
    embed_fn: Callable[[list[str]], Optional[list[list[float]]]] = _get_embeddings,
    fetch_fn: Callable[[DbPath], list[tuple[str, Optional[np.ndarray]]]] = db.get_category_embeddings,
    register_fn: Callable[[str, Optional[Vector], DbPath], bool] = _register_new_category,
    update_fn: Callable[[str, Vector, DbPath], None] = _update_embedding,
) -> str:
    """
    T4が提案したカテゴリ名を、DB内の既存カテゴリ(embedding込み)と比較し、
    実際に使うカテゴリ名を決定する。新規の場合はembeddingと一緒にDBへ登録する。
    embeddingが未計算の既存カテゴリがあれば、この呼び出しの中でまとめて計算しキャッシュする。

    Args:
        proposed_category: T4が提案したカテゴリ名（入力）
        db_path: SQLiteファイルのパス。省略時は db.DB_PATH（通常は省略してよい）
        threshold: コサイン類似度の判定しきい値
        embed_fn / fetch_fn / register_fn / update_fn: テスト時に差し替え可能な依存関数

    Returns:
        最終的に使うカテゴリ名（文字列）。
        - 類似度がthreshold以上の既存カテゴリがあればそれを返す（DB登録はしない）
        - 無ければproposed_categoryをembeddingと一緒にDBへ登録してそのまま返す
        - proposed_categoryが空/Noneの場合は登録せず "未分類" を返す
    """
    if not proposed_category or not proposed_category.strip():
        logger.warning("[T5] proposed_categoryが空です。'未分類'として扱います")
        return "未分類"

    proposed_category = proposed_category.strip()

    existing = fetch_fn(db_path)
    missing_names = [name for name, vec in existing if vec is None]

    # 提案カテゴリ + embedding未計算の既存カテゴリをまとめて1回で問い合わせる
    embeddings = embed_fn([proposed_category] + missing_names)

    if embeddings is None:
        logger.error(f"[T5] embedding取得失敗のため新規カテゴリとして扱う: {proposed_category}")
        register_fn(proposed_category, None, db_path)
        return proposed_category

    proposed_vec = embeddings[0]
    missing_vecs = embeddings[1:]

    # 既存カテゴリのうち未計算だったものをDBに書き戻す（次回以降キャッシュが効く）
    for name, vec in zip(missing_names, missing_vecs):
        update_fn(name, vec, db_path)

    vec_by_name: dict[str, Vector] = {
        name: vec for name, vec in existing if vec is not None
    }
    vec_by_name.update(zip(missing_names, missing_vecs))

    if not vec_by_name:
        logger.info(f"[T5] 既存カテゴリが無いため新規採用: {proposed_category}")
        register_fn(proposed_category, proposed_vec, db_path)
        return proposed_category

    best_match: Optional[str] = None
    best_score = -1.0

    for name, vec in vec_by_name.items():
        score = _cosine_similarity(proposed_vec, vec)
        logger.debug(f"[T5] 類似度: '{proposed_category}' vs '{name}' = {score:.4f}")
        if score > best_score:
            best_score = score
            best_match = name

    if best_match is not None and best_score >= threshold:
        logger.info(
            f"[T5] 類似度{best_score:.4f}が閾値{threshold}以上のため既存カテゴリを流用: "
            f"'{proposed_category}' -> '{best_match}'"
        )
        return best_match

    logger.info(
        f"[T5] 最高類似度{best_score:.4f}が閾値{threshold}未満のため新規採用: {proposed_category}"
    )
    register_fn(proposed_category, proposed_vec, db_path)
    return proposed_category


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    tests = [
        "領収書",        # 既存と完全一致 → 初回はembedding未計算なのでバックフィルされる
        "請求書",        # 領収書とは意味が違うので新規になるか要確認
        "契約書関連書類",  # 既存の「契約書」に近いか
        "旅先での写真",    # 既存の「旅行の写真」に近いか
    ]

    for proposed in tests:
        result = resolve_category(proposed)
        print(f"[T5] '{proposed}' -> '{result}'")

    print(f"\n[T5] 現在の既存カテゴリ一覧: {get_category_names()}")
