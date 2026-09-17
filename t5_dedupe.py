"""
T5. カテゴリ重複防止（表記の正規化 + db.py の categories テーブルを利用）

【前提】DBアクセスは全て db.py に集約されている。
    このファイルはSQLを直接書かない。DBファイルのパスも db.DB_PATH ただ1つが正。

入力: 新しいカテゴリ名（T4の出力の category。CATEGORY_SEPARATOR区切りのフルパス）
出力: 最終的に使うカテゴリ名（文字列）。既存流用 or 新規のどちらか。

【方針】意味の近さの判定は T4（GPT）に任せる。
    T4のプロンプトには既存カテゴリのツリーを渡し、同じ意味なら既存の表記を使わせている。
    以前はここでembeddingの類似度を見ていたが、「雇用契約書」と「賃貸借契約書」のように
    別物でも類似度が高く出て、正しい分類を誤って統合してしまっていた。
    ここでは全角/半角・大文字小文字・空白の違いだけを吸収し、階層を1段ずつ既存の表記に揃える。

外から使われるのは resolve_category() と get_category_names() の2つ。
"""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Optional, Union

import db
from t4_classify import CATEGORY_SEPARATOR

logger = logging.getLogger(__name__)

# DBファイルのパスは db.py が唯一の正。ここでは別名を張るだけ。
DB_PATH = db.DB_PATH

DbPath = Optional[Union[str, Path]]


def get_category_names(db_path: DbPath = None) -> list[str]:
    """T4のプロンプトに埋め込む用の既存カテゴリ名一覧"""
    return db.get_category_names(db_path)


def _normalize(name: str) -> str:
    """表記の揺れだけを吸収した比較用キー（Ｐｙｔｈｏｎ / python / Py thon を同じとみなす）"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).casefold()


def _register_new_category(category: str, db_path: DbPath = None) -> None:
    """
    新規カテゴリをcategoriesテーブルへ登録する。既に同名があれば何もしない。
    DBエラーはここで握り潰す（カテゴリ登録に失敗してもパイプライン全体は止めない）。
    """
    try:
        db.insert_category(category, None, db_path)
    except sqlite3.Error as e:
        logger.error(f"[T5] DB登録に失敗: {category} ({e})")
        return
    logger.info(f"[T5] 新規カテゴリをDBに登録: {category}")


def resolve_category(proposed_category: str, db_path: DbPath = None) -> str:
    """
    T4が提案したカテゴリパス（例: "アニメ・ゲーム／鬼滅の刃／グッズ写真"）を、
    階層を1段ずつ降りながら「同じ親を持つ既存カテゴリ（兄弟）」と表記で照合し、
    最終的に使うフルパスを決定する。新規の階層はDBへ登録する。

    Returns:
        最終的に使うカテゴリのフルパス文字列。
        - proposed_categoryが空/Noneの場合は登録せず "未分類" を返す
        - 各階層ごとに、表記が一致する既存カテゴリがあればその表記に揃える
        - 無ければその階層を新規登録する（親は揃えた後の表記を使う）
    """
    parts = [p.strip() for p in (proposed_category or "").split(CATEGORY_SEPARATOR) if p.strip()]
    if not parts:
        logger.warning("[T5] proposed_categoryが空です。'未分類'として扱います")
        return "未分類"

    existing = get_category_names(db_path)

    resolved: list[str] = []
    for part in parts:
        match = next(
            (
                name for name in existing
                if name.split(CATEGORY_SEPARATOR)[:-1] == resolved
                and _normalize(name.split(CATEGORY_SEPARATOR)[-1]) == _normalize(part)
            ),
            None,
        )
        if match is not None:
            resolved = match.split(CATEGORY_SEPARATOR)
            continue

        resolved = resolved + [part]
        full = CATEGORY_SEPARATOR.join(resolved)
        _register_new_category(full, db_path)
        existing.append(full)

    result = CATEGORY_SEPARATOR.join(resolved)
    if result != proposed_category:
        logger.info(f"[T5] 既存の表記に揃えました: '{proposed_category}' -> '{result}'")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    db.init_db()

    for proposed in ["経費精算", "ｹｲﾋ精算", "契約書／雇用契約書", "契約書／賃貸借契約書"]:
        print(f"[T5] '{proposed}' -> '{resolve_category(proposed)}'")

    print(f"\n[T5] 現在の既存カテゴリ一覧: {get_category_names()}")
