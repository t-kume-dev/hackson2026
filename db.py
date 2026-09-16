"""
SQLite DB層（T5・T6・T7以降が共通で使う「外側の箱」）

テーブル定義は仕様書（おまかせ整理Bot_仕様書.md 8章）に準拠。
接続・スキーマ管理・BLOB変換はここに閉じ込め、各チケットは下記の関数を呼ぶだけでよい。

【重要】DBファイルのパスはこのファイルの DB_PATH ただ1つが正。
呼び出し側で "xxx.db" のような相対パスを書かないこと（実行時のカレントディレクトリが
変わると別のDBファイルが作られ、カテゴリが消えたように見えるため）。

- init_db()                                       : DBファイルとテーブルを用意する（起動時に呼ぶ）
- get_category_names() / get_category_embeddings()
  / insert_category() / update_category_embedding()  : T5（カテゴリ重複防止）が使う
- insert_file() / insert_trash_log()                 : T6（ファイル移動＋DB保存）が使う
- get_active_files() / get_file() / touch_file()
  / update_file_location() / set_file_status()       : T7（意味検索・ファイル操作）が使う

embeddingはnumpy配列（float32）としてやり取りし、DBにはBLOBとして保存する。
categories.embedding だけはNULLを許容する（T5がembedding API呼び出しに失敗したとき、
カテゴリ名だけ先に登録し、後の呼び出しでバックフィルする運用のため）。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

DB_PATH = Path(__file__).parent / "omakase.db"

# embeddingとして受け付ける型（numpy配列でも素のfloatリストでもよい）
Vector = Union[np.ndarray, Sequence[float]]

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
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS trash_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES files(id),
    original_path TEXT NOT NULL,
    action_type TEXT NOT NULL,
    acted_at DATETIME NOT NULL
);
"""


def _resolve_path(db_path: Optional[Union[str, Path]]) -> Union[str, Path]:
    """db_pathが省略された場合は既定のDB_PATHを使う。"""
    return DB_PATH if db_path is None else db_path


def _migrate(conn: sqlite3.Connection) -> None:
    """
    古いスキーマのDBを開いた場合の埋め合わせ。
    embeddingカラムが無い時代のcategoriesテーブルが残っていても、
    既存の id / name のデータを保ったままカラムを追加する。
    """
    columns = [row[1] for row in conn.execute("PRAGMA table_info(categories)").fetchall()]
    if "embedding" not in columns:
        conn.execute("ALTER TABLE categories ADD COLUMN embedding BLOB")


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    DB接続を1つ返す。テーブルが無ければこの中で作るので、呼び出し側は
    事前にinit_db()を呼んでいなくても安全に使える。

    接続の閉じ忘れを避けるため、呼び出し側は try/finally で close() すること。
    （sqlite3.Connection を `with` に渡してもトランザクションが閉じるだけで
      接続自体は閉じられない点に注意）
    """
    conn = sqlite3.connect(_resolve_path(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """テーブルが無ければ作成する。アプリ起動時に一度呼べばよい（何度呼んでも安全）。"""
    conn = get_connection(db_path)
    conn.close()


def to_blob(embedding: Vector) -> bytes:
    """embeddingベクトルをBLOB保存用のbytesに変換する（float32で保存しサイズを抑える）。"""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """BLOBからembeddingベクトルを復元する。"""
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# T5: カテゴリ重複防止
# ---------------------------------------------------------------------------

def get_category_names(db_path: Optional[Union[str, Path]] = None) -> list[str]:
    """既存カテゴリ名の一覧（T4のプロンプトに埋め込む用）。"""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT name FROM categories").fetchall()
    finally:
        conn.close()
    return [row["name"] for row in rows]


def get_category_embeddings(
    db_path: Optional[Union[str, Path]] = None,
) -> list[tuple[str, Optional[np.ndarray]]]:
    """
    既存カテゴリの (名前, embedding) 一覧。コサイン類似度計算に使う。
    embeddingがまだ計算されていないカテゴリは None が入る。
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT name, embedding FROM categories").fetchall()
    finally:
        conn.close()
    return [
        (row["name"], from_blob(row["embedding"]) if row["embedding"] is not None else None)
        for row in rows
    ]


def insert_category(
    name: str,
    embedding: Optional[Vector] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    新規カテゴリを登録する。同名が既にあれば何もしない。
    embeddingがNoneの場合はカテゴリ名だけ登録する（後でバックフィルする前提）。
    """
    blob = to_blob(embedding) if embedding is not None else None
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO categories (name, embedding) VALUES (?, ?)",
            (name, blob),
        )
        conn.commit()
    finally:
        conn.close()


def update_category_embedding(
    name: str,
    embedding: Vector,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """既存カテゴリ行のembeddingカラムを埋める（初回アクセス時のキャッシュ書き戻し用）。"""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE categories SET embedding = ? WHERE name = ?",
            (to_blob(embedding), name),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# T6: ファイル移動＋DB保存
# ---------------------------------------------------------------------------

def insert_file(
    path: str,
    filetype: str,
    category: str,
    subtags: list[str],
    summary: str,
    embedding: Vector,
    file_hash: str,
    file_size: int,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """
    移動後のファイル情報をfilesテーブルに保存する。

    connを渡した場合はその接続を使い、commitもcloseもしない
    （insert_trash_logと同じトランザクションにまとめたい場合に使う）。

    Returns:
        挿入した行のid（trash_log等から参照する際に使う）
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO files
                (path, filename, filetype, category, subtags, summary,
                 embedding, hash, file_size, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                path, Path(path).name, filetype, category,
                json.dumps(subtags, ensure_ascii=False), summary,
                to_blob(embedding), file_hash, file_size,
                datetime.now().isoformat(),
            ),
        )
        if owns_conn:
            conn.commit()
        return cursor.lastrowid
    finally:
        if owns_conn:
            conn.close()


def insert_trash_log(
    file_id: int,
    original_path: str,
    action_type: str,
    conn: Optional[sqlite3.Connection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    ファイル移動・削除操作をtrash_logに記録する（Undo用）。action_type: 'move' | 'delete'

    connを渡した場合はその接続を使い、commitもcloseもしない。
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO trash_log (file_id, original_path, action_type, acted_at)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, original_path, action_type, datetime.now().isoformat()),
        )
        if owns_conn:
            conn.commit()
    finally:
        if owns_conn:
            conn.close()


# ---------------------------------------------------------------------------
# T7: 意味検索・ファイル操作
# ---------------------------------------------------------------------------

def _row_to_file(row: sqlite3.Row) -> dict:
    """filesテーブルの1行を扱いやすいdictに変換する（embeddingはnumpy配列に復元）。"""
    return {
        "id": row["id"],
        "path": row["path"],
        "filename": row["filename"],
        "filetype": row["filetype"],
        "category": row["category"],
        "subtags": json.loads(row["subtags"]) if row["subtags"] else [],
        "summary": row["summary"],
        "embedding": from_blob(row["embedding"]),
        "hash": row["hash"],
        "file_size": row["file_size"],
        "created_at": row["created_at"],
        "last_accessed_at": row["last_accessed_at"],
        "status": row["status"],
    }


def get_active_files(db_path: Optional[Union[str, Path]] = None) -> list[dict]:
    """
    status='active' のファイルを全件返す（T7の意味検索・T8の放置判定が使う）。

    件数が数千を超えるまでは全件をメモリに載せて総当たりで十分速い。
    近似最近傍インデックスが必要になったらここを差し替える。
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM files WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_file(row) for row in rows]


def get_file(file_id: int, db_path: Optional[Union[str, Path]] = None) -> Optional[dict]:
    """idで1件取得する。見つからなければNone。"""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_file(row) if row is not None else None


def touch_file(file_id: int, db_path: Optional[Union[str, Path]] = None) -> None:
    """
    last_accessed_at を現在時刻で更新する。アプリ経由でファイルを開いた瞬間に呼ぶ。

    OSのatimeは設定次第で更新されず信頼できないため、放置ファイル検出（T8）の
    「未アクセス期間」はこの記録を一次情報として使う。
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE files SET last_accessed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), file_id),
        )
        conn.commit()
    finally:
        conn.close()


def update_file_location(
    file_id: int,
    new_path: str,
    new_category: str,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    分類をユーザーが訂正したときに、保存先とカテゴリを更新する。

    実ファイルの移動は呼び出し側（T6）の責任。ここはDBの整合を取るだけ。
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE files SET path = ?, filename = ?, category = ? WHERE id = ?",
            (new_path, Path(new_path).name, new_category, file_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_file_status(
    file_id: int,
    status: str,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """status を 'active' / 'trashed' の間で切り替える（T9のゴミ箱・Undo用）。"""
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE files SET status = ? WHERE id = ?", (status, file_id))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"DB初期化完了: {DB_PATH}")
