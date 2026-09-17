"""
T9: ゴミ箱・Undo

db.pyは変更しない。db.get_connection()で得た接続を使い、
trash_log / files テーブルへは直接SQLでアクセスする。

前提:
- 削除は db.mark_trashed() が元々 trash_log に action_type='delete', original_path
  付きで記録している（db.py側は無変更）
- 「移動」（カテゴリ訂正など）のUndoに対応するには、呼び出し側が移動の前に
  record_move() を呼んで同じtrash_logに記録しておく必要がある
- ファイルは常に organized/<カテゴリ>/ファイル名 の構成で保存されている前提なので、
  original_path の親フォルダ名をそのままカテゴリとして復元する（専用カラムは増やさない）
- Undo対象は常に有効なファイルだった前提なので、戻すステータスは常に'active'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import db

logger = logging.getLogger(__name__)


def record_move(file_id: int, original_path: str) -> None:
    """
    移動（カテゴリ訂正など）をUndoできるよう記録する。
    db.update_file_location()はtrash_logに書かないので、呼び出し側で
    実際の移動処理の前にこれを呼ぶ。db.pyの既存関数をそのまま使うだけ。
    """
    db.insert_trash_log(file_id, original_path, "move")


def get_last_action() -> Optional[dict]:
    """直近のmove/delete操作を返す（DBは変更しない）。"""
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM trash_log ORDER BY acted_at DESC, id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        file_row = conn.execute(
            "SELECT path FROM files WHERE id = ?", (row["file_id"],)
        ).fetchone()
    finally:
        conn.close()
    if file_row is None:
        return None
    original_path = Path(row["original_path"])
    return {
        "log_id": row["id"],
        "file_id": row["file_id"],
        "action_type": row["action_type"],
        "current_path": file_row["path"],
        "original_path": str(original_path),
        "original_category": original_path.parent.name or None,
    }


def undo_last_action() -> Optional[dict]:
    """
    直近のmove/delete操作を1件取り消す。
    実ファイルをcurrent_path -> original_pathへ戻してから、files/trash_logを更新する。
    """
    action = get_last_action()
    if action is None:
        logger.info("[T9] 元に戻す操作がありません")
        return None

    current = Path(action["current_path"])
    original = Path(action["original_path"])

    if not current.exists():
        logger.error(f"[T9] 復元元のファイルが見つかりません: {current}")
        return None
    if original.exists():
        logger.error(f"[T9] 復元先に既に別のファイルがあります: {original}")
        return None

    original.parent.mkdir(parents=True, exist_ok=True)

    try:
        current.rename(original)
    except OSError as e:
        logger.error(f"[T9] ファイルを元に戻せませんでした: {current} -> {original} ({e})")
        return None

    conn = db.get_connection()
    try:
        with conn:
            category = action["original_category"]
            if category:
                conn.execute(
                    "UPDATE files SET path = ?, filename = ?, category = ?, status = 'active' WHERE id = ?",
                    (str(original), original.name, category, action["file_id"]),
                )
            else:
                conn.execute(
                    "UPDATE files SET path = ?, filename = ?, status = 'active' WHERE id = ?",
                    (str(original), original.name, action["file_id"]),
                )
            conn.execute("DELETE FROM trash_log WHERE id = ?", (action["log_id"],))
    except Exception:
        # DB更新に失敗したら実ファイルも戻して食い違いを作らない
        original.rename(current)
        raise
    finally:
        conn.close()

    logger.info(f"[T9] 元に戻しました({action['action_type']}): {current} -> {original}")
    return action


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = undo_last_action()
    if result is None:
        print("元に戻せる操作はありませんでした")
    else:
        print(f"元に戻しました: {result['action_type']} / {result['original_path']}")