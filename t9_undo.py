"""
T9: ゴミ箱・Undo

T8のtrash_file()や、db.update_file_location()経由の移動を1件取り消す。
常に「直近の1件」だけを対象にする（db.get_last_action()がtrash_logの最新行を返す）。

実ファイルを先にcurrent_path -> original_pathへ戻し、成功してからDBを更新する。
T8のtrash_file()と同じ理由（食い違いを作らないため）で、DB更新に失敗したら
実ファイルの移動もロールバックする。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import db

logger = logging.getLogger(__name__)


def peek_last_action() -> Optional[dict]:
    """
    直近のmove/delete操作の情報を返す（何も変更しない）。
    UI側で「〇〇を元に戻しますか？」と確認表示するのに使う。
    """
    return db.get_last_action()


def undo_last_action() -> Optional[dict]:
    """
    直近のmove/delete操作を1件取り消す。

    Returns:
        取り消した操作の情報(dict)。取り消す対象が無い/失敗した場合はNone。
    """
    action = db.get_last_action()
    if action is None:
        logger.info("[T9] 元に戻す操作がありません")
        return None

    current = Path(action["current_path"])
    original = Path(action["original_path"])

    if not current.exists():
        logger.error(f"[T9] 復元元のファイルが見つかりません: {current}")
        return None

    if original.exists():
        # 元の場所に別のファイルができていたら、上書きせず諦める
        logger.error(f"[T9] 復元先に既に別のファイルがあります: {original}")
        return None

    original.parent.mkdir(parents=True, exist_ok=True)

    try:
        current.rename(original)
    except OSError as e:
        logger.error(f"[T9] ファイルを元に戻せませんでした: {current} -> {original} ({e})")
        return None

    try:
        db.apply_undo(action["log_id"])
    except Exception:
        # DB更新に失敗したら実ファイルも戻して食い違いを作らない
        original.rename(current)
        raise

    logger.info(f"[T9] 元に戻しました({action['action_type']}): {current} -> {original}")
    return action


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = undo_last_action()
    if result is None:
        print("元に戻せる操作はありませんでした")
    else:
        print(f"元に戻しました: {result['action_type']} / {result['original_path']}")