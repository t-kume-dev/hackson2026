"""
T9: ゴミ箱・Undo

分類済みファイルへの「カテゴリの移し替え」と「ごみ箱へ」を1件ずつ取り消す。

前提:
- 削除は db.mark_trashed() が trash_log に action_type='delete' と元のパスを記録している
- カテゴリの移し替えは t6_filemanager.recategorize() が action_type='move' で記録している
- 分類時（ダウンロードフォルダ → organized/）の移動も 'move' で記録されるが、
  これは取り消し対象にしない。戻すとファイルがダウンロードフォルダに戻り、
  監視に再検出されて分類し直されてしまうため。
  見分け方は「元のパスが organized/ の中か」
- ファイルは organized/<階層1>/<階層2>/.../ファイル名 に置かれているので、
  元のパスのフォルダ階層をそのままカテゴリとして復元する（専用カラムは増やさない）
- Undo対象は常に有効なファイルだった前提なので、戻すステータスは常に'active'

DBアクセスは db.py の関数だけを使う。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import db
from t4_classify import CATEGORY_SEPARATOR

logger = logging.getLogger(__name__)

# t6_filemanager.ORGANIZED_ROOT と同じ場所。t6 を import すると
# 埋め込みモデルのライブラリまで読み込まれるので、ここでは直接組み立てる（t8と同じ）。
ORGANIZED_ROOT = Path(__file__).parent / "organized"


def _category_of(path: Path) -> Optional[str]:
    """organized/ 以下のパスからカテゴリを復元する。organized/ の外なら None。"""
    try:
        parts = path.parent.relative_to(ORGANIZED_ROOT).parts
    except ValueError:
        return None
    if not parts or parts[0] == ".trash":
        return None
    return CATEGORY_SEPARATOR.join(parts)


def get_last_action(file_id: Optional[int] = None) -> Optional[dict]:
    """
    取り消せる直近の操作を返す（DBは変更しない）。

    Args:
        file_id: 指定するとそのファイルの操作に限る。「元に戻す」ボタンが
            押されるまでの間に別のファイルが分類されても、狙った操作だけを戻すため。
    """
    for log in db.get_trash_logs(file_id):
        original = Path(log["original_path"])
        category = _category_of(original)
        if category is None:
            # 分類時の移動（ダウンロードフォルダから）。取り消し対象外
            continue
        file = db.get_file(log["file_id"])
        if file is None:
            continue
        return {
            "log_id": log["id"],
            "file_id": log["file_id"],
            "action_type": log["action_type"],
            "current_path": file["path"],
            "original_path": str(original),
            "original_category": category,
        }
    return None


def undo_last_action(file_id: Optional[int] = None) -> Optional[dict]:
    """
    直近の移動・削除を1件取り消す。
    実ファイルを current_path -> original_path へ戻してから、files/trash_log を更新する。

    Returns:
        取り消した操作（get_last_action() と同じ形）。戻せなかった場合は None
    """
    action = get_last_action(file_id)
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

    try:
        db.restore_file(
            action["file_id"], str(original), action["original_category"], action["log_id"]
        )
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
