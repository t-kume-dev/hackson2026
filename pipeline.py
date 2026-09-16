"""
T1〜T6をひとつながりに実行する処理（配線本体）。

コンソール版（main.py）とGUI版（ui_kohaku.py）の両方から呼ばれる。
以前は main.py の中にだけ書かれていたが、GUIでも同じ順序で呼ぶ必要が出たため、
「前のチケットの出力を次のチケットに渡す」部分だけをここに切り出した。

各チケットの中身には踏み込まない。ここがやるのは受け渡しと、
途中で止まったときに何段目で止まったかを呼び出し側に伝えることだけ。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from t2_screening import screen_file
from t3_content import extract_content
from t4_classify import classify_content
from t5_dedupe import get_category_names, resolve_category
from t6_filemanager import save_result

logger = logging.getLogger(__name__)

# 進捗の通知先（GUIが「読んでるね…」を出すために使う）。使わない場合はNone。
ProgressCallback = Optional[Callable[[str], None]]


def process_file(file_path: str, on_progress: ProgressCallback = None) -> Optional[dict]:
    """
    1ファイルを検出直後の状態から「分類済み・移動済み・DB保存済み」まで進める。

    Returns:
        成功した場合:
            {"file_id": int, "moved_path": str, "filename": str,
             "category": str, "subtags": list[str], "summary": str, "filetype": str}
        途中で除外・失敗した場合: None
    """
    def progress(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    filetype = screen_file(file_path)
    if filetype is None:
        logger.info(f"[T2] 対象外のため除外: {file_path}")
        return None

    progress("中身を読んでいます")
    content = extract_content(file_path, filetype)
    if content is None:
        logger.warning(f"[T3] コンテンツを抽出できませんでした: {file_path}")
        return None

    progress("分類しています")
    classified = classify_content(content, get_category_names())
    if classified is None:
        logger.warning(f"[T4] 分類に失敗しました: {file_path}")
        return None

    # T4が提案したカテゴリ名を、既存カテゴリと突き合わせて確定させる
    final_category = resolve_category(classified["category"])

    progress("片付けています")
    saved = save_result(
        file_path,
        {
            "category": final_category,
            "subtags": classified["subtags"],
            "summary": classified["summary"],
        },
    )
    if not saved["db_saved"]:
        logger.error(f"[T6] 保存に失敗しました: {file_path}")
        return None

    return {
        "file_id": saved["file_id"],
        "moved_path": saved["moved_path"],
        "filename": Path(saved["moved_path"]).name,
        "category": final_category,
        "subtags": classified["subtags"],
        "summary": classified["summary"],
        "filetype": filetype,
    }
