"""
T2. 拡張子判定・不要ファイル除外

入力: ファイルパス（T1の出力）
出力: 「処理していいか」の判定 + ファイル種別（text/pdf/photo）

- screen_file() が「処理対象かどうか」と「ファイル種別」を判定する
- .tmp / .crdownload 等の一時ファイルは対象外として弾く
- 読み込みに失敗したファイルはエラーログを出してスキップ（Noneを返す）

このファイルの中身（実装方法）はT2担当の裁量。
外から使われるのは screen_file() 関数だけ。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)

# 一時ファイル・ダウンロード途中ファイルの拡張子（無条件で除外）
TEMP_FILE_EXTENSIONS: Set[str] = {".tmp", ".crdownload", ".part", ".download"}

# 拡張子 -> ファイル種別 の対応表（T4のAI分類に渡す種別）
EXTENSION_TYPE_MAP: dict[str, str] = {
    ".txt": "text", ".md": "text", ".csv": "text",
    ".pdf": "pdf",
    ".png": "photo", ".jpg": "photo", ".jpeg": "photo", ".gif": "photo",
}


def _wait_until_stable(
    path: Path,
    stable_duration: float = 1.0,
    max_wait: float = 300.0,
    check_interval: float = 0.2,
) -> bool:
    """
    ファイルサイズの変化が止まる（=書き込み完了）まで待つ。

    サイズが変化し続けている間（＝ダウンロード中）は待ち続けるので、
    大容量ファイルで5秒以上かかっても問題ない。
    ただし壊れたファイル等でサイズが変化し続けて終わらないケースに備えて
    max_wait（最大待機時間）で強制的に打ち切る。

    Args:
        path: 対象ファイル
        stable_duration: サイズ変化が止まってから「安定」と判定するまでの秒数
        max_wait: これ以上は待たない、という上限（秒）
        check_interval: チェック間隔（秒）

    Returns:
        True: 安定を確認できた
        False: max_waitまでに安定しなかった（またはファイルが消えた）
    """
    start_time = time.time()
    last_size = -1
    last_change_time = start_time

    while True:
        if time.time() - start_time > max_wait:
            return False

        if not path.exists():
            time.sleep(check_interval)
            continue

        try:
            current_size = path.stat().st_size
        except OSError:
            time.sleep(check_interval)
            continue

        if current_size != last_size:
            # サイズが変化した = まだ書き込み中なので、安定タイマーをリセット
            last_size = current_size
            last_change_time = time.time()
        elif current_size > 0 and (time.time() - last_change_time) >= stable_duration:
            # stable_duration秒間サイズが変わっていない = 書き込み完了とみなす
            return True

        time.sleep(check_interval)


def screen_file(file_path: str) -> Optional[str]:
    """
    ファイルが後続処理の対象かどうかを判定する。

    Args:
        file_path: 判定対象のファイルパス（入力。T1の出力をそのまま渡せる）

    Returns:
        処理対象の場合: ファイル種別 "text" / "pdf" / "photo" のいずれか
        処理対象外・エラー時: None（この場合、後続処理には渡さない）
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    # 一時ファイル・ダウンロード途中ファイルは対象外
    if suffix in TEMP_FILE_EXTENSIONS:
        logger.info(f"[T2] 一時ファイルのため除外: {file_path}")
        return None

    # 対応拡張子でなければ対象外
    file_type = EXTENSION_TYPE_MAP.get(suffix)
    if file_type is None:
        logger.info(f"[T2] 対象外拡張子のため除外: {file_path}")
        return None

    # ダウンロード中/リネーム直後の場合があるので、安定するまで待つ
    if not _wait_until_stable(path):
        logger.error(f"[T2] ファイルが安定しない/見つからないためスキップ: {file_path}")
        return None

    # 読み込みできるか確認（壊れたファイル・権限エラーなどをここで弾く）
    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError as e:
        logger.error(f"[T2] 読み込み失敗のためスキップ: {file_path} ({e})")
        return None

    return file_type
