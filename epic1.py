"""
T1. フォルダ監視機能の実装
T2. 拡張子判定・不要ファイル除外

--- T1 ---
入力: 監視対象のフォルダパス
出力: 新規ファイルのパス（発生するたびに1件ずつ渡す）

- watchdog ライブラリでファイルシステムイベントを検知
- 対象外拡張子（DEFAULT_ALLOWED_EXTENSIONSに無いもの）はイベントを発火しない
- watch_folder() はジェネレータなので、呼び出し側（T2）は
  for new_file_path in watch_folder("対象フォルダ"): ... のように使う

--- T2 ---
入力: ファイルパス（T1の出力）
出力: 「処理していいか」の判定 + ファイル種別（text/pdf/photo）

- screen_file() が「処理対象かどうか」と「ファイル種別」を判定する
- .tmp / .crdownload 等の一時ファイルは対象外として弾く
- 読み込みに失敗したファイルはエラーログを出してスキップ（Noneを返す）
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from typing import Iterator, Optional, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# T2以降の対応形式（text / pdf / photo）に合わせたデフォルト拡張子
DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {
    ".txt", ".md", ".csv",           # text
    ".pdf",                          # pdf
    ".png", ".jpg", ".jpeg", ".gif", # photo
}


class _NewFileHandler(FileSystemEventHandler):
    """新規ファイル作成イベントを受け取り、内部キューに積むハンドラ"""

    def __init__(
        self,
        event_queue: "queue.Queue[str]",
        allowed_extensions: Set[str],
        debounce_seconds: float = 2.0,
    ):
        super().__init__()
        self._queue = event_queue
        self._allowed_extensions = allowed_extensions
        self._debounce_seconds = debounce_seconds
        self._last_enqueued: dict[str, float] = {}

    def _maybe_enqueue(self, path: Path) -> None:
        if path.suffix.lower() not in self._allowed_extensions:
            logger.debug(f"[T1] 対象外拡張子のため無視: {path}")
            return

        key = str(path)
        now = time.time()
        last = self._last_enqueued.get(key)

        # on_created と on_moved が同じファイルにほぼ同時に発火するケースがあるため、
        # 直近debounce_seconds以内に同じパスをキューに積んでいたら無視する
        if last is not None and (now - last) < self._debounce_seconds:
            return

        self._last_enqueued[key] = now
        logger.info(f"[T1] 新規ファイル検出: {path}")
        self._queue.put(key)

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        self._maybe_enqueue(Path(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:
        # ブラウザ等が「xxx.jpg.crdownload」→「xxx.jpg」のように
        # 一時ファイルから本来のファイル名へリネームした時にここで検知する
        if event.is_directory:
            return
        self._maybe_enqueue(Path(event.dest_path))


def watch_folder(
    target_dir: str,
    allowed_extensions: Optional[Set[str]] = None,
    poll_interval: float = 0.5,
) -> Iterator[str]:
    """
    指定フォルダを監視し、新規ファイルが作成されるたびに
    そのファイルパス（str）を1件ずつyieldする。

    Args:
        target_dir: 監視対象のフォルダパス（入力）
        allowed_extensions: 対象とする拡張子の集合。
            未指定時は DEFAULT_ALLOWED_EXTENSIONS を使用。
        poll_interval: 内部キューをチェックする間隔（秒）

    Yields:
        新規ファイルのパス（文字列）。T2にそのまま渡せる形式。

    Note:
        呼び出し側で無限ループするジェネレータなので、
        別スレッド/プロセスで動かすか、非同期で回すこと。
    """
    target_path = Path(target_dir)
    if not target_path.exists() or not target_path.is_dir():
        raise NotADirectoryError(f"監視対象フォルダが存在しません: {target_dir}")

    extensions = allowed_extensions if allowed_extensions is not None else DEFAULT_ALLOWED_EXTENSIONS

    event_queue: "queue.Queue[str]" = queue.Queue()
    handler = _NewFileHandler(event_queue, extensions)

    observer = Observer()
    observer.schedule(handler, str(target_path), recursive=False)
    observer.start()

    try:
        # アプリ起動中は監視を継続する
        while True:
            try:
                new_file_path = event_queue.get(timeout=poll_interval)
                yield new_file_path
            except queue.Empty:
                continue
    finally:
        observer.stop()
        observer.join()


# ============================================================
# T2. 拡張子判定・不要ファイル除外
# ============================================================

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使い方: python watch_folder.py <監視対象フォルダのパス>")
        sys.exit(1)

    watch_dir = sys.argv[1]
    print(f"[T1] 監視開始: {watch_dir}")
    print("(Ctrl+Cで終了)")

    try:
        for new_file in watch_folder(watch_dir):
            # T1の出力をそのままT2に渡す
            file_type = screen_file(new_file)

            if file_type is None:
                # T2で除外されたので後続処理には渡さない
                continue

            # ここでT3に渡すイメージ（今はprintのみ）
            print(f"[T2] 処理対象と判定: {new_file} (種別: {file_type}) -> T3へ渡す")
    except KeyboardInterrupt:
        print("\n[T1] 監視を終了しました")