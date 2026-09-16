"""
T1. フォルダ監視機能の実装

入力: 監視対象のフォルダパス
出力: 新規ファイルのパス（発生するたびに1件ずつ渡す）

- watchdog ライブラリでファイルシステムイベントを検知
- 対象外拡張子（DEFAULT_ALLOWED_EXTENSIONSに無いもの）はイベントを発火しない
- watch_folder() はジェネレータなので、呼び出し側（T2）は
  for new_file_path in watch_folder("対象フォルダ"): ... のように使う

このファイルの中身（実装方法）はT1担当の裁量。
外から使われるのは watch_folder() 関数だけ。
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from typing import Iterator, Optional, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

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
