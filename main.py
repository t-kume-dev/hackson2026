"""
配線担当（SM）が書く部分。

「前のチケットの出力を次のチケットに渡す」処理だけをここに書く。
各チケットの中身には一切踏み込まない。

現状の配線: T1(watch_folder) -> T2(screen_file) -> (T3以降は未実装なのでprintのみ)
"""

from __future__ import annotations

import logging
import sys

from t1_watch import watch_folder
from t2_screening import screen_file

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    if len(sys.argv) < 2:
        print("使い方: python main.py <監視対象フォルダのパス>")
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


if __name__ == "__main__":
    main()
