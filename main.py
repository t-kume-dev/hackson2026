"""
配線担当（SM）が書く部分。

「前のチケットの出力を次のチケットに渡す」処理だけをここに書く。
各チケットの中身には一切踏み込まない。

現状の配線: T1(watch_folder) -> T2(screen_file) -> T3(extract_content) -> (T4以降は未実装なのでprintのみ)

【T3の出力形式】常にdict。中身は filetype で判別する。
- filetype が "text" / "pdf" の場合: {"filetype", "file_path", "content"}
- filetype が "photo" の場合       : {"filetype", "file_path", "image_bytes", "mime_type"}
"""

from __future__ import annotations

import logging
import sys

from t1_watch import watch_folder
from t2_screening import screen_file
from t3_content import extract_content

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

            print(f"[T2] 処理対象と判定: {new_file} (種別: {file_type}) -> T3へ渡す")

            # T2の出力（ファイルパス＋種別）をそのままT3に渡す
            t3_result = extract_content(new_file, file_type)

            if t3_result is None:
                # T3で抽出失敗/無効だったので後続処理には渡さない
                print(f"[T3] コンテンツ抽出失敗のためスキップ: {new_file}")
                continue

            # T3の出力をそのままT4に渡すイメージ（今はprintのみ）
            if t3_result["filetype"] == "photo":
                size_kb = len(t3_result["image_bytes"]) / 1024
                print(
                    f"[T3] 画像を読み込み完了: {t3_result['file_path']} "
                    f"({t3_result['mime_type']}, {size_kb:.1f}KB) -> T4へ画像入力として渡す"
                )
            else:
                preview = t3_result["content"][:100].replace("\n", " ")
                print(
                    f"[T3] コンテンツ抽出完了: {t3_result['file_path']} "
                    f"(先頭100文字: {preview}...) -> T4へテキスト入力として渡す"
                )
    except KeyboardInterrupt:
        print("\n[T1] 監視を終了しました")


if __name__ == "__main__":
    main()