"""
T2のAC「読み込みに失敗したファイルはエラーログを出力し、処理をスキップする」を
確認するためのテストスクリプト（OSの権限操作に頼らず、open()をモックして
読み込み失敗を確実に再現する版）

使い方:
    python test_t2_read_failure.py

epic1.py（本体）と同じフォルダに置いて実行してください。
"""

import os
import tempfile
from unittest.mock import patch

from epic1 import screen_file  # noqa: E402  ファイル名が違う場合はここを合わせる


def main() -> None:
    # 1. テスト用ファイルを作成（対象拡張子の.txt、中身も入れておく）
    fd, temp_path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("これはテストファイルです")

    print(f"[TEST] テストファイル作成: {temp_path}")

    try:
        # 2. open() を差し替えて「呼んだら必ず失敗する」ようにする
        #    -> OSの権限設定に関係なく、確実に読み込み失敗を再現できる
        print("[TEST] open()を失敗するようにモックして screen_file() を実行します...")
        with patch("builtins.open", side_effect=PermissionError("模擬的な読み込み失敗")):
            result = screen_file(temp_path)

        # 3. 検証
        if result is None:
            print("[TEST] OK: Noneが返り、後続処理はスキップされました（エラーログも出ているはず）")
        else:
            print(f"[TEST] NG: 読み込み失敗のはずが種別 '{result}' が返ってきました")

    finally:
        os.remove(temp_path)
        print(f"[TEST] テストファイルを削除しました: {temp_path}")


if __name__ == "__main__":
    main()
