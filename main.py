"""
配線担当（SM）が書く部分（コンソール版）。

T1〜T6の受け渡しそのものは pipeline.process_file() に切り出してある。
GUI版（ui_kohaku.py）も同じ関数を呼ぶので、順序を変えるときは pipeline.py を直すこと。
ここに残っているのは「監視ループを回して結果を標準出力に出す」部分だけ。

デモや常駐で使うのはGUI版:
    python ui_kohaku.py <監視対象フォルダ>

【DBについて】
- SQLiteへのアクセスは全て db.py に集約されている（T5がcategoriesテーブル、
  T6がfiles/trash_logテーブル、T7が検索とlast_accessed_atを担当）。
  DBファイルのパスは db.DB_PATH ただ1つが正。
"""

from __future__ import annotations

import logging
import sys

import db
from pipeline import process_file
from t1_watch import watch_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    if len(sys.argv) < 2:
        print("使い方: python main.py <監視対象フォルダのパス>")
        sys.exit(1)

    # テーブルが無ければここで作る（何度呼んでも安全）
    db.init_db()
    print(f"[DB] 使用するDB: {db.DB_PATH}")

    watch_dir = sys.argv[1]
    print(f"[T1] 監視開始: {watch_dir}")
    print("(Ctrl+Cで終了)")

    try:
        for new_file in watch_folder(watch_dir):
            print(f"[T1] 検出: {new_file}")

            result = process_file(new_file, on_progress=lambda m: print(f"  ... {m}"))

            if result is None:
                print(f"[--] 処理されませんでした: {new_file}")
                continue

            print(f"[T6] 保存完了: {result['moved_path']}")
            print(f"     カテゴリ: {result['category']} / タグ: {result['subtags']}")
            print(f"     要約: {result['summary']}")
    except KeyboardInterrupt:
        print("\n[T1] 監視を終了しました")


if __name__ == "__main__":
    main()
