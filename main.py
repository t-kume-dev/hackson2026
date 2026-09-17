"""
おまかせ整理Bot の入口。

    python main.py                   # ダウンロードフォルダを監視してUIを起動
    python main.py <フォルダ>         # 監視対象を指定してUIを起動
    python main.py <フォルダ> --console  # UIなし。ログだけ流すコンソール版
    python main.py --undo            # 直近のmove/delete操作を1件取り消す（T9動作確認用）

既定ではネイティブアプリとして起動し、デスクトップ左下に常駐キャラ（MOFU）が出る。
ウィンドウは出ない。キャラをクリックすると会話パネルが開き、自然文でファイルを探せる。

【全体の構成】
- T1〜T6の受け渡しは pipeline.process_file() にまとまっている。
  UI版もコンソール版も同じ関数を呼ぶので、処理順を変えるときは pipeline.py を直す。
- UIの組み立ては ui_kohaku.py。ここでは起動方法を決めるだけで、UIの中身は持たない。
- SQLiteへのアクセスは全て db.py に集約されている。DBのパスは db.DB_PATH ただ1つが正。

--console は配線の確認・デバッグ用に残してある。GUIを起動せずに
「検出 → 分類 → 移動 → DB保存」の各段がどこで止まったかを見たいとき用。
--undo も同じ位置づけ。T9(ゴミ箱・Undo)を会話UIに組み込む前に、
コマンド一発で「直近の操作を戻す」動作だけ確認できるようにしてある。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import db
from pipeline import process_file
from t1_watch import watch_folder

DEFAULT_WATCH_DIR = Path.home() / "Downloads"


def run_console(watch_dir: str) -> None:
    """UIを使わず、標準出力にパイプラインの進行を書き出す。"""
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


def run_undo() -> None:
    """
    直近のmove/delete操作を1件取り消す（T9動作確認用）。
    会話UIに組み込む前に、コマンド一発で挙動を確認するためのもの。
    """
    from t9_undo import undo_last_action

    result = undo_last_action()
    if result is None:
        print("[T9] 元に戻せる操作はありませんでした")
    else:
        print(f"[T9] 元に戻しました: {result['action_type']} / {result['original_path']}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    args = [a for a in sys.argv[1:] if a not in ("--console", "--undo")]
    console = "--console" in sys.argv
    undo = "--undo" in sys.argv

    # テーブルが無ければここで作る（何度呼んでも安全）
    db.init_db()
    print(f"[DB] 使用するDB: {db.DB_PATH}")

    # --undo は監視フォルダを必要としないので、フォルダの存在チェックより先に処理する
    if undo:
        run_undo()
        return

    watch_dir = Path(args[0]) if args else DEFAULT_WATCH_DIR

    if not watch_dir.is_dir():
        print(f"監視対象のフォルダが見つかりません: {watch_dir}")
        print("使い方: python main.py [監視するフォルダ] [--console | --undo]")
        sys.exit(1)

    if console:
        run_console(str(watch_dir))
        return

    # UIの読み込みはここまで遅らせる。--console しか使わない環境に
    # PySide6が入っていなくても、コンソール版は動かせるようにするため。
    from ui_kohaku import run

    sys.exit(run(str(watch_dir)))


if __name__ == "__main__":
    main()