"""
T6単体テスト

T1〜T4は使わず、T4から渡ってくる想定の分類結果を
仮データとしてT6に直接渡す。
"""

from pathlib import Path

from db import init_db
from t6_filemanager import save_result


def main() -> None:
    # =========================
    # DBを初期化
    # =========================

    init_db()

    # =========================
    # T6用の仮ファイルを作成
    # =========================

    test_file = Path("t6_test.txt")

    test_file.write_text(
        """社内プロジェクトの会議メモ

プロジェクト名：おまかせ整理Bot

本日の打ち合わせ内容：
・ファイル自動整理機能について確認
・Gemini APIを利用したファイル分類を実装
・分類されたファイルをカテゴリごとのフォルダーへ移動
・SQLiteデータベースにファイル情報を保存
・今後は意味検索機能を追加する予定

次回までのタスク：
・T6のファイル移動処理を確認する
・データベースへの保存結果を確認する
""",
        encoding="utf-8",
    )

    print("=== T6単体テスト開始 ===")
    print(f"テストファイル: {test_file}")

    # =========================
    # T4から渡ってくる想定の仮データ
    # =========================

    classification = {
        "category": "プロジェクト",
        "subtags": [
            "会議",
            "開発",
            "ファイル整理",
        ],
        "summary": "おまかせ整理Botの開発に関する会議メモ",
    }

    print("仮の分類結果:")
    print(classification)

    # =========================
    # T6実行
    # =========================

    result = save_result(
        file_path=str(test_file),
        classification=classification,
    )

    # =========================
    # 結果表示
    # =========================

    print("\n=== T6実行結果 ===")
    print(result)

    if result["db_saved"]:
        print("\n[T6] 成功！")
        print(f"ファイル移動先: {result['moved_path']}")
        print("DB保存: 成功")
    else:
        print("\n[T6] 失敗")
        print("DB保存: 失敗")


if __name__ == "__main__":
    main()