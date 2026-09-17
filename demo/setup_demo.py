"""
デモ用のDBと実ファイルを作り直す。

    python demo/setup_demo.py

1. omakase.db と organized/ を消す（アプリは終了しておくこと）
2. 「以前から使っていた」想定のファイルを実際の分類パイプライン（OpenAI）に通して登録する
3. 登録日時・最後に開いた日時を、デモの筋書きに合わせて過去にずらす
4. 本番で監視フォルダに入れるファイルを demo/drop/ に用意する

日付は実行した日を基準にするので、**デモ当日に実行し直すこと**。

日付の設計（t7_search の期間の幅に合わせている）:
- 「昨日」= 1〜2日前 に請求書とグラフを置く → 「昨日のPDF」で請求書だけが出る
- 「先週」= 4〜11日前 には何も置かない → 「先週の契約書」は「見つからなかった」になる
- 30日以上開いていないもの と 重複 を置く → 片付けタイムの候補になる
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import db  # noqa: E402
import t6_filemanager as t6  # noqa: E402
from pipeline import process_file  # noqa: E402

DEMO = Path(__file__).resolve().parent
DROP = DEMO / "drop"
FONT = "C:/Windows/Fonts/meiryo.ttc"


# --- ファイルの素 -------------------------------------------------------------

def make_pdf(path: Path, title: str, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 80), title, fontname="japan", fontsize=20)
    y = 130
    for line in lines:
        page.insert_text((60, y), line, fontname="japan", fontsize=12)
        y += 24
    doc.save(path)


def make_receipt(path: Path) -> None:
    im = Image.new("RGB", (420, 620), "white")
    d = ImageDraw.Draw(im)
    big, small = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 20)
    d.text((150, 30), "領収書", fill="black", font=big)
    rows = ["カフェ・ド・渋谷", "2026年9月17日 12:41", "", "打合せ ランチセット ×2  2,400",
            "ブレンドコーヒー ×2    1,000", "", "合計          ¥3,400", "(内消費税10%   ¥309)",
            "", "但し 会議費として", "上記正に領収いたしました"]
    for i, row in enumerate(rows):
        d.text((40, 110 + i * 40), row, fill="black", font=small)
    im.save(path)


def make_chart(path: Path) -> None:
    im = Image.new("RGB", (640, 420), "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 22)
    d.text((20, 12), "2026年度 月別売上（万円）", fill="black", font=f)
    for i, (m, v) in enumerate(zip(["4月", "5月", "6月", "7月", "8月", "9月"], [120, 150, 90, 200, 170, 230])):
        x = 70 + i * 90
        d.rectangle([x, 380 - v, x + 50, 380], fill=(60, 120, 200))
        d.text((x + 5, 350 - v), str(v), fill="black", font=f)
        d.text((x + 5, 385), m, fill="black", font=ImageFont.truetype(FONT, 16))
    im.save(path)


def make_landscape(path: Path) -> None:
    im = Image.new("RGB", (480, 360), (135, 206, 235))
    d = ImageDraw.Draw(im)
    d.polygon([(0, 260), (140, 110), (280, 260)], fill=(90, 110, 140))
    d.polygon([(180, 260), (340, 90), (480, 260)], fill=(70, 90, 120))
    d.polygon([(300, 130), (340, 90), (380, 130)], fill="white")
    d.rectangle([0, 250, 480, 360], fill=(60, 160, 70))
    d.ellipse([380, 20, 450, 90], fill=(255, 215, 0))
    im.save(path)


# 以前から使っていたファイル: (ファイル名, 作り方, 何日前に入れたか, 何日前に開いたか or None=一度も開いていない)
def history_files(src: Path) -> list[tuple[Path, int, int | None]]:
    items: list[tuple[str, object, int, int | None]] = [
        ("統計学II_第3回_講義ノート.txt",
         "統計学II 第3回 講義ノート\n単回帰分析。最小二乗法で係数を推定し、決定係数R^2で当てはまりを評価する。残差プロットで外れ値を確認。", 20, 2),
        ("統計学II_第4回_講義ノート.txt",
         "統計学II 第4回 講義ノート\n重回帰分析と多重共線性。VIFで確認する。ダミー変数の扱い。", 13, 3),
        ("SparseAttention_論文.pdf",
         ("Efficient Sparse Attention for Long Documents",
          ["Abstract: We propose a sparse attention mechanism for long-document",
           "Transformers that reduces memory from O(n^2) to O(n log n).",
           "Experiments on long-range benchmarks show comparable accuracy."]), 25, 3),
        ("賃貸借契約書_渋谷1K.pdf",
         ("建物賃貸借契約書",
          ["賃貸人 田中一郎（甲） 賃借人 山田太郎（乙）",
           "物件 東京都渋谷区〇〇1-2-3 101号室（1K）",
           "賃料 月額85,000円 共益費 5,000円 敷金 1ヶ月",
           "契約期間 2026年4月1日から2年間"]), 60, 50),
        ("雇用契約書_インターン.pdf",
         ("雇用契約書",
          ["株式会社テック（甲）と 山田太郎（乙）は以下のとおり雇用契約を締結する。",
           "職種 ソフトウェアエンジニア（インターン）",
           "賃金 時給1,500円 期間 2026年10月1日〜2027年3月31日"]), 14, 1),
        ("請求書_Web制作_INV0912.pdf",
         ("請求書",
          ["株式会社サンプル商事 御中", "請求番号 INV-2026-0912",
           "Webサイト制作費 一式 550,000円（税込）",
           "お支払期限 2026年10月31日 振込先 みずほ銀行 渋谷支店"]), 1, None),
        ("電気料金_8月分.txt",
         "電気料金のお知らせ（ご請求）\nご使用期間 8/1〜8/31 使用量 312kWh\nご請求金額 9,874円 口座振替日 9/25", 40, None),
        ("チキンカレーのレシピ.txt",
         "チキンカレー（4人分）\n鶏もも肉400g 玉ねぎ2個 カレールー1/2箱\n1. 玉ねぎを飴色になるまで炒める\n2. 鶏肉を加えて炒め、水を入れて20分煮込む", 70, None),
        ("京都旅行のしおり.txt",
         "京都 秋の旅 しおり 11/22-11/23\n1日目 清水寺→祇園→鴨川沿いで夕食 宿: 四条烏丸\n2日目 嵐山 竹林の道 トロッコ列車", 90, None),
        ("お薬の説明書_ロキソプロフェン.txt",
         "お薬の説明書\nロキソプロフェンNa錠60mg 1回1錠 1日3回 毎食後\n副作用: 胃の不快感。空腹時の服用は避けてください。", 35, 33),
        ("定例ミーティング議事録_0905.txt",
         "定例ミーティング議事録 2026/09/05\n出席: 山田、佐藤、鈴木\n決定事項: 新サービスのリリースを10月に延期。次回までに見積もりを作成。", 12, 12),
        ("月別売上グラフ.png", make_chart, 1, 1),
        ("山の風景イラスト.png", make_landscape, 16, None),
        # 重複: 同じ中身をもう一度ダウンロードしたもの。元の方は開いているので、こちらが片付け候補になる
        ("雇用契約書_インターン (1).pdf", "DUP:雇用契約書_インターン.pdf", 14, None),
        ("月別売上グラフ (1).png", "DUP:月別売上グラフ.png", 1, None),
    ]
    made: list[tuple[Path, int, int | None]] = []
    for name, spec, created, opened in items:
        path = src / name
        if isinstance(spec, str) and spec.startswith("DUP:"):
            shutil.copy(src / spec[4:], path)
        elif isinstance(spec, str):
            path.write_text(spec, encoding="utf-8")
        elif isinstance(spec, tuple):
            make_pdf(path, *spec)
        else:
            spec(path)
        made.append((path, created, opened))
    return made


def make_drop_files() -> None:
    """本番で監視フォルダへ入れるファイル。"""
    if DROP.exists():
        shutil.rmtree(DROP)
    DROP.mkdir(parents=True)
    make_pdf(DROP / "業務委託契約書_データ分析.pdf", "業務委託契約書", [
        "委託者 株式会社テック（甲） 受託者 佐藤花子（乙）",
        "業務内容 販売データの分析およびレポート作成",
        "委託料 月額300,000円（税別） 期間 2026年10月1日〜2027年3月31日",
        "成果物の著作権は甲に帰属する。",
    ])
    make_receipt(DROP / "領収書_カフェ打合せ.png")
    (DROP / "出張旅費精算書_名古屋.txt").write_text(
        "出張旅費精算書\n申請日 2026/09/17 申請者 山田太郎（営業部）\n"
        "名古屋支店 打合せ 新幹線往復 21,540円 宿泊 1泊 8,800円\n合計 30,340円",
        encoding="utf-8",
    )
    (DROP / "おまけ_対象外.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    db.DB_PATH.unlink(missing_ok=True)
    shutil.rmtree(t6.ORGANIZED_ROOT, ignore_errors=True)
    db.init_db()

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp)
        for path, created, opened in history_files(src):
            result = process_file(str(path))
            if result is None:
                sys.exit(f"分類に失敗しました: {path.name}（OPENAI_API_KEY を確認）")
            # 期間の検索は暦の日付で数えるので、「N日前の何時」で置く（実行時刻に左右されない）
            created_at = datetime.combine(date.today() - timedelta(days=created), time(9, 30)).isoformat()
            accessed_at = (
                created_at if opened is None
                else datetime.combine(date.today() - timedelta(days=opened), time(17, 0)).isoformat()
            )
            conn = sqlite3.connect(db.DB_PATH)
            with conn:
                conn.execute(
                    "UPDATE files SET created_at = ?, last_accessed_at = ? WHERE id = ?",
                    (created_at, accessed_at, result["file_id"]),
                )
            conn.close()
            print(f"{created:3d}日前  {result['category']:32s} {path.name}")

    make_drop_files()
    print(f"\nデモで監視フォルダに入れるファイル: {DROP}")


if __name__ == "__main__":
    main()
