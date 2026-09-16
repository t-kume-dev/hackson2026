"""
T7. 意味検索（自然文 → 該当ファイル）

【前提】DBアクセスは全て db.py に集約されている。このファイルはSQLを直接書かない。
    クエリのembeddingは必ず t6_filemanager.embed_query() を使う。
    files.embedding と同じモデル・同じプレフィックス・同じ正規化で作らないと、
    次元も意味も噛み合わず検索結果がデタラメになるため。

入力: 自然文のクエリ（例：「先週の統計学の資料」）
出力: SearchHit のリスト（スコアの高い順）

【自然文の時間表現について】
「先週の統計学の資料」を素のembedding検索に投げると、「先週」はベクトル上ほとんど
意味を持たず、統計学の資料が新旧まとめて並ぶだけになる。そこで検索を2段に分ける:

  1. クエリから期間・ファイル種別を正規表現で抜き出し、メタデータの絞り込み条件にする
  2. 残った文字列だけをembedding化して意味検索にかける

抜き出した条件は SearchResult.filters に入るので、UI側で「期間: 先週」のように
何で絞ったかを提示できる（勝手に絞られたと思われないようにするため）。

将来この解釈をLLMに任せる場合も、parse_query() の戻り値の形だけ揃えれば
search() から下はそのまま使える。
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

import db
from t6_filemanager import embed_query

logger = logging.getLogger(__name__)

# 返す件数。カードが大きいのでパネル幅では3件が上限。
DEFAULT_LIMIT = 3

# これ未満の類似度しかないものは「見つからなかった」と答える。
# 無理に何か返すより、外した結果を出さないほうが信用される。
# e5のコサイン類似度は無関係な文でも0.78前後まで出るため、絶対値だけでは切れない。
# 実測では正解が0.85〜0.91、無関係が0.78〜0.79に分布したので0.82を採用。
MIN_SCORE = 0.82

# 1位からこれ以上離れた結果は捨てる。「1件だけ強く当たっている」ときに
# 2位以下の惜しくないものを並べないための相対的な足切り。
RELATIVE_MARGIN = 0.06

# ファイル名・カテゴリにクエリの語がそのまま含まれていた場合の加点。
# embeddingは要約ベースなので、ファイル名の固有名詞に弱い分を補う。
EXACT_MATCH_BONUS = 0.04


@dataclass
class PeriodFilter:
    """「先週」のような時間表現を、検出日時からの経過日数の範囲に変換したもの。"""
    label: str
    lo: int  # 何日前から（含む）
    hi: int  # 何日前まで（含む）


# 期間の表現。上から順に評価し、最初に当たったものを使う。
# 「先週」の幅は厳密な暦週（月曜起点）ではなく4〜11日前としている。
# 暦どおりに切ると、水曜に「先週の資料」と言われたときに8日前の資料を取りこぼす。
PERIOD_PATTERNS: list[tuple[re.Pattern, PeriodFilter]] = [
    (re.compile(r"一昨日|おととい"), PeriodFilter("期間: 一昨日", 2, 3)),
    (re.compile(r"昨日|きのう"), PeriodFilter("期間: 昨日", 1, 2)),
    (re.compile(r"今日|きょう|さっき"), PeriodFilter("期間: 今日", 0, 1)),
    (re.compile(r"先々週"), PeriodFilter("期間: 先々週", 11, 18)),
    (re.compile(r"先週"), PeriodFilter("期間: 先週", 4, 11)),
    (re.compile(r"今週|ここ数日|最近"), PeriodFilter("期間: 今週", 0, 7)),
    (re.compile(r"先月"), PeriodFilter("期間: 先月", 30, 60)),
    (re.compile(r"今月"), PeriodFilter("期間: 今月", 0, 31)),
    (re.compile(r"去年|昨年"), PeriodFilter("期間: 去年", 365, 730)),
]

# ファイル種別の表現 → files.filetype の値
FILETYPE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"PDF|pdf|ＰＤＦ"), "pdf", "種別: PDF"),
    (re.compile(r"写真|画像|スクショ|スクリーンショット"), "photo", "種別: 写真"),
    (re.compile(r"テキスト|メモ帳"), "text", "種別: テキスト"),
]

# 意味を持たない助詞などは検索語としても除外する（ファイル名の直接一致判定用）
PARTICLES = re.compile(r"[のをでにがはやとも、。！？!?\s]+")


@dataclass
class ParsedQuery:
    """クエリを「絞り込み条件」と「意味検索にかける文字列」に分解した結果。"""
    text: str
    period: Optional[PeriodFilter] = None
    filetype: Optional[str] = None
    filter_labels: list[str] = field(default_factory=list)


@dataclass
class SearchHit:
    file: dict
    score: float


@dataclass
class SearchResult:
    query: str
    parsed: ParsedQuery
    hits: list[SearchHit]

    @property
    def filters(self) -> list[str]:
        """UIに「何で絞ったか」を出すためのラベル一覧。"""
        return self.parsed.filter_labels


def parse_query(query: str) -> ParsedQuery:
    """
    自然文のクエリから期間・ファイル種別を抜き出し、残りを意味検索用の文字列として返す。

    抜き出した表現はテキストから取り除く。「先週」をembeddingに残すと、
    たまたま「週」を含む要約が上位に来るノイズになるため。
    """
    text = query
    period: Optional[PeriodFilter] = None
    filetype: Optional[str] = None
    labels: list[str] = []

    for pattern, candidate in PERIOD_PATTERNS:
        if pattern.search(text):
            period = candidate
            labels.append(candidate.label)
            text = pattern.sub(" ", text)
            break

    for pattern, value, label in FILETYPE_PATTERNS:
        if pattern.search(text):
            filetype = value
            labels.append(label)
            text = pattern.sub(" ", text)
            break

    text = text.strip()
    if not text:
        # 「先週のやつ」のように条件しか言われなかった場合。
        # embeddingは効かせず、絞り込みだけで新しい順に返す。
        text = ""

    return ParsedQuery(text=text, period=period, filetype=filetype, filter_labels=labels)


def _days_since(created_at: Optional[str]) -> Optional[int]:
    """created_at（ISO文字列）から今日までの経過日数。壊れていればNone。"""
    if not created_at:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(created_at)).days
    except ValueError:
        logger.warning(f"[T7] created_atを解釈できません: {created_at}")
        return None


def _matches_filters(file: dict, parsed: ParsedQuery) -> bool:
    if parsed.filetype and file["filetype"] != parsed.filetype:
        return False
    if parsed.period:
        days = _days_since(file["created_at"])
        if days is None or not (parsed.period.lo <= days <= parsed.period.hi):
            return False
    return True


def _exact_match_bonus(file: dict, text: str) -> float:
    """ファイル名・カテゴリ・サブタグにクエリの語がそのまま入っていれば加点する。"""
    if not text:
        return 0.0
    haystack = " ".join(
        [file["filename"], file["category"], " ".join(file["subtags"])]
    )
    for word in PARTICLES.split(text):
        if len(word) >= 2 and word in haystack:
            return EXACT_MATCH_BONUS
    return 0.0


def search(
    query: str,
    limit: int = DEFAULT_LIMIT,
    min_score: float = MIN_SCORE,
) -> SearchResult:
    """
    自然文のクエリでファイルを検索する。

    files.embedding もクエリのembeddingもL2正規化済みなので、
    コサイン類似度は内積そのもので求まる。
    """
    parsed = parse_query(query)
    candidates = [f for f in db.get_active_files() if _matches_filters(f, parsed)]

    if not candidates:
        return SearchResult(query=query, parsed=parsed, hits=[])

    # 条件だけ指定された場合（「先週のやつ」等）は新しい順に返す
    if not parsed.text:
        hits = [SearchHit(file=f, score=1.0) for f in candidates[:limit]]
        return SearchResult(query=query, parsed=parsed, hits=hits)

    try:
        query_vector = embed_query(parsed.text)
    except Exception as e:
        logger.error(f"[T7] クエリのembedding生成に失敗しました: {e}")
        return SearchResult(query=query, parsed=parsed, hits=[])

    scored: list[SearchHit] = []
    for file in candidates:
        vector = file["embedding"]
        if vector is None or vector.shape != query_vector.shape:
            # embeddingモデルを変更した直後の古い次元のレコードはここで弾く
            logger.warning(f"[T7] embeddingの次元が一致しません: {file['filename']}")
            continue
        score = float(np.dot(query_vector, vector)) + _exact_match_bonus(file, parsed.text)
        scored.append(SearchHit(file=file, score=score))

    scored.sort(key=lambda h: h.score, reverse=True)
    if not scored:
        return SearchResult(query=query, parsed=parsed, hits=[])

    top = scored[0].score
    hits = [
        h for h in scored
        if h.score >= min_score and top - h.score <= RELATIVE_MARGIN
    ][:limit]
    return SearchResult(query=query, parsed=parsed, hits=hits)


def find_similar(file_id: int, limit: int = 2) -> list[SearchHit]:
    """
    指定ファイルに似たファイルを返す（「これに似たファイル」ボタン用）。

    files.embedding 同士を直接比較するだけなので、embeddingの作り直しは不要。
    エクスプローラーには原理的にできない操作で、同じ講義の別の回などが芋づるで出る。
    """
    files = db.get_active_files()
    target = next((f for f in files if f["id"] == file_id), None)
    if target is None:
        logger.error(f"[T7] ファイルが見つかりません: id={file_id}")
        return []

    hits: list[SearchHit] = []
    for file in files:
        if file["id"] == file_id:
            continue
        if file["embedding"] is None or file["embedding"].shape != target["embedding"].shape:
            continue
        score = float(np.dot(target["embedding"], file["embedding"]))
        # 同じカテゴリなら少し優先する（要約が似ていなくても関連は強い）
        if file["category"] == target["category"]:
            score += 0.03
        hits.append(SearchHit(file=file, score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def suggest_categories(file_id: int, limit: int = 3) -> list[str]:
    """
    分類を訂正するときの入れ直し先候補を、確からしい順に返す。

    カテゴリ名を新たにembedding化するのではなく、同じカテゴリに入っている
    既存ファイルのembeddingと比べて、最も近いカテゴリを拾う。
    numpyの内積だけで済むのでモデル呼び出しが要らず、UIを待たせない。

    まだ他にファイルが無いカテゴリは比較材料が無いので、末尾に新しい順で足す。
    """
    files = db.get_active_files()
    target = next((f for f in files if f["id"] == file_id), None)
    if target is None:
        return []

    best: dict[str, float] = {}
    for file in files:
        if file["id"] == file_id or file["category"] == target["category"]:
            continue
        if file["embedding"] is None or file["embedding"].shape != target["embedding"].shape:
            continue
        score = float(np.dot(target["embedding"], file["embedding"]))
        if score > best.get(file["category"], -1.0):
            best[file["category"]] = score

    ranked = [name for name, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]

    # 比較材料が無かったカテゴリ（まだ1件も入っていない等）も候補には残す
    for name in db.get_category_names():
        if name != target["category"] and name not in ranked:
            ranked.append(name)

    return ranked[:limit]


def open_file(file: dict) -> bool:
    """
    ファイルをOSの既定アプリで開く（エクスプローラーでダブルクリックしたのと同じ挙動）。

    ビューアは自前で持たない。PDFはブラウザ、写真はフォトに任せる。
    開く直前に last_accessed_at を記録する。放置ファイル検出（T8）の
    「未アクセス期間」はこの記録に乗っているため、ここを通さずに開かれると
    放置と誤判定される。
    """
    path = Path(file["path"])
    if not path.exists():
        logger.error(f"[T7] ファイルが見つかりません: {path}")
        return False

    db.touch_file(file["id"])

    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path)], check=True)
    except OSError as e:
        logger.error(f"[T7] ファイルを開けませんでした: {path} ({e})")
        return False

    logger.info(f"[T7] 既定アプリで開きました: {path}")
    return True


def reveal_file(file: dict) -> bool:
    """ファイルのある場所をエクスプローラー/Finderで開き、そのファイルを選択状態にする。"""
    path = Path(file["path"])
    if not path.exists():
        logger.error(f"[T7] ファイルが見つかりません: {path}")
        return False

    try:
        if sys.platform == "win32":
            subprocess.run(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=True)
        else:
            subprocess.run(["xdg-open", str(path.parent)], check=True)
    except OSError as e:
        logger.error(f"[T7] 場所を開けませんでした: {path} ({e})")
        return False
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("T7 t7_search.py")
    print(f"DB: {db.DB_PATH}")

    files = db.get_active_files()
    print(f"検索対象: {len(files)}件")

    for q in ["先週の統計学の資料", "契約書どこだっけ", "予防接種の知らせ"]:
        result = search(q)
        print(f"\n> {q}")
        if result.filters:
            print(f"  絞り込み: {' / '.join(result.filters)}")
        if not result.hits:
            print("  該当なし")
        for hit in result.hits:
            print(f"  {hit.score:.3f}  {hit.file['filename']}  [{hit.file['category']}]")
