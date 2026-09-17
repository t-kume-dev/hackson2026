"""
T8: 放置ファイル検出（片付けタイム）

分類済みのファイルから「しばらく開いていないもの」と「同じ中身が重複しているもの」を拾い、
MOFUとの会話で「残す」か「ごみ箱へ」かを1件ずつ決めてもらう。

【設計の前提】（詳細は Document/T8_放置ファイル検出_仕様書.md）
- 常時監視はしない。片付けタイムを始めるたびに、その場で候補を算出する
- 未アクセス期間は last_accessed_at だけで測る（分類時に created_at で初期化済み）
- 候補の条件は「STALE_DAYS日以上開いていない」か「重複していて残す側ではない」のどちらか。
  点数による重み付けはしない。理由はカードにそのまま出す
- 「ごみ箱へ」は即時削除しない。organized/.trash/ へ退避し、T9で元に戻せるようにする

重いモデルは使わないので、UIスレッドから直接呼んでよい。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import db

logger = logging.getLogger(__name__)

STALE_DAYS = 30   # これ以上開いていなければ候補にする
KEEP_DAYS = 30    # 「残す」を選んだファイルは、この期間は候補に出さない

# t6_filemanager.ORGANIZED_ROOT と同じ場所。t6 を import すると
# 埋め込みモデルのライブラリまで読み込まれるので、ここでは直接組み立てる。
TRASH_DIR = Path(__file__).parent / "organized" / ".trash"


@dataclass
class Candidate:
    file: dict                 # db.get_file() と同じ形
    stale_days: int            # 未アクセス日数
    duplicate: bool            # 重複の条件に当てはまるか
    reasons: list[str] = field(default_factory=list)  # カードにそのまま出す文言


def _parse(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning(f"[T8] 日時を解釈できません: {value}")
        return None


def _last_access(file: dict) -> datetime:
    """未アクセス期間の起点。last_accessed_at が壊れていたら created_at に倒す。"""
    return _parse(file["last_accessed_at"]) or _parse(file["created_at"]) or datetime.now()


def _keepers(files: list[dict]) -> dict[str, int]:
    """
    ハッシュごとに「残す側」のファイルidを決める。

    一番最近開かれたものを残し、同じなら先に入れたものを残す。
    """
    def rank(file: dict) -> tuple[datetime, float]:
        created = _parse(file["created_at"]) or datetime.now()
        return (_last_access(file), -created.timestamp())

    best: dict[str, dict] = {}
    for file in files:
        current = best.get(file["hash"])
        if current is None or rank(file) > rank(current):
            best[file["hash"]] = file
    return {h: f["id"] for h, f in best.items()}


def find_candidates(now: Optional[datetime] = None) -> list[Candidate]:
    """
    片付けの候補を算出し、仕様書3.3の順に並べて返す。

    並び順: 両方に当てはまる → 重複だけ → 未アクセスだけ。同じ順位なら未アクセス日数の長い順。
    """
    now = now or datetime.now()
    # 実ファイルが外で消されたものは、比べる相手にも候補にも含めない
    files = [f for f in db.get_active_files() if Path(f["path"]).exists()]

    keepers = _keepers(files)
    by_id = {f["id"]: f for f in files}
    counts: dict[str, int] = {}
    for file in files:
        counts[file["hash"]] = counts.get(file["hash"], 0) + 1

    candidates: list[Candidate] = []
    for file in files:
        kept_at = _parse(file["kept_at"])
        if kept_at is not None and (now - kept_at).days < KEEP_DAYS:
            continue

        stale_days = max(0, (now - _last_access(file)).days)
        stale = stale_days >= STALE_DAYS
        duplicate = counts[file["hash"]] > 1 and keepers[file["hash"]] != file["id"]
        if not (stale or duplicate):
            continue

        reasons = []
        if duplicate:
            keeper = by_id[keepers[file["hash"]]]
            reasons.append(f"「{keeper['category']}」に同じファイルがある")
        if stale:
            reasons.append(f"{stale_days}日開いてない")
        if file["last_accessed_at"] == file["created_at"]:
            reasons.append("入れてから一度も開いてない")

        candidates.append(Candidate(file, stale_days, duplicate, reasons))

    def order(c: Candidate) -> tuple[int, int]:
        stale = c.stale_days >= STALE_DAYS
        group = 0 if (c.duplicate and stale) else 1 if c.duplicate else 2
        return (group, -c.stale_days)

    candidates.sort(key=order)
    return candidates


def keep_file(file_id: int) -> None:
    """「残す」。KEEP_DAYS の間は候補に出さない。"""
    db.set_kept(file_id)


def trash_file(file_id: int) -> Optional[str]:
    """
    「ごみ箱へ」。実ファイルを organized/.trash/ へ移し、DBを trashed にする。

    Returns:
        退避先のパス。ファイルが見つからない・移せない場合は None
    """
    file = db.get_file(file_id)
    if file is None or file["status"] != "active":
        return None
    source = Path(file["path"])
    if not source.exists():
        logger.error(f"[T8] ファイルが見つかりません: {source}")
        return None

    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    # 同じ名前のファイルが何度もごみ箱に入るので、idを頭に付けて衝突を避ける
    destination = TRASH_DIR / f"{file_id}_{source.name}"

    try:
        source.rename(destination)
    except OSError as e:
        logger.error(f"[T8] ごみ箱へ移せませんでした: {source} ({e})")
        return None

    try:
        db.mark_trashed(file_id, str(source), str(destination))
    except Exception:
        # DBに残せなかったら、実ファイルも元に戻して食い違いを作らない
        destination.rename(source)
        raise

    logger.info(f"[T8] ごみ箱へ入れました: {source} -> {destination}")
    return str(destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    found = find_candidates()
    print(f"片付けの候補: {len(found)}件")
    for c in found:
        print(f"  id={c.file['id']} {c.file['category']}/{c.file['filename']} : {' / '.join(c.reasons)}")
