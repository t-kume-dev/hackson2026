"""
コハク: デスクトップ常駐の相棒UI（PySide6）

デスクトップの隅にキャラが浮かんでいて、
- ダウンロードを検出したら吹き出しで「どこに入れたか」を知らせる（自発）
- クリックするとチャットパネルが開き、自然文でファイルを探せる（ユーザー始動）
という2つの面を持つ。検索結果も分類通知も訂正も、すべて1本の会話ログに流す。

【設計の前提】
- ファイルビューアは自前で持たない。「開く」はOSの既定アプリに投げるだけ
  （エクスプローラーでダブルクリックしたのと同じ挙動）。
  この導線を通ったときに last_accessed_at が更新され、放置検出の根拠になる。
- 分類は必ず外れるので、通知から直接カテゴリを訂正できるようにしている。
- 重い処理（分類パイプライン・embedding生成）はUIスレッドで走らせない。
  画面が固まると常駐アプリとしては致命的なので、全てワーカースレッドに逃がす。

【PyQt6ではなくPySide6を使っている理由】
WindowsのSmart App Control（アプリケーション制御ポリシー）が有効な環境では、
PyQt6のDLLがブロックされてimportすら通らない。PySide6はQt公式の署名付きバイナリなので
同じ環境で問題なく動く。APIはほぼ同じだが、シグナルの定義が pyqtSignal ではなく
Signal になる点だけ違う。

起動:
    python main.py <監視対象フォルダ>
    （アプリの入口は main.py。このファイルを直接叩いても動くが通常はそちら）
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QPoint, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPixmap,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

import db
import t6_filemanager as t6
import t7_search as t7
import t8_cleanup as t8
import t9_undo
from t4_classify import CATEGORY_SEPARATOR, MAX_CATEGORY_DEPTH
from t5_dedupe import resolve_category
from pipeline import process_file
from t1_watch import watch_folder

logger = logging.getLogger(__name__)

# 見た目は白黒のドット絵。色は使わず、黒地に白文字・白枠で統一する。
# 角丸も影もグラデーションも使わない（ドット絵の質感が濁るため）。
BLACK = "#111111"     # 真っ黒だと白との差が強すぎて目が疲れるので少しだけ浮かせる
WHITE = "#f4f4f4"
DIM = "#b9b9b9"       # 補助テキスト。暗くしすぎると黒地で読めなくなる
SOFT = "#6f6f6f"      # 控えめな枠線（二次的なボタン・入力欄）
LINE = "#2c2c2c"      # 区切り線。面を分けるだけで主張させない
SUNK = "#1d1d1d"      # カードの面。枠で囲まず、面の明るさで区切る
SPRITE_GRAY = "#a6a6a6"  # キャラのほっぺ・舌

# 通知の吹き出しが消えるまでの時間。作業中に居座られるのが一番嫌われるので短く。
TOAST_MS = 5000

# 片付けタイム（T8）
CLEANUP_BATCH = 5                          # 1回に出す候補の上限
INVITE_FIRST_DELAY_MS = 60 * 1000          # 起動してから最初に声をかけるまで
INVITE_INTERVAL = timedelta(hours=24)      # 声かけは1日1回まで
INVITE_RETRY_MS = 10 * 1000                # 通知中・分類中なら少し待って出し直す

# フォントは2系統に分ける。
# - VOICE: コハクの「声」（吹き出し・ボタン・タイトル）。同梱のドット絵フォント DotGothic16。
#   16px基準のフォントなので、それより小さくすると潰れて読めなくなる。14px以上で使う。
# - TEXT: ファイル名・要約など情報量の多い部分。小さくても読める普通のUIフォント。
#   全部ドット絵フォントにすると、世界観は揃うが実用に耐えない（実際に見づらかった）。
FONT_FILE = Path(__file__).parent / "assets" / "fonts" / "DotGothic16-Regular.ttf"
VOICE = '"DotGothic16", "MS Gothic", monospace'
TEXT = '"Yu Gothic UI", "Meiryo UI", "Meiryo", sans-serif'
FONT = VOICE


def _load_pixel_font() -> None:
    """同梱のドット絵フォントをアプリに登録する。QApplication生成後に呼ぶこと。"""
    if not FONT_FILE.exists():
        logger.warning(f"[UI] フォントが見つかりません: {FONT_FILE}")
        return
    if QFontDatabase.addApplicationFont(str(FONT_FILE)) < 0:
        logger.warning(f"[UI] フォントを読み込めませんでした: {FONT_FILE}")


def _stylize(html: str) -> str:
    """
    <b>…</b> を白黒反転の帯に置き換える。

    ドット絵フォントには太字が無く、そのまま使うと疑似太字で文字が滲む。
    色も使わない方針なので、強調は「白地に黒文字」の反転で表す。
    """
    return html.replace(
        "<b>", f"<span style='background:{WHITE}; color:{BLACK};'>&nbsp;"
    ).replace("</b>", "&nbsp;</span>")


# ---------------------------------------------------------------------------
# ワーカースレッド
# ---------------------------------------------------------------------------

class WatcherThread(QThread):
    """フォルダ監視と分類パイプラインをUIスレッドの外で回す。"""

    progressed = Signal(str)       # 「分類しています」等
    classified = Signal(dict)      # process_file() の戻り値
    failed = Signal(str)           # 失敗したファイルのパス

    def __init__(self, watch_dir: str) -> None:
        super().__init__()
        self.watch_dir = watch_dir

    def run(self) -> None:
        for new_file in watch_folder(self.watch_dir):
            try:
                result = process_file(new_file, on_progress=self.progressed.emit)
            except Exception as e:  # パイプラインが落ちても常駐は続ける
                logger.exception(f"[UI] 処理中に例外が発生しました: {new_file} ({e})")
                self.failed.emit(new_file)
                continue
            if result is None:
                self.failed.emit(new_file)
            else:
                self.classified.emit(result)


class SearchThread(QThread):
    """検索1回分をUIスレッドの外で実行する（初回はモデル読み込みで数秒かかる）。"""

    done = Signal(object)

    def __init__(self, query: str) -> None:
        super().__init__()
        self.query = query

    def run(self) -> None:
        try:
            self.done.emit(t7.search(self.query))
        except Exception as e:
            logger.exception(f"[UI] 検索に失敗しました: {e}")
            self.done.emit(None)


# ---------------------------------------------------------------------------
# 部品
# ---------------------------------------------------------------------------

def _button(text: str, kind: str = "plain") -> QPushButton:
    """会話ログの中に置く小さなボタン。"""
    # 主ボタンだけ白地に黒（反転）。それ以外は細い灰色の枠に留め、
    # 全部を同じ強さで囲まないことで「どれを押せばいいか」を分かりやすくする。
    if kind == "primary":
        base = f"background:{WHITE}; color:{BLACK}; border:2px solid {WHITE};"
    else:
        base = f"background:transparent; color:{WHITE}; border:1px solid {SOFT};"

    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(
        f"QPushButton {{ {base} border-radius:0px; padding:4px 11px;"
        f" font-size:14px; font-family:{VOICE}; }}"
        f"QPushButton:hover {{ background:{WHITE}; color:{BLACK}; border-color:{WHITE}; }}"
        f"QPushButton:disabled {{ color:{SOFT}; border-color:{LINE}; background:transparent; }}"
    )
    return b


def _human_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


class Bubble(QFrame):
    """会話の吹き出し1つ。"""

    MAX_WIDTH = 330

    def __init__(self, text: str, mine: bool = False) -> None:
        super().__init__()
        # コハクの発言は黒地に白枠（指定どおり）、自分の発言は枠なしの白地。
        # 白枠を持つのはコハクの吹き出しだけにして、会話の主役を分かりやすくする。
        if mine:
            frame = f"background:{WHITE}; border:0;"
            fg = BLACK
        else:
            frame = f"background:{BLACK}; border:2px solid {WHITE};"
            fg = WHITE
        self.setStyleSheet(f"QFrame {{ {frame} border-radius:0px; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        label = QLabel(_stylize(text))
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet(
            f"color:{fg}; font-size:15px; font-family:{VOICE}; background:transparent; border:0;"
        )
        layout.addWidget(label)
        self.content_layout = layout
        self.setMaximumWidth(self.MAX_WIDTH)

        # 折り返し付きのQLabelは、横に余裕があっても最小幅まで縮んで早く折り返す。
        # 一番長い行の幅を測り、吹き出しの上限までは1行で収まるように幅を確保する。
        plain = QTextDocument()
        plain.setHtml(text)
        metrics = label.fontMetrics()
        longest = max(
            (metrics.horizontalAdvance(line) for line in plain.toPlainText().splitlines()),
            default=0,
        )
        label.setMinimumWidth(min(longest + 16, self.MAX_WIDTH - 28))


class FileCard(QFrame):
    """ファイル1件のカード。開く / 別のカテゴリへ / 場所を表示。"""

    def __init__(self, file: dict, panel: "ChatPanel") -> None:
        super().__init__()
        self.file = file
        self.panel = panel
        self.setStyleSheet(
            f"QFrame#card {{ background:{SUNK}; border:0; border-radius:0px; }}"
        )
        self.setObjectName("card")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        badge = QLabel({"pdf": "PDF", "photo": "IMG", "text": "TXT"}.get(file["filetype"], "?"))
        badge.setFixedSize(42, 42)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 種別は色ではなく文字で見分ける。白地に黒のタグにして目立たせる。
        badge.setStyleSheet(
            f"background:{WHITE}; color:{BLACK}; border:0; border-radius:0px;"
            f" font-size:13px; font-family:{VOICE};"
        )
        outer.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

        body = QVBoxLayout()
        body.setSpacing(5)

        name = QLabel(file["filename"])
        name.setWordWrap(True)
        name.setStyleSheet(
            f"color:{WHITE}; font-size:13px; font-weight:600; font-family:{TEXT}; border:0;"
        )
        body.addWidget(name)

        meta = QLabel(f"{file['category']} ・ {_human_size(file['file_size'])}")
        meta.setStyleSheet(f"color:{DIM}; font-size:11px; font-family:{TEXT}; border:0;")
        body.addWidget(meta)

        summary = QLabel(file["summary"])
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color:{WHITE}; font-size:12px; font-family:{TEXT}; border:0;")
        body.addWidget(summary)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        open_btn = _button("開く", "primary")
        open_btn.clicked.connect(self._open)
        actions.addWidget(open_btn)
        # 分類の訂正はカードからも触れるようにしておく。
        # 通知の吹き出しは5秒で消えるので、そこにしか入口が無いと訂正できない。
        recat_btn = _button("別のカテゴリへ")
        recat_btn.clicked.connect(
            lambda: panel.offer_recategorize(file["id"], file["category"])
        )
        actions.addWidget(recat_btn)
        reveal_btn = _button("場所を表示")
        reveal_btn.clicked.connect(lambda: t7.reveal_file(file))
        actions.addWidget(reveal_btn)
        # T9: 検索結果のカードからも削除できるようにする。
        # 即時削除ではなくごみ箱(.trash)へ入れるだけなので、間違えても「元に戻す」で戻せる。
        delete_btn = _button("削除")
        delete_btn.clicked.connect(lambda: panel.delete_file(file["id"], self))
        actions.addWidget(delete_btn)
        actions.addStretch(1)
        body.addLayout(actions)

        outer.addLayout(body, 1)

    def _open(self) -> None:
        """既定アプリに投げる。ビューアは持たないので、ここで会話は引っ込める。"""
        if t7.open_file(self.file):
            self.panel.hide_panel()
            # 既定アプリのウィンドウは少し遅れて前面に出てくるので、
            # その後にもう一度キャラを押し上げる（1回だけだと裏に隠れる）
            mascot = self.panel.state.mascot
            for delay in (400, 1200, 2500):
                QTimer.singleShot(delay, mascot.ensure_on_top)
        else:
            self.panel.say("あれ、そのファイルが見つからない。<br>外で移動か削除をされたのかも。")


class CleanupCard(QFrame):
    """
    片付けタイムの候補1件。「残す」「ごみ箱へ」の二択だけを置く。

    選択肢を増やすと手が止まるので、ここには他のボタンを足さない。
    中身を確かめたいときはファイル名を押して開く。
    """

    def __init__(self, candidate: "t8.Candidate", on_decide) -> None:
        super().__init__()
        file = candidate.file
        self.file = file
        self.setStyleSheet(
            f"QFrame#card {{ background:{SUNK}; border:0; border-radius:0px; }}"
        )
        self.setObjectName("card")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        badge = QLabel({"pdf": "PDF", "photo": "IMG", "text": "TXT"}.get(file["filetype"], "?"))
        badge.setFixedSize(42, 42)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            f"background:{WHITE}; color:{BLACK}; border:0; border-radius:0px;"
            f" font-size:13px; font-family:{VOICE};"
        )
        outer.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

        body = QVBoxLayout()
        body.setSpacing(5)

        # ファイル名はリンクにして、押すと既定アプリで開く（中身を見てから決められるように）。
        # 開くと last_accessed_at が更新され、次回からは未アクセスの候補に出なくなる。
        name = QLabel(f"<a href='open' style='color:{WHITE};'>{file['filename']}</a>")
        name.setWordWrap(True)
        name.setTextFormat(Qt.TextFormat.RichText)
        name.setToolTip("押すと開いて中身を確かめられます")
        name.setStyleSheet(
            f"color:{WHITE}; font-size:13px; font-weight:600; font-family:{TEXT}; border:0;"
        )
        name.linkActivated.connect(lambda _: t7.open_file(file))
        body.addWidget(name)

        meta = QLabel(f"{file['category']} ・ {_human_size(file['file_size'])}")
        meta.setStyleSheet(f"color:{DIM}; font-size:11px; font-family:{TEXT}; border:0;")
        body.addWidget(meta)

        # 候補になった理由。これを読めば判断できるように、要約より先に置く
        reasons = QLabel(" / ".join(candidate.reasons))
        reasons.setWordWrap(True)
        reasons.setStyleSheet(f"color:{WHITE}; font-size:14px; font-family:{VOICE}; border:0;")
        body.addWidget(reasons)

        summary = QLabel(file["summary"])
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color:{DIM}; font-size:12px; font-family:{TEXT}; border:0;")
        body.addWidget(summary)

        self.actions = QWidget()
        row = QHBoxLayout(self.actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        keep_btn = _button("残す")
        keep_btn.clicked.connect(lambda: self._decide(on_decide, "keep"))
        row.addWidget(keep_btn)
        trash_btn = _button("ごみ箱へ", "primary")
        trash_btn.clicked.connect(lambda: self._decide(on_decide, "trash"))
        row.addWidget(trash_btn)
        row.addStretch(1)
        body.addWidget(self.actions)

        outer.addLayout(body, 1)

    def _decide(self, on_decide, choice: str) -> None:
        # 二度押しで2回処理されないよう、押した時点でボタンを消す
        self.actions.hide()
        on_decide(self, choice)


# ---------------------------------------------------------------------------
# チャットパネル
# ---------------------------------------------------------------------------

class ChatPanel(QWidget):
    """キャラの隣に出る会話パネル。検索も訂正も片付けもここに流れる。"""

    def __init__(self, app_state: "Kohaku") -> None:
        super().__init__()
        self.state = app_state
        self.threads: list[QThread] = []
        self.greeted = False
        self.cleanup: Optional[dict] = None   # 片付けタイムの途中経過。閉じたら None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(400, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        frame = QFrame()
        frame.setObjectName("panel")
        frame.setStyleSheet(
            f"QFrame#panel {{ background:{BLACK}; border:2px solid {WHITE}; border-radius:0px; }}"
        )
        root.addWidget(frame)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        inner.addWidget(self._build_header())

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet(
            f"QScrollArea {{ background:{BLACK}; border:0; }}"
            f"QScrollBar:vertical {{ background:{BLACK}; width:8px; margin:0; }}"
            f"QScrollBar::handle:vertical {{ background:{WHITE}; min-height:24px; }}"
            f"QScrollBar::add-line, QScrollBar::sub-line {{ height:0; }}"
            f"QScrollBar::add-page, QScrollBar::sub-page {{ background:{BLACK}; }}"
        )
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.log_widget = QWidget()
        self.log_widget.setObjectName("log")
        # セレクタ無しで background を書くと子のラベルにまで継承され、
        # カードの面の上に黒い帯が出る。自分自身だけに効かせる。
        self.log_widget.setStyleSheet(f"QWidget#log {{ background:{BLACK}; }}")
        self.log = QVBoxLayout(self.log_widget)
        self.log.setContentsMargins(16, 16, 16, 16)
        self.log.setSpacing(12)
        self.log.addStretch(1)
        self.scroll.setWidget(self.log_widget)
        inner.addWidget(self.scroll, 1)

        # 新しい発言が入るたびに最下部へ追従する。
        # 吹き出しは折り返しで高さが後から決まるので、追加直後にスクロールしても
        # まだ伸びきっておらず届かない。高さが確定した瞬間（rangeChanged）に動かす。
        bar = self.scroll.verticalScrollBar()
        bar.rangeChanged.connect(self._follow_bottom)
        bar.valueChanged.connect(self._check_follow)
        self._follow = True

        inner.addWidget(self._build_composer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header.setStyleSheet(
            f"QWidget#header {{ background:{BLACK}; border-bottom:1px solid {LINE}; }}"
        )
        row = QHBoxLayout(header)
        row.setContentsMargins(16, 12, 12, 12)

        title = QLabel("コハク")
        title.setStyleSheet(f"color:{WHITE}; font-size:18px; font-family:{VOICE};")
        row.addWidget(title)

        self.status = QLabel("ダウンロードフォルダを見張り中")
        self.status.setStyleSheet(f"color:{DIM}; font-size:11px; font-family:{TEXT};")
        row.addWidget(self.status)
        row.addStretch(1)

        close = QPushButton("×")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setFixedSize(24, 24)
        close.setStyleSheet(
            f"QPushButton {{ border:0; color:{WHITE}; font-size:15px; background:transparent;"
            f" font-family:{FONT}; }}"
            f"QPushButton:hover {{ background:{WHITE}; color:{BLACK}; }}"
        )
        close.clicked.connect(self.hide_panel)
        row.addWidget(close)
        return header

    def _build_composer(self) -> QWidget:
        box = QWidget()
        box.setObjectName("composer")
        box.setStyleSheet(
            f"QWidget#composer {{ background:{BLACK}; border-top:1px solid {LINE}; }}"
        )
        row = QHBoxLayout(box)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(8)

        self.input = QLineEdit()
        self.input.setPlaceholderText("探しものを言葉で（例：先週の統計学の資料）")
        self.input.setStyleSheet(
            f"QLineEdit {{ background:{SUNK}; border:1px solid {SOFT}; border-radius:0px;"
            f" padding:8px 10px; font-size:13px; color:{WHITE}; font-family:{TEXT};"
            f" selection-background-color:{WHITE}; selection-color:{BLACK}; }}"
            f"QLineEdit:focus {{ border-color:{WHITE}; }}"
        )
        self.input.returnPressed.connect(self._submit)
        row.addWidget(self.input, 1)

        send = QPushButton("▶")
        send.setCursor(Qt.CursorShape.PointingHandCursor)
        send.setFixedSize(36, 36)
        send.setStyleSheet(
            f"QPushButton {{ background:{WHITE}; color:{BLACK}; border:0; border-radius:0px;"
            f" font-size:12px; }}"
            f"QPushButton:pressed {{ background:{DIM}; }}"
        )
        send.clicked.connect(self._submit)
        row.addWidget(send)
        return box

    # --- 会話ログの操作 -----------------------------------------------------

    def _append(self, widget: QWidget, align_right: bool = False) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if align_right:
            row.addStretch(1)
            row.addWidget(widget)
        else:
            row.addWidget(widget)
            row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        self.log.insertWidget(self.log.count() - 1, holder)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return widget

    def _append_wide(self, widget: QWidget) -> QWidget:
        self.log.insertWidget(self.log.count() - 1, widget)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return widget

    def _scroll_to_bottom(self) -> None:
        self._follow = True
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _follow_bottom(self) -> None:
        """ログの高さが変わったとき、追従中なら最下部へ。"""
        if not self._follow:
            return
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _check_follow(self, value: int) -> None:
        """自分で上へスクロールしたら追従をやめる。下端に戻したら再開する。"""
        bar = self.scroll.verticalScrollBar()
        self._follow = value >= bar.maximum() - 8

    def say(self, text: str) -> Bubble:
        """コハク側の発言。"""
        return self._append(Bubble(text))

    def say_user(self, text: str) -> Bubble:
        return self._append(Bubble(text, mine=True), align_right=True)

    def add_chips(self, labels: list[str], on_click) -> None:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for label in labels:
            chip = _button(label)
            chip.clicked.connect(lambda _, v=label: on_click(v))
            row.addWidget(chip)
        row.addStretch(1)
        self._append_wide(holder)

    def add_choices(self, labels: list[str], on_click) -> None:
        """
        縦に並べる選択肢。カテゴリの階層名（「経理／経費精算／出張精算」）は長いので、
        横並びのチップだと画面からはみ出す。
        """
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        for label in labels:
            choice = _button(label)
            choice.setStyleSheet(choice.styleSheet() + "QPushButton { text-align: left; }")
            choice.clicked.connect(lambda _, v=label: on_click(v))
            column.addWidget(choice)
        self._append_wide(holder)

    def add_cards(self, files: list[dict]) -> None:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(8)
        for file in files:
            column.addWidget(FileCard(file, self))
        self._append_wide(holder)

    def set_status(self, text: str, revert_after: int = 4000) -> None:
        self.status.setText(text)
        if revert_after:
            QTimer.singleShot(
                revert_after, lambda: self.status.setText("ダウンロードフォルダを見張り中")
            )

    # --- 検索 ---------------------------------------------------------------

    def _submit(self) -> None:
        query = self.input.text().strip()
        if not query:
            return
        self.input.clear()
        self.ask(query)

    def ask(self, query: str) -> None:
        self.say_user(query)
        self.set_status("探しています…", 0)

        thread = SearchThread(query)
        thread.done.connect(lambda result: self._show_result(query, result))
        thread.finished.connect(lambda: self.threads.remove(thread))
        self.threads.append(thread)
        thread.start()

    def _show_result(self, query: str, result) -> None:
        self.set_status("ダウンロードフォルダを見張り中", 0)

        if result is None:
            self.say("検索でつまずいた。もう一度試してみて。")
            return
        if not result.hits:
            if result.filters:
                self.say(
                    f"{' / '.join(result.filters)} で絞ってみたけど、見つからなかった。<br>"
                    "期間を外すか、別の言い方で言ってみて。"
                )
            else:
                self.say("うーん、それらしいファイルが見つからなかった。<br>別の言い方で言ってみて。")
            return

        head = f"これかな。{len(result.hits)}件あったよ。"
        if result.filters:
            head += f"<br>[{' / '.join(result.filters)}] で絞り込んでいます。"
        self.say(head)
        self.add_cards([hit.file for hit in result.hits])

    # --- 分類の訂正 ---------------------------------------------------------

    NEW_CATEGORY = "新しく作る"

    def offer_recategorize(self, file_id: int, current_category: str) -> None:
        """
        通知の「違うカテゴリ」から呼ばれる。
        AIの分類は必ず外れるので、その場で入れ直せる逃げ道を用意しておく。

        候補は「同じ内容のファイルが既に入っているカテゴリ」を近い順に並べる。
        当てはまるものが無いこともあるので、必ず「新しく作る」を最後に足す。
        """
        self.show_panel()
        self.say(f"いまは「{current_category}」に入れてあるよ。<br>どこに入れ直す？")

        candidates = t7.suggest_categories(file_id)
        self.add_choices(
            candidates + [self.NEW_CATEGORY],
            lambda name: self._pick_category(file_id, name),
        )

    TOP_LEVEL = "いちばん上に作る"

    def _pick_category(self, file_id: int, category: str) -> None:
        if category == self.NEW_CATEGORY:
            # 階層のどこに作るかを先に選んでもらう。最下層の下には作れない。
            parents = sorted(
                name for name in db.get_category_names()
                if len(name.split(CATEGORY_SEPARATOR)) < MAX_CATEGORY_DEPTH
            )
            self.say("どこの中に作る？")
            self.add_choices(
                [self.TOP_LEVEL] + parents,
                lambda parent: self._create_category(file_id, parent),
            )
            return
        self._move_to(file_id, category)

    def _create_category(self, file_id: int, parent: str) -> None:
        prefix = "" if parent == self.TOP_LEVEL else parent + CATEGORY_SEPARATOR
        name, ok = QInputDialog.getText(
            self, "新しいカテゴリ",
            f"{prefix or '（いちばん上）'} の中に作るカテゴリ名\n"
            "「/」で区切ると、さらに下の階層までまとめて作れます",
        )
        # 「/」「>」でも階層を区切れるようにする（全角の区切り文字は打ちにくい）
        parts = [p.strip() for p in re.split(r"[/／>＞]", name if ok else "") if p.strip()]
        if not parts:
            self.say("そのままにしておくね。")
            return
        full = prefix + CATEGORY_SEPARATOR.join(parts)
        if len(full.split(CATEGORY_SEPARATOR)) > MAX_CATEGORY_DEPTH:
            self.say(f"階層は{MAX_CATEGORY_DEPTH}段までにしてね。")
            return
        # 途中の階層も含めて categories に登録し、既存と表記が揺れていれば揃える。
        # 登録しないと、次の分類でT4のプロンプトに候補として出てこない。
        self._move_to(file_id, resolve_category(full))

    def _move_to(self, file_id: int, category: str) -> None:
        moved = t6.recategorize(file_id, category)
        if not moved:
            self.say("移せなかった。ファイルが見当たらない。")
            return

        self.say(f"「{category}」に移したよ。")
        file = db.get_file(file_id)
        if file is not None:
            self.add_cards([file])
        self.add_chips([self.UNDO], lambda _: self.undo_last_action(file_id))

    # --- 削除・Undo（T9） -----------------------------------------------------

    UNDO = "元に戻す"

    def delete_file(self, file_id: int, card: Optional[QWidget] = None) -> None:
        """
        検索結果のカードから「削除」を選んだときに呼ばれる。
        即時削除ではなく、T8と同じ trash_file() でごみ箱(.trash)へ移すだけなので
        間違えてもすぐ「元に戻す」で復元できる。
        """
        file = db.get_file(file_id)
        if file is None:
            self.say("あれ、そのファイルが見当たらない。")
            return

        destination = t8.trash_file(file_id)
        if destination is None:
            self.say("削除できなかった。<br>外で移動されたか、既にごみ箱に入っているのかも。")
            return

        if card is not None:
            card.setParent(None)
            card.deleteLater()

        self.say(f"<b>{file['filename']}</b> をごみ箱へ入れたよ。")
        # 間に別のダウンロードが分類されても、このファイルの削除だけを戻す
        self.add_chips([self.UNDO], lambda _: self.undo_last_action(file_id))

    def undo_last_action(self, file_id: Optional[int] = None) -> None:
        """直前の移動・削除を1件取り消す（T9）。file_id を渡すとそのファイルの操作に限る。"""
        result = t9_undo.undo_last_action(file_id)
        if result is None:
            self.say("元に戻せる操作は無いよ。")
            return

        file = db.get_file(result["file_id"])
        name = file["filename"] if file else Path(result["original_path"]).name
        label = "削除" if result["action_type"] == "delete" else "移動"
        self.say(f"<b>{name}</b> の{label}を取り消したよ。")
        if file is not None:
            self.add_cards([file])

    def undo_from_menu(self) -> None:
        """右クリックメニュー・トレイメニューの「元に戻す」から呼ばれる。"""
        self.show_panel(greet=False)
        self.undo_last_action()

    # --- 片付けタイム（T8） ---------------------------------------------------

    START = "はじめる"
    LATER = "またこんど"
    CONTINUE = "続ける"

    @staticmethod
    def invite_text(count: int) -> str:
        return f"<b>{count}件</b> 片付けられそうなのがあるよ。<br>片付けタイムにしない？"

    def offer_cleanup(self, count: int) -> None:
        """声かけを会話ログに残す。吹き出しが消えた後でも、ここから始められる。"""
        self.say(self.invite_text(count))
        self.add_chips([self.START, self.LATER], self._answer_invite)

    def _answer_invite(self, answer: str) -> None:
        if answer in (self.START, self.CONTINUE):
            self.start_cleanup()
        else:
            self.say("じゃあ、またこんどね。")

    def invite_from_menu(self) -> None:
        """右クリックメニューから。声かけを経ていないので、まず誘うところから。"""
        self.show_panel()
        candidates = t8.find_candidates()
        if not candidates:
            self.say("いまは片付けるものは無いよ。えらい。")
            return
        self.offer_cleanup(len(candidates))

    def start_cleanup(self) -> None:
        # 誘いに応じた流れの途中なので、「おかえり」の挨拶は挟まない
        self.show_panel(greet=False)
        # 声かけから時間がたっているかもしれないので、押した時点で算出し直す
        candidates = t8.find_candidates()
        if not candidates:
            self.say("いまは片付けるものは無いよ。えらい。")
            return
        self.cleanup = {"queue": candidates, "shown": 0, "trashed": 0, "kept": 0, "size": 0}
        self.say("よし、1件ずつ見ていこう。<br>「残す」か「ごみ箱へ」で答えてね。")
        self._next_candidate()

    def _next_candidate(self) -> None:
        session = self.cleanup
        if session is None:
            return
        if not session["queue"] or session["shown"] >= CLEANUP_BATCH:
            self._finish_cleanup()
            return
        candidate = session["queue"].pop(0)
        session["shown"] += 1
        self._append_wide(
            CleanupCard(
                candidate,
                lambda card, choice, s=session: self._decide_cleanup(s, card, choice),
            )
        )

    def _decide_cleanup(self, session: dict, card: CleanupCard, choice: str) -> None:
        # パネルを閉じて終わった回のカードが後から押されても、処理しない
        if session is not self.cleanup:
            self.say("この回はもう終わってるよ。<br>右クリックの「片付けタイム」からまた始めてね。")
            return
        file = card.file
        if choice == "keep":
            t8.keep_file(file["id"])
            session["kept"] += 1
            self.say(f"{file['filename']} は残しておくね。<br>しばらくは聞かないよ。")
        elif t8.trash_file(file["id"]) is None:
            self.say("あれ、そのファイルが見つからない。<br>外で移動か削除をされたのかも。")
        else:
            session["trashed"] += 1
            session["size"] += file["file_size"]
            self.say(f"{file['filename']} をごみ箱へ入れたよ。")
        self._next_candidate()

    def _finish_cleanup(self) -> None:
        session = self.cleanup
        self.cleanup = None
        if session is None:
            return
        parts = []
        if session["trashed"]:
            parts.append(f"ごみ箱へ {session['trashed']}件")
        if session["kept"]:
            parts.append(f"残す {session['kept']}件")
        text = "おつかれさま。"
        if parts:
            text += f"<br><b>{' / '.join(parts)}</b>"
        if session["trashed"]:
            text += f"<br>{_human_size(session['size'])} 片付いた。"
        self.say(text)

        rest = len(session["queue"])
        if rest:
            self.say(f"まだ <b>{rest}件</b> あるよ。続ける？")
            self.add_chips([self.CONTINUE, self.LATER], self._answer_invite)

    # --- 表示制御 -----------------------------------------------------------

    def greet_if_empty(self) -> None:
        # ログの件数で判定すると、パネルを開く前に流れた声かけや通知で挨拶が飛ぶ
        if self.greeted:
            return
        self.greeted = True
        files = db.get_active_files()
        if files:
            categories = sorted({f["category"] for f in files})[:3]
            self.say(
                f"おかえり。いま <b>{len(files)}件</b> 預かってるよ。<br>"
                f"よく使うのは「{'」「'.join(categories)}」あたり。"
            )
        else:
            self.say("おかえり。まだ何も預かっていないよ。<br>ダウンロードすれば勝手に片付けておくね。")
        self.add_chips(
            ["先週の資料", "契約書どこだっけ", "今日入れたやつ"], self.ask
        )

    def show_panel(self, greet: bool = True) -> None:
        self.state.hide_toast()
        self.move(self.state.panel_position())
        self.show()
        self.raise_()
        self.activateWindow()
        if greet:
            self.greet_if_empty()
        self.input.setFocus()
        self.state.mascot.hide()

    def hide_panel(self) -> None:
        # 片付けタイムは途中で閉じたらその回は終わり（次に開いても続きからは再開しない）
        self.cleanup = None
        self.hide()
        self.state.mascot.show()
        self.state.mascot.ensure_on_top()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide_panel()
        else:
            super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# 吹き出し（自発通知）
# ---------------------------------------------------------------------------

class Toast(QWidget):
    """キャラの横に出る通知。5秒で消える。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(300)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.frame = QFrame()
        self.frame.setObjectName("toast")
        self.frame.setStyleSheet(
            f"QFrame#toast {{ background:{BLACK}; border:2px solid {WHITE}; border-radius:0px; }}"
        )
        root.addWidget(self.frame)

        self.box = QVBoxLayout(self.frame)
        self.box.setContentsMargins(14, 12, 14, 12)
        self.box.setSpacing(10)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setStyleSheet(
            f"color:{WHITE}; font-size:15px; font-family:{VOICE}; border:0; background:transparent;"
        )
        self.box.addWidget(self.label)

        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        self.box.addLayout(self.actions)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_message(self, html: str, actions: list[tuple[str, str, object]], at: QPoint) -> None:
        self.label.setText(_stylize(html))

        while self.actions.count():
            item = self.actions.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for label, kind, handler in actions:
            button = _button(label, kind)
            button.clicked.connect(handler)
            self.actions.addWidget(button)
        self.actions.addStretch(1)

        self.frame.layout().activate()
        self.label.setFixedHeight(self.label.heightForWidth(self.label.width()))
        self.adjustSize()
        self.move(at)
        self.show()
        self.raise_()
        self.timer.start(TOAST_MS)


# ---------------------------------------------------------------------------
# 常駐キャラ
# ---------------------------------------------------------------------------

# ドット絵。 "." = 透明 / "#" = 黒 / "o" = 白 / "g" = 灰（ほっぺ・舌）。
# 白い体に黒い縁取りなので、明るい壁紙でも暗い壁紙でも輪郭が消えない。
#
# 可愛く見せるための約束事:
# - 輪郭は丸く、下ぶくれ。小さな足を2つ出す
# - 目は顔の低め・離れ気味に置き、ほっぺを付ける
# - 口は普段は小さく、食べるときだけ顔の半分まで大きく開く（ファイルを食べるキャラ）

_BODY_TOP = [
    ".......####.......",
    ".....##oooo##.....",
    "....#oooooooo#....",
    "...#oooooooooo#...",
    "..#oooooooooooo#..",
    ".#oooooooooooooo#.",
    ".#oooooooooooooo#.",
]
_BODY_BOTTOM = [
    ".#oooooooooooooo#.",
    "..#oooooooooooo#..",
    "...#oo######oo#...",
    "...####....####...",
]

# 普段: 目はぱっちり、口は小さな「ω」
SPRITE_IDLE = _BODY_TOP + [
    "#oooo##oooo##oooo#",
    "#oooo##oooo##oooo#",
    "#ooggoo#oo#ooggoo#",
    "#ooooooo##ooooooo#",
    "#oooooooooooooooo#",
] + _BODY_BOTTOM

# まばたき: 目が線になる
SPRITE_BLINK = _BODY_TOP + [
    "#oooooooooooooooo#",
    "#oooo##oooo##oooo#",
    "#ooggoo#oo#ooggoo#",
    "#ooooooo##ooooooo#",
    "#oooooooooooooooo#",
] + _BODY_BOTTOM

# 満足: 食べ終わったあと。目が「^ ^」
SPRITE_HAPPY = _BODY_TOP + [
    "#oooo#oooooo#oooo#",
    "#ooo#o#oooo#o#ooo#",
    "#ooggoo#oo#ooggoo#",
    "#ooooooo##ooooooo#",
    "#oooooooooooooooo#",
] + _BODY_BOTTOM

# 食べる途中: 口が開いて舌が見える
SPRITE_OPEN = _BODY_TOP + [
    "#oooo##oooo##oooo#",
    "#oooo##oooo##oooo#",
    "#ooggo######oggoo#",
    "#ooooo##gg##ooooo#",
    "#oooooo####oooooo#",
] + _BODY_BOTTOM

# 食べる瞬間: 目を「> <」にして、顔の半分まで口をあんぐり開ける
SPRITE_WIDE = [
    ".......####.......",
    ".....##oooo##.....",
    "....#oooooooo#....",
    "...#oooooooooo#...",
    "..#oooooooooooo#..",
    ".#oooooooooooooo#.",
    ".#oo#oooooooo#oo#.",
    "#oooo#oooooo#oooo#",
    "#ooo#oooooooo#ooo#",
    "#o##############o#",
    "#o#oo########oo#o#",
    "#o##############o#",
    ".#o####gggg####o#.",
    "..#o##########o#..",
    "...#oo######oo#...",
    "...####....####...",
]

# 食べられる側のファイル（右上の角が折れた紙）
SPRITE_FILE = [
    "####..",
    "#oo#o.",
    "#oo###",
    "#oooo#",
    "#o##o#",
    "#oooo#",
    "######",
]

# 呼吸の上下（ドット単位）。ゆっくり上がって、ゆっくり下がる。
BREATH_PATTERN = [0, 0, 0, 1, 1, 1]

# まばたきの間隔（tick数）。一定だと機械的に見えるので2種類を交互に使う。
BLINK_EVERY = (31, 23)

# 食べるときの1サイクル。ファイルが口に近づく間は口を開け続け、
# 飲み込んだ瞬間に閉じて、もぐもぐしてからまた開く。
CHOMP_STEPS = 12
CHOMP_SWALLOW_AT = 7   # このステップでファイルが消えて口が閉じる

# 食べ終わったあと「^ ^」の顔でいる時間（tick数）
HAPPY_TICKS = 14

_SPRITE_COLORS = {"#": BLACK, "o": WHITE, "g": SPRITE_GRAY}


def _draw_sprite(painter: QPainter, sprite: list[str], ox: int, oy: int, px: int) -> None:
    """文字列のドット絵を px 四方のブロックで描く。"""
    colors = {key: QColor(value) for key, value in _SPRITE_COLORS.items()}
    for y, row in enumerate(sprite):
        for x, cell in enumerate(row):
            if cell == ".":
                continue
            painter.fillRect(ox + x * px, oy + y * px, px, px, colors[cell])


class Mascot(QWidget):
    """
    枠なし・背景透過・常に最前面の小さなウィンドウ。
    クリックで会話パネル、ドラッグで移動、右クリックでメニュー。
    """

    clicked = Signal()

    PIXEL = 4                                  # ドット1つの大きさ（画面上のpx）
    BODY_W = len(SPRITE_IDLE[0]) * PIXEL       # 体の幅
    BODY_H = len(SPRITE_IDLE) * PIXEL          # 体の高さ
    # 右側は食べられるファイルが飛んでくるための余白。上は呼吸で浮くぶんの余白。
    WIDTH = BODY_W + 8 * PIXEL
    HEIGHT = BODY_H + 2 * PIXEL
    TICK_MS = 120

    def __init__(self, app_state: "Kohaku") -> None:
        super().__init__()
        self.state = app_state
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._drag_origin: Optional[QPoint] = None
        self._moved = False
        self.working = False

        # 他のアプリを開くと、最前面指定があってもその下に潜ってしまうことがある。
        # （特にファイルを既定アプリで開いた直後。相手が前面を奪う）
        # 定期的に最前面へ押し上げ直す。フォーカスは奪わない。
        self.topmost_timer = QTimer(self)
        self.topmost_timer.timeout.connect(self.ensure_on_top)
        self.topmost_timer.start(1500)

        # アニメーション。普段は呼吸とまばたき、分類中はファイルを食べる。
        self.tick = 0
        self.happy_until = -1     # このtickまでは食べ終わりの「^ ^」顔
        self.next_blink = BLINK_EVERY[0]
        self.blink_turn = 0
        self.animation = QTimer(self)
        self.animation.timeout.connect(self._tick)
        self.animation.start(self.TICK_MS)

    def ensure_on_top(self) -> None:
        """
        フォーカスを奪わずに最前面へ戻す。

        raise_() はウィンドウをアクティブ化しようとして、入力中のアプリから
        フォーカスを取り上げてしまうことがある。Windowsでは SetWindowPos を
        SWP_NOACTIVATE 付きで直接呼び、「最前面に置くが触らない」を実現する。
        """
        if not self.isVisible():
            return
        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                # argtypes を宣言しないと HWND_TOPMOST(-1) が64bitに符号拡張されず、
                # 例外も出ないまま「無効なハンドル」で毎回失敗する。
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                user32.SetWindowPos.argtypes = [
                    wintypes.HWND, wintypes.HWND,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    wintypes.UINT,
                ]
                user32.SetWindowPos.restype = wintypes.BOOL

                HWND_TOPMOST = -1
                SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
                ok = user32.SetWindowPos(
                    int(self.winId()), HWND_TOPMOST, 0, 0, 0, 0,
                    SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE,
                )
                if not ok:
                    # raise_() に落とすと1.5秒ごとにフォーカスを奪いかねないので、記録だけする
                    logger.debug(f"[UI] SetWindowPosに失敗しました: エラー {ctypes.get_last_error()}")
                return
            except Exception as e:  # ctypesが使えない環境ではQtの方法に落とす
                logger.debug(f"[UI] SetWindowPosに失敗しました: {e}")
        self.raise_()

    def _tick(self) -> None:
        self.tick += 1
        self.update()

    def set_working(self, working: bool) -> None:
        if working and not self.working:
            self.tick = 0  # 食べ始めは必ずサイクルの頭から
        if self.working and not working:
            # 食べ終わった。しばらく満足した顔をする
            self.tick = 0
            self.happy_until = HAPPY_TICKS
            self.next_blink = HAPPY_TICKS + BLINK_EVERY[0]
        self.working = working
        self.update()

    def _frame(self) -> tuple[list[str], int, Optional[int]]:
        """
        いまのコマを決める。

        Returns:
            (体のドット絵, 体の上下オフセット[ドット], ファイルのx座標[px] or None)
        """
        if not self.working:
            breath = BREATH_PATTERN[(self.tick // 4) % len(BREATH_PATTERN)]
            if self.tick <= self.happy_until:
                # 満足顔のあいだは少し弾む
                return SPRITE_HAPPY, 1 if self.tick % 4 < 2 else 0, None
            if self.tick >= self.next_blink:
                # まばたきは1コマだけ
                self.blink_turn += 1
                self.next_blink = self.tick + BLINK_EVERY[self.blink_turn % len(BLINK_EVERY)]
                return SPRITE_BLINK, breath, None
            return SPRITE_IDLE, breath, None

        step = self.tick % CHOMP_STEPS
        px = self.PIXEL
        start_x = self.WIDTH - len(SPRITE_FILE[0]) * px
        end_x = self.BODY_W - 5 * px   # 口の奥

        if step < CHOMP_SWALLOW_AT:
            # 口を開けて待ち構え、ファイルが右から吸い込まれてくる
            progress = step / (CHOMP_SWALLOW_AT - 1)
            file_x = int(start_x + (end_x - start_x) * progress)
            body = SPRITE_OPEN if step == 0 else SPRITE_WIDE
            return body, 1, file_x
        if step == CHOMP_SWALLOW_AT:
            # 飲み込んだ瞬間。口を閉じて少し沈む
            return SPRITE_HAPPY, 0, None
        # もぐもぐ
        body = SPRITE_OPEN if step % 2 == 0 else SPRITE_IDLE
        return body, 1 if step % 2 == 0 else 0, None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        # ドット絵はにじませない
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        body, lift, file_x = self._frame()
        px = self.PIXEL
        body_y = (2 - lift) * px

        # ファイルを先に描き、体をその上に重ねる。体の縁を越えた瞬間に
        # 口の中へ消えていくように見える。
        if file_x is not None:
            file_y = body_y + 8 * px
            _draw_sprite(painter, SPRITE_FILE, file_x, file_y, px)

        _draw_sprite(painter, body, 0, body_y, px)
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.pos()
            self._moved = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.state.show_menu(event.globalPosition().toPoint())

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is not None:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            self._moved = True

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            if not self._moved:
                self.clicked.emit()


# ---------------------------------------------------------------------------
# アプリ本体
# ---------------------------------------------------------------------------

class Kohaku:
    """キャラ・パネル・通知・監視スレッドをまとめる。"""

    def __init__(self, app: QApplication, watch_dir: str) -> None:
        self.app = app
        self.watch_dir = watch_dir

        db.init_db()

        self.mascot = Mascot(self)
        self.panel = ChatPanel(self)
        self.toast = Toast()

        self.mascot.clicked.connect(self.panel.show_panel)

        self._place_mascot()
        self.mascot.show()

        self.tray = self._build_tray()

        self.watcher = WatcherThread(watch_dir)
        self.watcher.progressed.connect(self._on_progress)
        self.watcher.classified.connect(self._on_classified)
        self.watcher.failed.connect(self._on_failed)
        self.watcher.start()

        # 片付けタイムの声かけ。起動直後は作業の邪魔なので少し待ってから
        self.last_invite: Optional[datetime] = None
        QTimer.singleShot(INVITE_FIRST_DELAY_MS, self.maybe_invite_cleanup)

    def _place_mascot(self) -> None:
        """画面左下に置く。タスクバーを避けるため作業領域を基準にする。"""
        screen = self.app.primaryScreen().availableGeometry()
        self.mascot.move(screen.left() + 24, screen.bottom() - Mascot.HEIGHT - 16)

    def panel_position(self) -> QPoint:
        """パネルはキャラの上に重ねて出す（キャラは隠れる）。"""
        screen = self.app.primaryScreen().availableGeometry()
        x = max(screen.left() + 12, self.mascot.x())
        y = self.mascot.y() + Mascot.HEIGHT - self.panel.height()
        y = max(screen.top() + 12, y)
        return QPoint(x, y)

    def toast_position(self) -> QPoint:
        # 体の右隣に出す（ファイルが飛んでくる余白の上に重ねる）
        return QPoint(self.mascot.x() + Mascot.BODY_W + 8, self.mascot.y() - 24)

    def hide_toast(self) -> None:
        self.toast.hide()

    def _build_tray(self) -> QSystemTrayIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        _draw_sprite(painter, SPRITE_IDLE, -2, 0, 2)
        painter.end()
        tray = QSystemTrayIcon(QIcon(pixmap))
        tray.setToolTip("コハク")

        menu = QMenu()
        open_action = QAction("コハクと話す", menu)
        open_action.triggered.connect(self.panel.show_panel)
        menu.addAction(open_action)
        tidy_action = QAction("片付けタイム", menu)
        tidy_action.triggered.connect(self.panel.invite_from_menu)
        menu.addAction(tidy_action)
        undo_action = QAction("元に戻す", menu)
        undo_action.triggered.connect(self.panel.undo_from_menu)
        menu.addAction(undo_action)
        menu.addSeparator()
        quit_action = QAction("終了", menu)
        quit_action.triggered.connect(self.app.quit)
        menu.addAction(quit_action)

        # キャラに被さって隠れないよう、開いている間は最前面への押し上げを止める（show_menu と同じ）
        menu.aboutToShow.connect(self.mascot.topmost_timer.stop)
        menu.aboutToHide.connect(self.mascot.topmost_timer.start)
        tray.setContextMenu(menu)
        tray.activated.connect(lambda _: self.panel.show_panel())
        tray.show()
        return tray

    def show_menu(self, at: QPoint) -> None:
        menu = QMenu()
        talk = QAction("コハクと話す", menu)
        talk.triggered.connect(self.panel.show_panel)
        menu.addAction(talk)
        tidy = QAction("片付けタイム", menu)
        tidy.triggered.connect(self.panel.invite_from_menu)
        menu.addAction(tidy)
        undo = QAction("元に戻す", menu)
        undo.triggered.connect(self.panel.undo_from_menu)
        menu.addAction(undo)
        menu.addSeparator()
        quit_action = QAction("終了", menu)
        quit_action.triggered.connect(self.app.quit)
        menu.addAction(quit_action)
        # キャラは定期的に最前面へ押し上げているので、そのままだとメニューの上に被さって
        # 項目が押せなくなる。開いている間だけ止める。
        self.mascot.topmost_timer.stop()
        try:
            menu.exec(at)
        finally:
            self.mascot.topmost_timer.start()

    # --- 監視スレッドからの通知 ---------------------------------------------

    def _on_progress(self, message: str) -> None:
        self.mascot.set_working(True)
        self.panel.set_status(message, 0)

    def _on_classified(self, result: dict) -> None:
        self.mascot.set_working(False)
        self.panel.set_status("ダウンロードフォルダを見張り中", 0)

        # 会話ログにも残す。通知が消えても、パネルを開けば履歴として読める。
        self.panel.say(
            f"<b>{result['filename']}</b> を<br>"
            f"「{result['category']}」に入れておいたよ。"
        )

        file = db.get_file(result["file_id"])
        if file is not None:
            self.panel.add_cards([file])

        self.toast.show_message(
            f"<b>{result['category']}</b> に入れといたよ。<br>{result['filename']}",
            [
                ("開く", "primary", lambda: self._open_from_toast(result["file_id"])),
                (
                    "違うカテゴリ",
                    "plain",
                    lambda: self.panel.offer_recategorize(
                        result["file_id"], result["category"]
                    ),
                ),
            ],
            self.toast_position(),
        )
        # 前回の声かけから1日たっていれば、分類の通知が消えた後に誘う
        self.maybe_invite_cleanup()

    # --- 片付けタイムの声かけ ------------------------------------------------

    def maybe_invite_cleanup(self) -> None:
        """
        候補があれば吹き出しで片付けタイムに誘う。1日1回まで。

        ダウンロードの通知と重なると両方読まれなくなるので、通知中・分類中は待つ。
        """
        if self.last_invite is not None and datetime.now() - self.last_invite < INVITE_INTERVAL:
            return
        if self.toast.isVisible() or self.mascot.working:
            QTimer.singleShot(INVITE_RETRY_MS, self.maybe_invite_cleanup)
            return

        candidates = t8.find_candidates()
        if not candidates:
            return
        # 「またこんど」を押しても、吹き出しが消えただけでも、次は24時間後
        self.last_invite = datetime.now()

        self.panel.offer_cleanup(len(candidates))
        if self.panel.isVisible():
            return
        self.toast.show_message(
            ChatPanel.invite_text(len(candidates)),
            [
                (ChatPanel.START, "primary", self._start_cleanup_from_toast),
                (ChatPanel.LATER, "plain", self.hide_toast),
            ],
            self.toast_position(),
        )

    def _start_cleanup_from_toast(self) -> None:
        self.hide_toast()
        self.panel.start_cleanup()

    def _open_from_toast(self, file_id: int) -> None:
        self.hide_toast()
        file = db.get_file(file_id)
        if file is not None:
            t7.open_file(file)

    def _on_failed(self, path: str) -> None:
        self.mascot.set_working(False)
        self.panel.set_status("ダウンロードフォルダを見張り中", 0)
        logger.info(f"[UI] 処理されなかったファイル: {path}")


def run(watch_dir: str) -> int:
    """
    GUIを起動して常駐する。アプリの入口は main.py なので、通常はそちらから呼ばれる。

    Returns:
        Qtの終了コード
    """
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # パネルを閉じても常駐を続ける
    _load_pixel_font()
    app.setFont(QFont("DotGothic16", 10))
    app.setApplicationName("コハク")

    Kohaku(app, watch_dir)

    print(f"[コハク] 監視開始: {watch_dir}")
    print("[コハク] 左下のキャラをクリックすると話せます。終了はキャラを右クリック。")

    return app.exec()


if __name__ == "__main__":
    # 単体で動かしたいとき用。通常は main.py から起動する。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "Downloads")
    if not Path(target).is_dir():
        print(f"監視対象のフォルダが見つかりません: {target}")
        sys.exit(1)
    sys.exit(run(target))
