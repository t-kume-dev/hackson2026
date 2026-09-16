"""
コハク: デスクトップ常駐の相棒UI（PySide6）

デスクトップの隅にキャラが浮かんでいて、
- ダウンロードを検出したら吹き出しで「どこに入れたか」を知らせる（自発）
- クリックするとチャットパネルが開き、自然文でファイルを探せる（ユーザー始動）
という2つの面を持つ。検索結果も分類通知も訂正も、すべて1本の会話ログに流す。

【設計の前提】
- ファイルビューアは自前で持たない。「開く」はOSの既定アプリに投げるだけ
  （エクスプローラーでダブルクリックしたのと同じ挙動）。
  この導線を通ったときだけ last_accessed_at が記録され、放置検出の根拠になる。
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
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QPoint, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QPainter,
    QPainterPath,
    QPixmap,
    QRadialGradient,
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
from pipeline import process_file
from t1_watch import watch_folder

logger = logging.getLogger(__name__)

# 見た目のトーン。モックと同じ配色。
INK = "#23262e"
DIM = "#6b7280"
PANEL = "#fbfaf8"
SUNK = "#f2efea"
LINE = "#e3e1dc"
TEAL = "#1f6f66"
AMBER = "#d9873a"
DANGER = "#b3413a"

# 通知の吹き出しが消えるまでの時間。作業中に居座られるのが一番嫌われるので短く。
TOAST_MS = 5000

FONT = '"Yu Gothic UI", "Meiryo", sans-serif'


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


class SimilarThread(QThread):
    """「似たファイル」の計算。embedding同士の内積なので速いが一応スレッドで。"""

    done = Signal(object)

    def __init__(self, file_id: int) -> None:
        super().__init__()
        self.file_id = file_id

    def run(self) -> None:
        try:
            self.done.emit(t7.find_similar(self.file_id))
        except Exception as e:
            logger.exception(f"[UI] 類似検索に失敗しました: {e}")
            self.done.emit([])


# ---------------------------------------------------------------------------
# 部品
# ---------------------------------------------------------------------------

def _button(text: str, kind: str = "plain") -> QPushButton:
    """会話ログの中に置く小さなボタン。"""
    colors = {
        "plain": (PANEL, INK, LINE),
        "primary": (TEAL, "#ffffff", TEAL),
        "danger": (PANEL, DANGER, "#e8cdcb"),
    }
    bg, fg, border = colors[kind]
    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setStyleSheet(
        f"QPushButton {{ background:{bg}; color:{fg}; border:1px solid {border};"
        f" border-radius:7px; padding:4px 10px; font-size:11px; font-family:{FONT}; }}"
        f"QPushButton:hover {{ background:{SUNK}; }}"
        f"QPushButton:disabled {{ color:#b6bac0; border-color:{LINE}; background:{PANEL}; }}"
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

    def __init__(self, text: str, mine: bool = False) -> None:
        super().__init__()
        bg, fg = (TEAL, "#ffffff") if mine else (SUNK, INK)
        radius = "13px 13px 4px 13px" if mine else "13px 13px 13px 4px"
        self.setStyleSheet(
            f"QFrame {{ background:{bg}; border-radius:13px;"
            f" border-bottom-{'right' if mine else 'left'}-radius:4px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet(f"color:{fg}; font-size:12px; font-family:{FONT}; background:transparent;")
        layout.addWidget(label)
        self.content_layout = layout
        self.setMaximumWidth(300)


class FileCard(QFrame):
    """検索結果1件。開く / 似たファイル / 場所を表示。"""

    def __init__(self, file: dict, panel: "ChatPanel") -> None:
        super().__init__()
        self.file = file
        self.panel = panel
        self.setStyleSheet(
            f"QFrame#card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:11px; }}"
        )
        self.setObjectName("card")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        badge = QLabel({"pdf": "PDF", "photo": "IMG", "text": "TXT"}.get(file["filetype"], "?"))
        badge.setFixedSize(40, 50)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color = {"pdf": "#a8453f", "photo": "#2f6f8f", "text": "#5a6570"}.get(
            file["filetype"], "#5a6570"
        )
        badge.setStyleSheet(
            f"background:{color}; color:#fff; border-radius:5px;"
            f" font-size:9px; font-weight:700; font-family:{FONT};"
        )
        outer.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

        body = QVBoxLayout()
        body.setSpacing(4)

        name = QLabel(file["filename"])
        name.setWordWrap(True)
        name.setStyleSheet(f"color:{INK}; font-size:11.5px; font-weight:700; font-family:{FONT};")
        body.addWidget(name)

        meta = QLabel(f"{file['category']} ・ {_human_size(file['file_size'])}")
        meta.setStyleSheet(f"color:{DIM}; font-size:10px; font-family:{FONT};")
        body.addWidget(meta)

        summary = QLabel(file["summary"])
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color:{DIM}; font-size:10.5px; font-family:{FONT};")
        body.addWidget(summary)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        open_btn = _button("開く", "primary")
        open_btn.clicked.connect(self._open)
        actions.addWidget(open_btn)
        similar_btn = _button("似たファイル")
        similar_btn.clicked.connect(lambda: panel.show_similar(file))
        actions.addWidget(similar_btn)
        reveal_btn = _button("場所を表示")
        reveal_btn.clicked.connect(lambda: t7.reveal_file(file))
        actions.addWidget(reveal_btn)
        actions.addStretch(1)
        body.addLayout(actions)

        outer.addLayout(body, 1)

    def _open(self) -> None:
        """既定アプリに投げる。ビューアは持たないので、ここで会話は引っ込める。"""
        if t7.open_file(self.file):
            self.panel.hide_panel()
        else:
            self.panel.say("あれ、そのファイルが見つからない。<br>外で移動か削除をされたのかも。")


# ---------------------------------------------------------------------------
# チャットパネル
# ---------------------------------------------------------------------------

class ChatPanel(QWidget):
    """キャラの隣に出る会話パネル。検索も訂正も片付けもここに流れる。"""

    def __init__(self, app_state: "Kohaku") -> None:
        super().__init__()
        self.state = app_state
        self.threads: list[QThread] = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setFixedSize(376, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        frame = QFrame()
        frame.setObjectName("panel")
        frame.setStyleSheet(
            f"QFrame#panel {{ background:{PANEL}; border:1px solid {LINE}; border-radius:14px; }}"
        )
        root.addWidget(frame)

        inner = QVBoxLayout(frame)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(0)

        inner.addWidget(self._build_header())

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background:transparent;")
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.log_widget = QWidget()
        self.log = QVBoxLayout(self.log_widget)
        self.log.setContentsMargins(14, 14, 14, 14)
        self.log.setSpacing(10)
        self.log.addStretch(1)
        self.scroll.setWidget(self.log_widget)
        inner.addWidget(self.scroll, 1)

        inner.addWidget(self._build_composer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setStyleSheet(f"border-bottom:1px solid {LINE};")
        row = QHBoxLayout(header)
        row.setContentsMargins(14, 10, 10, 10)

        title = QLabel("コハク")
        title.setStyleSheet(f"color:{INK}; font-size:12.5px; font-weight:700; font-family:{FONT};")
        row.addWidget(title)

        self.status = QLabel("ダウンロードフォルダを見張り中")
        self.status.setStyleSheet(f"color:{DIM}; font-size:10px; font-family:{FONT};")
        row.addWidget(self.status)
        row.addStretch(1)

        close = QPushButton("×")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setFixedSize(24, 24)
        close.setStyleSheet(
            f"QPushButton {{ border:0; color:{DIM}; font-size:15px; background:transparent; }}"
            f"QPushButton:hover {{ background:{SUNK}; border-radius:6px; }}"
        )
        close.clicked.connect(self.hide_panel)
        row.addWidget(close)
        return header

    def _build_composer(self) -> QWidget:
        box = QWidget()
        box.setStyleSheet(f"border-top:1px solid {LINE};")
        row = QHBoxLayout(box)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(8)

        self.input = QLineEdit()
        self.input.setPlaceholderText("探しものを言葉で（例：先週の統計学の資料）")
        self.input.setStyleSheet(
            f"QLineEdit {{ background:{SUNK}; border:1px solid {LINE}; border-radius:9px;"
            f" padding:7px 10px; font-size:12px; color:{INK}; font-family:{FONT}; }}"
            f"QLineEdit:focus {{ border-color:{AMBER}; }}"
        )
        self.input.returnPressed.connect(self._submit)
        row.addWidget(self.input, 1)

        send = QPushButton("↑")
        send.setCursor(Qt.CursorShape.PointingHandCursor)
        send.setFixedSize(32, 32)
        send.setStyleSheet(
            f"QPushButton {{ background:{AMBER}; color:#fff; border:0; border-radius:9px;"
            f" font-size:14px; }}"
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
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

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
            head += f"<br><span style='color:{AMBER};'>{' / '.join(result.filters)}</span> で絞り込んでいます。"
        self.say(head)
        self.add_cards([hit.file for hit in result.hits])

    def show_similar(self, file: dict) -> None:
        self.say(f"「{file['filename']}」に似たファイルだね。")
        thread = SimilarThread(file["id"])
        thread.done.connect(self._show_similar_result)
        thread.finished.connect(lambda: self.threads.remove(thread))
        self.threads.append(thread)
        thread.start()

    def _show_similar_result(self, hits) -> None:
        if not hits:
            self.say("似ているものは見つからなかった。")
            return
        self.add_cards([hit.file for hit in hits])

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
        self.add_chips(
            candidates + [self.NEW_CATEGORY],
            lambda name: self._pick_category(file_id, name),
        )

    def _pick_category(self, file_id: int, category: str) -> None:
        if category == self.NEW_CATEGORY:
            name, ok = QInputDialog.getText(self, "新しいカテゴリ", "カテゴリ名を入れてください")
            name = name.strip() if ok else ""
            if not name:
                self.say("そのままにしておくね。")
                return
            category = name

        moved = t6.recategorize(file_id, category)
        if not moved:
            self.say("移せなかった。ファイルが見当たらない。")
            return

        # 新しいカテゴリは categories テーブルにも登録しておく。
        # ここを忘れると、次の分類でT4のプロンプトに候補として出てこない。
        if category not in db.get_category_names():
            db.insert_category(category)

        self.say(f"「{category}」に移したよ。")
        file = db.get_file(file_id)
        if file is not None:
            self.add_cards([file])

    # --- 表示制御 -----------------------------------------------------------

    def greet_if_empty(self) -> None:
        if self.log.count() > 1:
            return
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

    def show_panel(self) -> None:
        self.state.hide_toast()
        self.move(self.state.panel_position())
        self.show()
        self.raise_()
        self.activateWindow()
        self.greet_if_empty()
        self.input.setFocus()
        self.state.mascot.hide()

    def hide_panel(self) -> None:
        self.hide()
        self.state.mascot.show()

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
        self.setFixedWidth(280)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.frame = QFrame()
        self.frame.setObjectName("toast")
        self.frame.setStyleSheet(
            f"QFrame#toast {{ background:{PANEL}; border:1px solid {LINE}; border-radius:13px; }}"
        )
        root.addWidget(self.frame)

        self.box = QVBoxLayout(self.frame)
        self.box.setContentsMargins(13, 11, 13, 11)
        self.box.setSpacing(9)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setStyleSheet(f"color:{INK}; font-size:12px; font-family:{FONT};")
        self.box.addWidget(self.label)

        self.actions = QHBoxLayout()
        self.actions.setSpacing(6)
        self.box.addLayout(self.actions)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_message(self, html: str, actions: list[tuple[str, str, object]], at: QPoint) -> None:
        self.label.setText(html)

        while self.actions.count():
            item = self.actions.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for label, kind, handler in actions:
            button = _button(label, kind)
            button.clicked.connect(handler)
            self.actions.addWidget(button)
        self.actions.addStretch(1)

        self.adjustSize()
        self.move(at)
        self.show()
        self.raise_()
        self.timer.start(TOAST_MS)


# ---------------------------------------------------------------------------
# 常駐キャラ
# ---------------------------------------------------------------------------

class Mascot(QWidget):
    """
    枠なし・背景透過・常に最前面の小さなウィンドウ。
    クリックで会話パネル、ドラッグで移動、右クリックでメニュー。
    """

    clicked = Signal()

    SIZE = 86

    def __init__(self, app_state: "Kohaku") -> None:
        super().__init__()
        self.state = app_state
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._drag_origin: Optional[QPoint] = None
        self._moved = False
        self.working = False

        # 処理中の明滅
        self.phase = 0.0
        self.animation = QTimer(self)
        self.animation.timeout.connect(self._tick)
        self.animation.start(60)

    def _tick(self) -> None:
        if self.working:
            self.phase += 0.12
            self.update()

    def set_working(self, working: bool) -> None:
        self.working = working
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        import math

        glow = 1.0 + (0.18 * math.sin(self.phase) if self.working else 0.0)
        center = self.SIZE / 2

        # 体
        gradient = QRadialGradient(center - 8, center - 10, self.SIZE * 0.8)
        gradient.setColorAt(0.0, QColor(255, 224, 176).lighter(int(100 * glow)))
        gradient.setColorAt(0.6, QColor(232, 152, 63))
        gradient.setColorAt(1.0, QColor(170, 92, 26))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)

        path = QPainterPath()
        path.addEllipse(8, 10, self.SIZE - 16, self.SIZE - 20)
        painter.drawPath(path)

        # 目
        painter.setBrush(QColor(58, 36, 17))
        painter.drawEllipse(int(center - 16), int(center - 4), 7, 10)
        painter.drawEllipse(int(center + 9), int(center - 4), 7, 10)

        # ほお
        painter.setBrush(QColor(200, 90, 60, 90))
        painter.drawEllipse(int(center - 27), int(center + 8), 10, 6)
        painter.drawEllipse(int(center + 17), int(center + 8), 10, 6)

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

    def _place_mascot(self) -> None:
        """画面左下に置く。タスクバーを避けるため作業領域を基準にする。"""
        screen = self.app.primaryScreen().availableGeometry()
        self.mascot.move(screen.left() + 24, screen.bottom() - Mascot.SIZE - 16)

    def panel_position(self) -> QPoint:
        """パネルはキャラの上に重ねて出す（キャラは隠れる）。"""
        screen = self.app.primaryScreen().availableGeometry()
        x = max(screen.left() + 12, self.mascot.x())
        y = self.mascot.y() + Mascot.SIZE - self.panel.height()
        y = max(screen.top() + 12, y)
        return QPoint(x, y)

    def toast_position(self) -> QPoint:
        return QPoint(self.mascot.x() + Mascot.SIZE - 4, self.mascot.y() - 20)

    def hide_toast(self) -> None:
        self.toast.hide()

    def _build_tray(self) -> QSystemTrayIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(232, 152, 63))
        tray = QSystemTrayIcon(QIcon(pixmap))
        tray.setToolTip("コハク")

        menu = QMenu()
        open_action = QAction("コハクと話す", menu)
        open_action.triggered.connect(self.panel.show_panel)
        menu.addAction(open_action)
        menu.addSeparator()
        quit_action = QAction("終了", menu)
        quit_action.triggered.connect(self.app.quit)
        menu.addAction(quit_action)

        tray.setContextMenu(menu)
        tray.activated.connect(lambda _: self.panel.show_panel())
        tray.show()
        return tray

    def show_menu(self, at: QPoint) -> None:
        menu = QMenu()
        talk = QAction("コハクと話す", menu)
        talk.triggered.connect(self.panel.show_panel)
        menu.addAction(talk)
        menu.addSeparator()
        quit_action = QAction("終了", menu)
        quit_action.triggered.connect(self.app.quit)
        menu.addAction(quit_action)
        menu.exec(at)

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
    app.setFont(QFont("Yu Gothic UI", 9))
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
