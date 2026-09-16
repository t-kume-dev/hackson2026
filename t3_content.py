"""
T3. コンテンツ抽出（text/pdf/photo）

入力: ファイルパス + ファイル種別（T2の出力）
出力: すべて dict。中身は種別によって異なる。
  - text / pdf: {"filetype": "text"/"pdf", "file_path": str, "content": str}
  - photo     : {"filetype": "photo", "file_path": str, "image_bytes": bytes, "mime_type": str}

photoは中身の解析（OCR・キャプション・Gemini呼び出し）をT3では行わない。
AIに渡すのはファイルパスではなく「画像データそのもの（バイト列）」＋
API呼び出しに必要な mime_type（例: image/jpeg）。
パスを渡してもAI（Gemini等）はローカルファイルを直接読みには行けないため、
T4側でこの image_bytes と mime_type を使ってAPIに画像入力として渡す想定。

抽出に失敗、または中身が空の場合は None を返す。

- text : ファイルをそのまま読み込み（文字コードを複数パターン試す）
- pdf  : pdfplumberでテキスト抽出。抽出結果が空（スキャンPDF）の場合は
         pdf2image + pytesseract でOCRにフォールバック
- photo: 有効な画像ファイルであることを検証し、バイト列として読み込む

このファイルの中身（実装方法）はT3担当の裁量。
外から使われるのは extract_content() 関数だけ。
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Optional

import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# OCR言語（日本語＋英語）。事前に tesseract の jpn.traineddata が必要（PDFのOCRフォールバック用）。
OCR_LANGUAGES = "jpn+eng"

# テキストファイル読み込み時に試す文字コード（日本語ファイル対策でcp932/shift_jisも試す）
TEXT_ENCODINGS_TO_TRY = ("utf-8", "cp932", "shift_jis")

# mimetypesで判定できなかった場合のフォールバック
DEFAULT_PHOTO_MIME_TYPE = "image/jpeg"


def _read_text_file(path: Path) -> Optional[str]:
    """テキストファイルを複数の文字コードで読み込む"""
    for encoding in TEXT_ENCODINGS_TO_TRY:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as e:
            logger.error(f"[T3] テキストファイル読み込み失敗: {path} ({e})")
            return None

    try:
        logger.warning(f"[T3] 文字コード判定に失敗、utf-8(errors=replace)で強制読み込み: {path}")
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.error(f"[T3] テキストファイル読み込み失敗: {path} ({e})")
        return None


def _extract_pdf_text(path: Path) -> Optional[str]:
    """pdfplumberでテキスト抽出。空ならOCRにフォールバック"""
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages_text).strip()
    except Exception as e:
        logger.error(f"[T3] pdfplumberでの抽出に失敗: {path} ({e})")

    if text:
        return text

    logger.info(f"[T3] テキスト抽出結果が空のためOCRにフォールバック（スキャンPDFの可能性）: {path}")
    return _ocr_pdf(path)


def _ocr_pdf(path: Path) -> Optional[str]:
    """PDFの各ページを画像化してOCR"""
    try:
        images = convert_from_path(str(path))
    except Exception as e:
        logger.error(f"[T3] PDFの画像化に失敗（poppler未インストールの可能性）: {path} ({e})")
        return None

    ocr_texts = []
    for i, image in enumerate(images):
        try:
            ocr_texts.append(pytesseract.image_to_string(image, lang=OCR_LANGUAGES))
        except Exception as e:
            logger.error(f"[T3] OCR失敗（{i + 1}ページ目）: {path} ({e})")

    result = "\n".join(ocr_texts).strip()
    return result if result else None


def _read_photo(path: Path) -> Optional[dict]:
    """
    画像を検証し、AIに渡せる形（バイト列 + mime_type）で読み込む。
    """
    # 壊れた画像ファイルでないことを確認（verify()は検証専用で、以降そのImageオブジェクトは使えない）
    try:
        with Image.open(path) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as e:
        logger.error(f"[T3] 画像として開けないためスキップ: {path} ({e})")
        return None

    try:
        image_bytes = path.read_bytes()
    except OSError as e:
        logger.error(f"[T3] 画像ファイルの読み込みに失敗: {path} ({e})")
        return None

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = DEFAULT_PHOTO_MIME_TYPE

    return {
        "filetype": "photo",
        "file_path": str(path),
        "image_bytes": image_bytes,
        "mime_type": mime_type,
    }


def extract_content(file_path: str, filetype: str) -> Optional[dict]:
    """
    ファイル種別に応じてコンテンツを取得する。

    Args:
        file_path: 対象ファイルパス（入力。T1/T2の出力をそのまま渡せる）
        filetype: "text" / "pdf" / "photo" のいずれか（T2の出力をそのまま渡せる）

    Returns:
        - text / pdf の場合: {"filetype": str, "file_path": str, "content": str}
        - photo の場合     : {"filetype": "photo", "file_path": str,
                              "image_bytes": bytes, "mime_type": str}
        - 失敗した場合     : None（呼び出し側は後続処理をスキップする）
    """
    path = Path(file_path)

    if not path.exists():
        logger.error(f"[T3] ファイルが存在しません: {file_path}")
        return None

    if filetype == "photo":
        return _read_photo(path)

    if filetype == "text":
        content = _read_text_file(path)
    elif filetype == "pdf":
        content = _extract_pdf_text(path)
    else:
        logger.error(f"[T3] 未知のファイル種別: {filetype} ({file_path})")
        return None

    if content is None or not content.strip():
        logger.warning(f"[T3] コンテンツ抽出結果が空: {file_path}")
        return None

    return {
        "filetype": filetype,
        "file_path": str(path),
        "content": content,
    }