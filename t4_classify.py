"""
T4. AI分類（Gemini API呼び出し）

入力: T3の出力（dict）。形式は filetype によって変わる。
    - text/pdf:
        {"filetype": "text" | "pdf", "file_path": str, "content": str}
    - photo:
        {"filetype": "photo", "file_path": str, "image_bytes": bytes, "mime_type": str}

出力: category / subtags / summary の辞書（失敗時はNone）

- classify_content() がGemini APIを呼び出し、分類結果を返す
- text/pdfは content をそのままプロンプトに埋め込む
- photoは image_bytes + mime_type をマルチモーダル入力としてそのままGeminiに渡す
- レスポンスがJSONとしてパースできない場合はNoneを返し、エラーログを残す
- API呼び出し失敗時は指数バックオフでリトライする

このファイルの中身（実装方法）はT4担当の裁量。
外から使われるのは classify_content() 関数だけ。

事前準備:
- pip install google-genai python-dotenv
  （旧 google-generativeai は非推奨になったため、新SDKのgoogle-genaiを使用）
- 環境変数 GEMINI_API_KEY にAPIキー（https://aistudio.google.com/apikey で発行）を設定
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Optional, TypedDict

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.6-flash"  # 無料枠対象。3.6系など有料限定モデルは避ける


class ClassificationResult(TypedDict):
    category: str
    subtags: List[str]
    summary: str


_TEXT_PROMPT_TEMPLATE = """あなたはファイル整理アシスタントです。
以下のファイル内容を分析し、最も適切な分類をJSON形式で1つだけ返してください。

# 既存カテゴリ一覧
{categories}

# ファイル内容
{content}

# 出力形式（このJSON以外は何も出力しないこと）
{{
  "category": "分類名（既存カテゴリに合うものがあればそれを使う。なければ新規名を提案する）",
  "subtags": ["関連キーワード1", "関連キーワード2"],
  "summary": "内容の要約（100文字以内）"
}}
"""

_IMAGE_PROMPT_TEMPLATE = """あなたはファイル整理アシスタントです。
添付された画像の内容を見て、最も適切な分類をJSON形式で1つだけ返してください。

# 既存カテゴリ一覧
{categories}

# 出力形式（このJSON以外は何も出力しないこと）
{{
  "category": "分類名（既存カテゴリに合うものがあればそれを使う。なければ新規名を提案する）",
  "subtags": ["関連キーワード1", "関連キーワード2"],
  "summary": "画像の内容の要約（100文字以内）"
}}
"""


def _categories_text(existing_categories: List[str]) -> str:
    return "\n".join(f"- {c}" for c in existing_categories) or "(まだ登録なし)"


def _parse_response(raw_text: str) -> Optional[ClassificationResult]:
    """
    Geminiのレスポンス文字列からcategory/subtags/summaryを取り出す。
    ```json ... ``` のようにコードブロックで囲まれてしまった場合も一応剥がして試みる。
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"[T4] レスポンスのJSONパースに失敗: {e} / raw={raw_text[:200]!r}")
        return None

    required_keys = ("category", "subtags", "summary")
    missing = [k for k in required_keys if k not in data]
    if missing:
        logger.error(f"[T4] レスポンスに必要なキーが不足: {missing} / data={data}")
        return None

    return {
        "category": data["category"],
        "subtags": data["subtags"],
        "summary": data["summary"],
    }


def classify_content(
    t3_output: dict,
    existing_categories: List[str],
    max_retries: int = 3,
) -> Optional[ClassificationResult]:
    """
    T3の出力をGemini APIに渡し、分類結果を取得する。

    Args:
        t3_output: T3からもらう辞書。
            text/pdf: {"filetype": "text"|"pdf", "file_path": str, "content": str}
            photo:    {"filetype": "photo", "file_path": str, "image_bytes": bytes, "mime_type": str}
        existing_categories: 既存カテゴリ名の一覧
        max_retries: API呼び出し失敗・パース失敗時の最大リトライ回数

    Returns:
        成功時: {"category": str, "subtags": list[str], "summary": str}
        失敗時: None（後続処理（T5）には渡さない）
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("[T4] GEMINI_API_KEY が設定されていません")
        return None

    filetype = t3_output.get("filetype")
    file_path = t3_output.get("file_path", "(不明)")
    categories_text = _categories_text(existing_categories)

    # --- filetypeでテキスト系 / 画像系に分岐 ---
    if filetype in ("text", "pdf"):
        content = t3_output.get("content")
        if content is None:
            logger.error(f"[T4] content が見つかりません: {file_path}")
            return None
        prompt = _TEXT_PROMPT_TEMPLATE.format(categories=categories_text, content=content)
        contents: list = [prompt]

    elif filetype == "photo":
        image_bytes = t3_output.get("image_bytes")
        mime_type = t3_output.get("mime_type")
        if image_bytes is None or mime_type is None:
            logger.error(f"[T4] image_bytes / mime_type が不足しています: {file_path}")
            return None
        prompt = _IMAGE_PROMPT_TEMPLATE.format(categories=categories_text)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        contents = [prompt, image_part]

    else:
        logger.error(f"[T4] 未対応のfiletypeです: {filetype} (file_path={file_path})")
        return None

    client = genai.Client(api_key=api_key)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
            )
            raw_text = response.text
        except Exception as e:
            last_error = e
            wait_seconds = 2 ** (attempt - 1)  # 1秒 -> 2秒 -> 4秒
            logger.error(
                f"[T4] API呼び出し失敗 ({attempt}/{max_retries}): {e} "
                f"-> {wait_seconds}秒後にリトライします"
            )
            time.sleep(wait_seconds)
            continue

        result = _parse_response(raw_text)
        if result is not None:
            return result

        logger.error(f"[T4] パースエラーのためリトライします ({attempt}/{max_retries})")
        time.sleep(1)

    logger.error(f"[T4] {max_retries}回試行しましたが分類に失敗しました。最終エラー: {last_error}")
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sample_categories = ["請求書", "レポート", "写真"]

    # --- text/pdf のテスト ---
    text_sample = {
        "filetype": "text",
        "file_path": "C:/sample/memo.txt",
        "content": "2026年度のハッカソン企画書。テーマはダウンロードファイルの自動整理。",
    }
    result = classify_content(text_sample, sample_categories)
    print(f"[T4/text] 結果: {result}")

    # --- photo のテスト（実際に試す場合は画像ファイルを読み込んでコメントアウトを外す） ---
    with open("pokemon_191127_04_01.jpg", "rb") as f:
        photo_sample = {
             "filetype": "photo",
             "file_path": "pokemon_191127_04_01.jpg",
             "image_bytes": f.read(),
             "mime_type": "image/jpg",
         }
    result = classify_content(photo_sample, sample_categories)
    print(f"[T4/photo] 結果: {result}")