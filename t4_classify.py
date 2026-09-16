"""
T4. AI分類（Gemini API呼び出し）

入力: 抽出済みテキスト（T3の出力）＋ 既存カテゴリ一覧
出力: category / subtags / summary の辞書

- classify_content() がGemini APIを呼び出し、分類結果を返す
- レスポンスがJSONとしてパースできない場合はNoneを返し、エラーログを残す

このファイルの中身（実装方法）はT4担当の裁量。
外から使われるのは classify_content() 関数だけ。

事前準備:
- pip install google-generativeai
- 環境変数 GEMINI_API_KEY にAPIキー（https://aistudio.google.com/apikey で発行）を設定
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"

_PROMPT_TEMPLATE = """あなたはファイル整理アシスタントです。
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


def classify_content(
    content: str,
    existing_categories: list[str],
) -> Optional[dict[str, Any]]:
    """
    抽出済みテキストをGemini APIに渡し、分類結果を取得する。

    Args:
        content: T3で抽出されたファイルの中身（テキスト）
        existing_categories: 既存カテゴリ名の一覧（表記ゆれ防止のためプロンプトに含める）

    Returns:
        {"category": str, "subtags": list[str], "summary": str} の辞書。
        API呼び出し失敗、またはレスポンスがJSONとしてパースできない場合はNone。
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("[T4] GEMINI_API_KEY が設定されていません")
        return None

    prompt = _PROMPT_TEMPLATE.format(
        categories="\n".join(f"- {c}" for c in existing_categories) or "(まだ登録なし)",
        content=content,
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        raw_text = response.text
    except Exception as e:
        logger.error(f"[T4] Gemini API呼び出しに失敗: {e}")
        return None

    return _parse_response(raw_text)


def _parse_response(raw_text: str) -> Optional[dict[str, Any]]:
    """
    Geminiのレスポンス文字列からcategory/subtags/summaryを取り出す。

    Geminiがコードブロック（```json ... ```）で囲んで返すことがあるため、
    それを取り除いてからJSONとしてパースする。
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"[T4] レスポンスのJSONパースに失敗: {e} / raw={raw_text!r}")
        return None

    if not all(key in data for key in ("category", "subtags", "summary")):
        logger.error(f"[T4] レスポンスに必要なキーが不足: {data}")
        return None

    return data
