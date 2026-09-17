"""
T4. AI分類（Gemini API / OpenAI API 呼び出し）

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

OpenAIを使う場合:
- pip install openai
- .env に OPENAI_API_KEY を書くと、自動でOpenAIを使う（AI_PROVIDER=gemini で明示的に戻せる）
- モデルは OPENAI_MODEL で変えられる
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Callable, List, Optional, TypedDict

from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.6-flash"  # 無料枠対象（1日20回まで）
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-5.5")


def _provider() -> str:
    """使うAPIを決める。AI_PROVIDER が無ければ、OPENAI_API_KEY があるときだけOpenAIにする。"""
    explicit = os.environ.get("AI_PROVIDER", "").strip().lower()
    if explicit in ("openai", "gemini"):
        return explicit
    return "openai" if os.environ.get("OPENAI_API_KEY") else "gemini"

# カテゴリ階層の区切り文字と最大深さ。t5_dedupe.py もこの2つをimportして使う。
CATEGORY_SEPARATOR = "／"
MAX_CATEGORY_DEPTH = 3


class ClassificationResult(TypedDict):
    category: str
    subtags: List[str]
    summary: str


_TEXT_PROMPT_TEMPLATE = """あなたはファイル整理アシスタントです。
以下のファイル内容を分析し、最も適切な分類をJSON形式で1つだけ返してください。

分類は階層構造（親 > 子 > 孫）で表現し、意味のある深さまでで構いません。
無理に{max_depth}階層まで分けず、それ以上細分化できないものは1〜2階層で止めてください。

既存カテゴリの使い方:
- 同じ意味のカテゴリが既にあれば、その表記を一字一句そのまま使う（言い換え・表記ゆれを作らない）
- 内容が違うなら、名前が似ていても既存に寄せず新しく作る（例: 雇用契約書 と 賃貸借契約書 は別物）
- 既存にぴったりのものが無いのに、無関係なカテゴリへ押し込まない

# 既存カテゴリ一覧（ツリー表示。合うものがあれば表記も揃えて使う）
{categories}

# ファイル内容
{content}

# 出力形式（このJSON以外は何も出力しないこと）
{{
  "category_path": ["第1階層", "第2階層(あれば)", "第3階層(あれば)"],
  "subtags": ["関連キーワード1", "関連キーワード2"],
  "summary": "内容の要約（100文字以内）"
}}
"""

_IMAGE_PROMPT_TEMPLATE = """あなたはファイル整理アシスタントです。
添付された画像の内容を見て、最も適切な分類をJSON形式で1つだけ返してください。

分類は階層構造（親 > 子 > 孫）で表現し、意味のある深さまでで構いません。
無理に{max_depth}階層まで分けず、それ以上細分化できないものは1〜2階層で止めてください。

既存カテゴリの使い方:
- 同じ意味のカテゴリが既にあれば、その表記を一字一句そのまま使う（言い換え・表記ゆれを作らない）
- 内容が違うなら、名前が似ていても既存に寄せず新しく作る（例: 雇用契約書 と 賃貸借契約書 は別物）
- 既存にぴったりのものが無いのに、無関係なカテゴリへ押し込まない

# 既存カテゴリ一覧（ツリー表示。合うものがあれば表記も揃えて使う）
{categories}

# 出力形式（このJSON以外は何も出力しないこと）
{{
  "category_path": ["第1階層", "第2階層(あれば)", "第3階層(あれば)"],
  "subtags": ["関連キーワード1", "関連キーワード2"],
  "summary": "画像の内容の要約（100文字以内）"
}}
"""


def _categories_tree_text(existing_categories: List[str]) -> str:
    """
    フルパス文字列のリスト（例: "アニメ・ゲーム／鬼滅の刃"）から
    インデント付きツリー表示を組み立てる。DBはフラットなままなので、
    表示専用の変換としてここでだけ行う。
    """
    if not existing_categories:
        return "(まだ登録なし)"
    lines = []
    for path in sorted(existing_categories):
        parts = path.split(CATEGORY_SEPARATOR)
        indent = "  " * (len(parts) - 1)
        lines.append(f"{indent}- {parts[-1]}")
    return "\n".join(lines)


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

    required_keys = ("category_path", "subtags", "summary")
    missing = [k for k in required_keys if k not in data]
    if missing:
        logger.error(f"[T4] レスポンスに必要なキーが不足: {missing} / data={data}")
        return None

    category_path = data["category_path"]
    if not isinstance(category_path, list) or not category_path:
        logger.error(f"[T4] category_pathが不正な形式です: {category_path!r}")
        return None

    # 区切り文字が万一そのままセグメントに混ざると階層がズレるので潰す。深さも上限で切る。
    safe_parts = [
        str(p).strip().replace(CATEGORY_SEPARATOR, "・")
        for p in category_path[:MAX_CATEGORY_DEPTH]
        if str(p).strip()
    ]
    if not safe_parts:
        logger.error(f"[T4] category_pathが空になりました: {category_path!r}")
        return None

    return {
        "category": CATEGORY_SEPARATOR.join(safe_parts),
        "subtags": data["subtags"],
        "summary": data["summary"],
    }


def _gemini_caller(
    api_key: str, prompt: str, image_bytes: Optional[bytes], mime_type: Optional[str]
) -> Callable[[], Optional[str]]:
    client = genai.Client(api_key=api_key)
    contents: list = [prompt]
    if image_bytes is not None:
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    def call() -> Optional[str]:
        return client.models.generate_content(model=MODEL_NAME, contents=contents).text

    return call


def _openai_caller(
    api_key: str, prompt: str, image_bytes: Optional[bytes], mime_type: Optional[str]
) -> Callable[[], Optional[str]]:
    # SDK自身のリトライは切り、下のループの指数バックオフに一本化する
    client = OpenAI(api_key=api_key, max_retries=0)
    content: list = [{"type": "text", "text": prompt}]
    if image_bytes is not None:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    def call() -> Optional[str]:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    return call


def classify_content(
    t3_output: dict,
    existing_categories: List[str],
    max_retries: int = 3,
) -> Optional[ClassificationResult]:
    """
    T3の出力をGemini API（またはOpenAI API）に渡し、分類結果を取得する。

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
    provider = _provider()
    key_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    api_key = os.environ.get(key_name)
    if not api_key:
        logger.error(f"[T4] {key_name} が設定されていません")
        return None

    filetype = t3_output.get("filetype")
    file_path = t3_output.get("file_path", "(不明)")
    categories_text = _categories_tree_text(existing_categories)

    # --- filetypeでテキスト系 / 画像系に分岐 ---
    image_bytes: Optional[bytes] = None
    mime_type: Optional[str] = None
    if filetype in ("text", "pdf"):
        content = t3_output.get("content")
        if content is None:
            logger.error(f"[T4] content が見つかりません: {file_path}")
            return None
        prompt = _TEXT_PROMPT_TEMPLATE.format(
            categories=categories_text, content=content, max_depth=MAX_CATEGORY_DEPTH
        )

    elif filetype == "photo":
        image_bytes = t3_output.get("image_bytes")
        mime_type = t3_output.get("mime_type")
        if image_bytes is None or mime_type is None:
            logger.error(f"[T4] image_bytes / mime_type が不足しています: {file_path}")
            return None
        prompt = _IMAGE_PROMPT_TEMPLATE.format(categories=categories_text, max_depth=MAX_CATEGORY_DEPTH)

    else:
        logger.error(f"[T4] 未対応のfiletypeです: {filetype} (file_path={file_path})")
        return None

    if provider == "openai":
        call = _openai_caller(api_key, prompt, image_bytes, mime_type)
    else:
        call = _gemini_caller(api_key, prompt, image_bytes, mime_type)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_text = call()
            if raw_text is None:
                # 安全フィルター等で本文が空の場合、例外を出さずNoneが返ることがある。
                # ここで例外化してリトライに乗せないと、次の_parse_response呼び出しで
                # AttributeErrorとなり分類処理全体が落ちてしまう。
                raise ValueError("レスポンスにtextがありません（安全フィルター等の可能性）")
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