# おまかせ整理Bot 仕様書

## 1. 概要
ダウンロードフォルダに保存されたファイル（テキスト、PDF、写真）を自動でスクリーニング・分類し、分類後の「見返す」「探す」「片付ける」コストを下げるデスクトップアプリ。
既存の自動仕分けツール（Hazel等）は分類までしかカバーせず、分類後の管理体験が弱いという課題に対し、**意味検索**と**放置ファイル検出**によって差別化する。

## 2. 背景・課題
- ダウンロードしたファイル（当初は動画も含めた整理を想定していたが、今回は対象外／将来拡張）の整理が手間
- 既存ツールで分類はできても、後から「あのファイルどこだっけ」と探すコストが高い
- 分類したまま放置され、ストレージを圧迫するファイルが溜まる

## 3. 対象ユーザー・ユースケース
- PCのダウンロードフォルダにファイルが溜まりがちな学生・社会人
- 例：授業資料（PDF・スライド・スクショ）を毎週ダウンロードし、気づくと未整理のまま溜まっている

## 4. スコープ（対象ファイル）
| 種別 | 対象 |
|---|---|
| テキストファイル | ○ |
| PDF | ○ |
| 写真 | ○ |
| 動画 | ×（将来拡張） |

## 5. システム全体構成
```
[ダウンロードフォルダ]
      │ watchdogで常時監視
      ▼
[① スクリーニング] 拡張子判定・壊れたファイル/一時ファイルの除外
      ▼
[② 分類（AI）] Claude APIで内容解析
      ├─ カテゴリ生成（既存カテゴリと類似度チェックして再利用 or 新規作成）
      ├─ サブタグ・要約生成
      └─ 埋め込みベクトル生成（写真は画像認識＋OCR）
      ▼
[③ ファイル移動] 分類結果のフォルダ構成へ実ファイルを移動
      ▼
[④ DB保存] SQLiteにメタデータを記録
      ▼
[⑤ 放置検出] 未アクセス期間＋重複（ハッシュ一致）＋サイズでスコアリング
      ▼
[⑥ 管理UI] 検索バー（自然文→類似検索→カード表示）／放置ファイル一覧／ゴミ箱
```

## 6. 機能要件

### 6.1 監視・検出機能（P0）
- `watchdog`でダウンロードフォルダを監視し、新規ファイルを検出したら自動でパイプラインを起動

### 6.2 スクリーニング機能（P0）
- 対象拡張子（txt/pdf/jpg/png等）の判定
- 破損ファイル・一時ファイル（.tmp, .crdownload等）の除外

### 6.3 分類・タグ付け機能（P0）

#### 6.3.1 全体の処理ステップ
1. ファイル種別に応じてコンテンツを抽出する
2. 既存カテゴリ一覧をDBから取得する
3. Claude APIに「コンテンツ＋既存カテゴリ一覧」を渡し、カテゴリ・サブタグ・要約をJSONで取得する
4. 返ってきたカテゴリ名を既存カテゴリと埋め込み類似度で照合し、必要なら統合する（重複防止）
5. 検索用の埋め込みベクトル（コンテンツ側）を生成する
6. ファイルのハッシュ値を計算する（重複ファイル検出用）
7. 最終カテゴリのフォルダへ実ファイルを移動する
8. DBにメタデータを保存する

#### 6.3.2 ファイル種別ごとのコンテンツ抽出
| 種別 | 抽出方法 |
|---|---|
| テキスト | ファイルをそのまま読み込み |
| PDF | `pdfplumber`でテキスト抽出。抽出結果が空（スキャンPDF）の場合は`pdf2image`でページを画像化し、`pytesseract`でOCR |
| 写真 | `pytesseract`でOCR（画像内の文字）＋Claude Vision APIで画像内容のキャプション生成。両方をテキストとして結合 |

#### 6.3.3 カテゴリ重複防止（二段構え）
- **1段目（プロンプト側）**：Claude APIへのリクエストに既存カテゴリ名の一覧を含め、「既存カテゴリに当てはまる場合はそれを使い、当てはまらない場合のみ新しいカテゴリ名を提案する」よう指示する
- **2段目（embedding側・フォールバック）**：それでも表記ゆれ（例：「統計学」「統計学入門」）が発生した場合に備え、返ってきたカテゴリ名を埋め込みベクトル化し、既存カテゴリとのコサイン類似度を計算。閾値（暫定0.85）以上なら既存カテゴリ名に差し替える

#### 6.3.4 処理フロー（疑似コード）
```python
def process_new_file(file_path):
    if not is_valid_target(file_path):          # スクリーニング済みが前提
        return

    filetype = detect_filetype(file_path)         # text / pdf / photo
    content = extract_content(file_path, filetype) # 6.3.2の方法で抽出

    existing_categories = db.get_category_names()

    # Claude APIへ1回の呼び出しで分類結果を取得
    ai_result = call_claude_classify(
        content=content[:MAX_CONTENT_LENGTH],
        filetype=filetype,
        existing_categories=existing_categories,
    )
    # ai_result = {"category": str, "subtags": [str], "summary": str}

    final_category = resolve_category(ai_result["category"], existing_categories)

    content_embedding = embed_model.encode(ai_result["summary"] + content[:2000])
    file_hash = sha256_of_file(file_path)

    new_path = move_to_category_folder(file_path, final_category)

    db.insert_file(
        path=new_path, filetype=filetype, category=final_category,
        subtags=ai_result["subtags"], summary=ai_result["summary"],
        embedding=content_embedding, hash=file_hash,
        file_size=os.path.getsize(new_path), created_at=now(),
    )


def resolve_category(proposed_name, existing_categories):
    proposed_embedding = embed_model.encode(proposed_name)
    best_name, best_score = find_most_similar(proposed_embedding, existing_categories)

    if best_score >= SIMILARITY_THRESHOLD:   # 暫定0.85
        return best_name                     # 既存カテゴリに統合
    else:
        db.insert_category(proposed_name, proposed_embedding)
        return proposed_name                 # 新規カテゴリとして採用
```

#### 6.3.5 Claude APIへのプロンプト設計イメージ
```
あなたはファイル整理アシスタントです。
以下のファイル内容を読み、次のJSON形式で分類してください。

【既存カテゴリ一覧】
{existing_categories}
※ 内容が既存カテゴリに当てはまる場合は必ずそれを使ってください。
　当てはまらない場合のみ、新しい適切なカテゴリ名を提案してください。

【ファイル内容】
{content}

【出力形式（JSONのみ）】
{
  "category": "カテゴリ名",
  "subtags": ["タグ1", "タグ2"],
  "summary": "2〜3文の要約"
}
```

### 6.4 ファイル移動機能（P0）
- 分類結果に基づき実ファイルを新フォルダ構成へ移動
- 移動前に元パスをDB（trash_log）に記録（Undo用）

### 6.5 放置ファイル検出機能（P2）
- 判定基準：未アクセス期間 ＋ 重複ファイル（ハッシュ完全一致） ＋ ファイルサイズの大きさ
- スコアが高いファイルを「整理候補」として一覧提示
- **実装方式：オンデマンド計算（常時監視・定期スキャンは行わない）**
  - 「放置ファイル」タブを開いたタイミングでDB上のアクティブファイルを対象にその場でスコアリングし、結果を表示する
  - 「未アクセス期間」はアプリ内で当該ファイルを開いた記録（`last_accessed_at`）を優先して使用し、記録が無い場合はOSのファイル更新日時（mtime）で代用する（OSのatimeはWindows/Linuxとも標準設定で更新が信頼できないため使用しない）
  - バックグラウンドジョブやスケジューラを持たないため、実装・運用コストが低い

### 6.6 検索・管理UI機能（P1）
- 検索バーに自然文入力 → 埋め込みベクトルの類似検索 → 結果をカード表示（要約・カテゴリ・サムネイル付き）
- 放置ファイルビュー：削除/アーカイブを提案するリスト表示

### 6.7 安全策：ゴミ箱/Undo（P3）
- ファイル移動・削除操作をtrash_logに記録し、直近の操作を取り消せるUndo機能
- 削除は即時削除ではなくゴミ箱（アプリ内論理削除）へ

## 7. 非機能要件
- プラットフォーム：Windows/Mac対応のデスクトップアプリ（Python）
- コスト：ハッカソン規模のため常時ポーリングではなくイベント駆動（新規ファイル検出時のみAPI呼び出し）でAPIコストを抑制
- レスポンス：検索は1秒以内を目標（ローカル埋め込みDBでの近似検索）

## 8. データ設計（SQLite）

**files テーブル**
| カラム | 型 | 内容 |
|---|---|---|
| id | INTEGER PK | |
| path | TEXT | 現在の保存パス |
| filename | TEXT | ファイル名 |
| filetype | TEXT | text / pdf / photo |
| category | TEXT | 大分類（AI自動生成） |
| subtags | TEXT(JSON) | サブタグ配列 |
| summary | TEXT | AI要約 |
| embedding | BLOB | 埋め込みベクトル |
| hash | TEXT | ファイルハッシュ（重複検出用） |
| file_size | INTEGER | バイト数 |
| created_at | DATETIME | 検出日時 |
| last_accessed_at | DATETIME | 最終アクセス日時 |
| status | TEXT | active / trashed |

**categories テーブル**
| カラム | 型 | 内容 |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | カテゴリ名 |
| embedding | BLOB | カテゴリ名の埋め込み（類似カテゴリ判定用） |

**trash_log テーブル**
| カラム | 型 | 内容 |
|---|---|---|
| id | INTEGER PK | |
| file_id | INTEGER FK | |
| original_path | TEXT | 移動前のパス |
| action_type | TEXT | move / delete |
| acted_at | DATETIME | |

## 9. 技術スタック
- 言語：Python
- フォルダ監視：`watchdog`
- DB：SQLite
- 埋め込み・類似検索：`sentence-transformers`（ローカル軽量モデル）
- 分類・要約・画像認識：Claude API
- UI：デスクトップアプリ（PyQt / Tkinter 等、チームの得意な方を選択）

## 10. 開発チケット分割案（優先順位付き・MVP基準）
| 優先度 | チケット | 内容 |
|---|---|---|
| P0 | フォルダ監視 | watchdogで新規ファイル検出 |
| P0 | スクリーニング | 拡張子判定・不要ファイル除外 |
| P0 | AI分類 | カテゴリ/タグ/要約生成＋カテゴリ重複防止ロジック |
| P0 | ファイル移動＋DB保存 | 実ファイル移動、メタデータ記録 |
| P1 | 検索UI | 検索バー＋embedding類似検索＋カード表示 |
| P2 | 放置ファイル検出 | 未アクセス＋重複＋サイズでスコアリング、一覧表示 |
| P3 | ゴミ箱/Undo | trash_log記録、操作の取り消し機能 |

P0が揃った時点でデモ可能な最小構成（監視→分類→移動→検索）になる想定。

## 11. デモシナリオ（想定）
1. 授業資料PDFを複数ダウンロード
2. 自動でカテゴリ「〇〇概論」フォルダが生成され、ファイルが移動される様子をリアルタイムで見せる
3. 検索バーに「先週の統計学の資料」と入力し、該当ファイルがカードで表示される
4. 放置ファイルビューで、重複・未使用の大容量ファイルが検出され、片付け提案が出る様子を見せる
