# 引き継ぎメモ（別マシンで作業を再開する人向け）

最終更新: 2026-09-16 / 対象ブランチ: `sub`

このファイルは、別のPCで作業を再開するときに最初に読むためのものです。
Claude Codeの会話履歴はマシンごとにローカル保存され同期されないため、
前の環境でのやりとりはここに書いてあることが全てです。

---

## 1. 新しいPCでのセットアップ手順

```bash
git clone https://github.com/t-kume-dev/hackson2026
cd hackson2026
git checkout sub          # main ではなく sub が最新
pip install -r requirements.txt
```

Python 3.12（3.12.3）と 3.13（3.13.14）で動作確認済み。
`requirements.txt` は全モジュールのimportを満たしていることを確認済み（過不足なし）。
torch は sentence-transformers が依存として引いてくるので明記していない。

### gitに入っていないので手で用意するもの

| 対象 | 内容 |
|---|---|
| `.env` | `GEMINI_API_KEY=<各自のキー>` の1行。Google AI Studioで取得する |
| `omakase.db` | 実行すれば自動で作られる。前の環境のテストデータは持ち込まなくてよい |
| `organized/` | 同上。T6が自動で作る |

### 初回実行時に時間がかかる点

T6が使う sentence-transformers のモデルがHuggingFaceから自動ダウンロードされます
（e5-baseで約1.1GB、`~/.cache/huggingface/` に入る）。ネットが細い環境やデモ直前だとここで
待たされるので、**本番の前に一度 `python main.py <フォルダ>` を走らせてキャッシュを
作っておくこと。**

### 任意（今は無くても動く）

`pdf2image` はPoppler、`pytesseract` はTesseract-OCR本体という外部バイナリを別途
必要とします。どちらも未インストールでも構いません。これらが効くのは
「テキストを含まないスキャンPDF」のOCRフォールバックだけで、
テキストPDF・画像・テキストファイルの処理には一切影響しません。
未インストールの場合、該当PDFは明示的なエラーログを出してスキップされます。

---

## 2. いまどこまで出来ているか

`main.py` で T1〜T6 が直列に配線され、**実ファイルでの通しテストに成功しています。**

```
T1 t1_watch.watch_folder      フォルダ監視（watchdog、新規ファイルのみ検知）
T2 t2_screening.screen_file   処理対象かの判定＋種別(text/pdf/photo)
T3 t3_content.extract_content 中身の抽出（画像はbytesのままT4へ）
T4 t4_classify.classify_content  Gemini APIで カテゴリ/サブタグ/要約 を生成
T5 t5_dedupe.resolve_category    カテゴリの表記ゆれをembeddingで統合
T6 t6_filemanager.save_result    カテゴリフォルダへ移動＋DB保存
```

通しテストではテキスト4件を投入し、検出→分類→カテゴリ統合→移動→DB保存まで
全て動作。`files` 4件 / `trash_log` 4件が整合していることを確認済み。

**T7（意味検索UI）・T8（放置ファイル検出）・T9（ゴミ箱Undo）は未着手です。**

---

## 3. DB構成（ここは最近大きく変えたので必読）

以前はT5とT6がそれぞれ独自に `categories.db` へ直接SQLを書いていて、
`db.py` はどこからも使われない死んだコードでした。これを解消し、
**SQLiteアクセスは全て `db.py` に集約済み**です。

- **DBファイルのパスは `db.DB_PATH`（= `omakase.db`）ただ1つが正。**
  呼び出し側で `"xxx.db"` のような相対パスを書かないこと。実行時のカレント
  ディレクトリ次第で別のDBが作られ、カテゴリが消えたように見えます。
  同じ理由で `t6_filemanager.ORGANIZED_ROOT` もモジュール基準のパスにしてあります。
- テーブルは仕様書8章どおり `files` / `categories` / `trash_log` の3つ。
- `categories.embedding` は **NULL許容**。T5はembedding APIに失敗したとき
  カテゴリ名だけ先に登録し、次回呼び出しでバックフィルする設計のためです。
- `insert_file` / `insert_trash_log` は `conn` 引数を取れます。T6は両INSERTを
  1トランザクションにまとめており、片方だけ残ってUndo不能になるのを防いでいます。

T7/T8/T9を書くときも、`sqlite3.connect` を直接呼ばず `db.py` に関数を足してください。

### embeddingが2種類あることに注意

| 保存先 | モデル | 次元 | 用途 |
|---|---|---|---|
| `categories.embedding` | Gemini `gemini-embedding-001` | 3072 | T5のカテゴリ表記ゆれ判定 |
| `files.embedding` | sentence-transformers `intfloat/multilingual-e5-base` | 768 | **T7の意味検索用** |

T7で検索クエリをベクトル化するときは、**必ず `t6_filemanager.embed_query()` を使うこと。**
モデルもプレフィックスも正規化もこの関数の内側で揃えてあります。別のモデルを使うと
次元も意味も噛み合わず、検索結果が完全にデタラメになります。

---

## 4. 未決事項（再開したらここから）

### (A) 【対応済み】embeddingモデルを e5-base に変更した

`t6_filemanager.EMBEDDING_MODEL_NAME` を `intfloat/multilingual-e5-base`（768次元）に
変更済み。旧 `paraphrase-multilingual-MiniLM-L12-v2` は現実的なノイズ入り300件の実測で
正解率83.3%（正解が20位まで落ちるような外し方をする）だったのに対し、e5-baseは94.4%。
e5-large(2.2GB)はe5-baseと同点だったので採用していない。

DBがまだ空の段階で変更したため、既存 `files.embedding` の再生成は不要だった。

e5系に必須のプレフィックス（文書側 `passage: ` / クエリ側 `query: `）は
`t6_filemanager.py` の内側に閉じ込めてある。**T7担当は自前でモデルを読まず、
`t6_filemanager.embed_query(クエリ文字列)` を呼ぶこと。** embeddingはL2正規化済みなので、
`files.embedding` との比較は内積だけでコサイン類似度になる。

なお初回実行時のモデルダウンロードは約1.1GBに増えた（旧MiniLMは約460MB）。
本番前のキャッシュ作成（1章参照）はより重要になっている。

### (B) Gemini APIの課金プラン確認

- モデルは `gemini-3.6-flash`。`gemini-2.5-flash` は
  `no longer available to new users` で404になるため使えません（実測確認済み）。
- `t4_classify.py` のコメントに「3.6系など有料限定モデルは避ける」とありますが、
  実際には3.6-flashで200応答が返っており**コメントが実態と食い違っています。**
- 無料枠かどうかはAPIからは判定できません（レスポンスの `X-Gemini-Service-Tier:
  standard` は課金の有無ではなく処理ティアを示すヘッダ）。
  **Google AI Studioのキー画面でプランを確認し、コメントを実態に合わせて直すこと。**

### (C) その他

- `epic1.py` と `test_t2_read_failure.py` が配線から外れたまま残っています。
  現役かどうか要確認。
- ブランチ `t3pair` / `t4_otake` / `t5_otakeeee` / `t6` は全て `sub` にマージ済み。
  ただし**これらのブランチには壊れたgitlink（入れ子リポジトリ `hackson2026`）が
  残っています。** `sub` では `289ea35` で削除済みなので、これらを再マージしないこと。
  再マージすると復活します。`main` は元から綺麗です。

---

## 5. 前の環境で踏んだ罠

- **カレントディレクトリ依存**: 以前は `"categories.db"` と `"organized"` が相対パスで
  3箇所にハードコードされていました。実行場所を変えると別のDBが作られて
  「カテゴリが消えた」ように見えます。現在は全てモジュール基準の絶対パスです。
- **テスト副産物の誤コミット**: 過去に `*.db` と `organized/` が複数ブランチに
  コミットされたことがあります。`.gitignore` に追加済みですが、
  `git add -A` の前に `git status` を確認する習慣をつけてください。
- **ベンチマークは規模が命**: embeddingモデルの比較を少数の文書でやると全モデルが
  満点になり、何も分かりません（上記(A)参照）。必ず現実的な件数と、
  同カテゴリ内の紛らわしい文書を入れて測ること。
