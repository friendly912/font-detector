# font-detector

PDFのテキストレイヤーから、指定したフォントで書かれた箇所を検出して強調表示します。

## セットアップ (初回のみ・インターネット接続が必要)

1. Python 3.10 をインストールします。
2. 画像内の文字判定を使う場合は **Tesseract OCR** をインストールします
   (Windows: [UB Mannheim 版](https://github.com/UB-Mannheim/tesseract/wiki)。
   `setup.bat` からも winget でインストールできます)。
   Tesseract が無くても、テキストレイヤーの判定は使えます。
3. セットアップを実行します。

| OS | 手順 |
|---|---|
| Windows | `setup.bat` をダブルクリック |
| macOS / Linux | `./setup.sh` |

仮想環境 (.venv) の作成、パッケージのインストール (`requirements.txt` でバージョン固定)、
Tesseract の日本語・英語データの `tessdata/` への取得、Tesseract の確認までを行います。
状態は `python -m font_detector check-ocr` でいつでも確認できます。

## 起動 (オフラインで動作)

| OS | 手順 |
|---|---|
| Windows | `start.bat` をダブルクリック |
| macOS / Linux | `./start.sh` |

ブラウザで http://127.0.0.1:8000 が自動で開きます。終了はコンソールで Ctrl+C。

セットアップ後はインターネット接続を必要としません。
- PDFはこのPC内のサーバー (127.0.0.1) で処理され、外部に送信されません
- OCR はローカルの Tesseract と `tessdata/` の言語データで行います
- UIは外部のCDN・フォント等を読み込みません

Tesseract・言語データの場所は環境変数で変更できます。

| 環境変数 | 内容 |
|---|---|
| `FONT_DETECTOR_TESSERACT` | tesseract 実行ファイルのパス (既定: PATH、`C:\Program Files\Tesseract-OCR`) |
| `FONT_DETECTOR_TESSDATA` | 言語データのフォルダ (既定: `tessdata/`、無ければ Tesseract の既定) |
| `FONT_DETECTOR_OCR_LANG` | OCR の言語 (既定: `jpn+eng`) |

## 使い方

```bash
# Web UI (http://127.0.0.1:8000)。--open でブラウザを開く
python -m font_detector serve --open

# PDF内のフォント一覧
python -m font_detector fonts input.pdf

# 指定フォントの箇所を検出し、注釈付きPDFを出力 (--images で画像内の文字も判定)
python -m font_detector scan input.pdf --font "MS 明朝" --body-only --images -o out.pdf
```

フォントは、UIの一覧から選ぶか名前で指定します。名前の表記揺れ (`MS-Mincho` / `ＭＳ 明朝` / `ABCDEF+MSMincho` など) は吸収します。

## 画像内の文字の判定

UIの「画像内の文字も判定」をオンにすると、画像領域を Tesseract で読み取り、
1文字ずつ、PDF内のフォントの見本字形と比べて、どのフォントで書かれているかを推定します。

- 見本字形は、PDFのテキストレイヤー (埋め込みフォント) から切り出します。
  足りない文字は、OSにインストールされている同名のフォントで補います
  (Windows なら `C:\Windows\Fonts`。環境変数 `FONT_DETECTOR_FONT_DIRS` で検索先を追加可能)。
- 行ごとに、最も似ているフォントの類似度がしきい値 (既定 0.85) 以上なら該当とします。
  PDFで使われていないフォントの行は、どの候補とも類似度が低くなり、除外されます。
- Tesseract の文字位置は不正確なため、行の文字列と確信度だけを使い、1文字ずつの区切りは
  画像のインクの切れ目から求めます。確信度の低い文字 (60未満) は比較に使いません。
- 画像はまとめて1回の Tesseract 起動で処理します。結果はドキュメントごとにキャッシュします。

## 制限事項

- OCRで付与された透明テキストは判定から除外
- 「本文のみ」は、本文サイズ (文字数最多のサイズ) ±1pt、かつ上下6%の余白外にある行 (テキストレイヤーのみ)
- 画像内判定の対象は横書きのみ。見本が無いフォント (未埋め込みで、OSにも無い) は候補になりません
- 文字高が 20px 未満の低解像度画像では、明朝・ゴシックのような大きな違いは判別できますが、
  似た書体同士 (ゴシック体の別フォントなど) の区別は難しくなります
