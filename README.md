# font-detector

PDFのテキストレイヤーから、指定したフォントで書かれた箇所を検出して強調表示します。

## セットアップ (初回のみ)

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
Windows の `setup.bat` は、`package/` があればインターネットを使わずにそこからインストールします
(下記)。`package/` が無い場合と `setup.sh` はインターネット接続が必要です。
状態は `python -m font_detector check-ocr` でいつでも確認できます。

### インターネットに接続できない Windows PC へのインストール

`package/` に Windows 10/11 (64bit)・Python 3.10 用のパッケージ一式 (wheel) を同梱しています。

1. 64bit 版の Python 3.10 をインストールします。
2. プロジェクト一式 (GitHub の「Download ZIP」など) をコピーします。
3. 画像内の文字も判定する場合は、Tesseract のインストーラーと `tessdata/` の言語データ
   (`jpn.traineddata`、`eng.traineddata`) も用意します。
4. `setup.bat` を実行します。`package/` を見つけると、自動でオフラインインストールします。

手動で行う場合は、プロジェクトのフォルダで次を実行します。

```bat
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-index --find-links package --upgrade pip
.venv\Scripts\python.exe -m pip install --no-index --find-links package -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-index --find-links package --no-deps -e .
```

Tesseract 本体と言語データ (`tessdata/*.traineddata`) は含まれないため、別途用意してください。
`requirements.txt` を更新したら、`package/` も取り直します。

```bash
python3.10 -m pip download --dest package --platform win_amd64 --python-version 3.10 \
    --implementation cp --abi cp310 --only-binary=:all: --no-deps -r requirements.txt
python3.10 -m pip download --dest package --platform win_amd64 --python-version 3.10 \
    --only-binary=:all: --no-deps "setuptools>=68" wheel pip
```

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
python -m font_detector scan input.pdf --font "MS 明朝" --body-only --images [--method ml|template] -o out.pdf
```

フォントは、UIの一覧から選ぶか名前で指定します。名前の表記揺れ (`MS-Mincho` / `ＭＳ 明朝` / `ABCDEF+MSMincho` など) は吸収します。

## 画像内の文字の判定

UIの「画像内の文字も判定」をオンにすると、画像領域を Tesseract で読み取り、
1文字ずつ、どのフォントで書かれているかを推定します。判定方式は2つあります。

### 機械学習 (既定)

仕組み・特徴量・ベンチマークの詳しい解説: [docs/ml-method.html](docs/ml-method.html)

PDFのテキストレイヤーは「どのフォントでどの字形が描かれているか」の正解付きデータになるため、
これを使って**文書ごとに分類器を学習**します (scikit-learn のロジスティック回帰、CPUのみ)。

- 学習データ: テキストレイヤーから切り出した字形を、縮小・ぼかし・JPEG圧縮・ノイズ・太さの変化で
  水増しし、画像内の文字の見え方に近づけます。埋め込まれていないフォントは同名のシステムフォントで補います。
- 「その他」クラス: PDFで使われていないインストール済みフォントの字形も学習し、
  PDFに無い書体の文字を「その他のフォント」に分類します。
- 特徴量: 形の傾向 (HOG) と、線の太さの分布・縦画と横画の太さの比 (明朝体の判別に効く) など。
  文字そのものではなく書体の特徴を学習するので、**PDFの本文に出てこない文字も判定できます**。
- 行ごとに各文字の確率を平均し、最も高いフォントの確率がしきい値 (既定 0.55) 以上なら該当とします。

### 字形照合

画像内の文字を、PDFの本文にある**同じ文字**の字形と比べます (類似度のしきい値 既定 0.85)。
同じ文字が本文に無いと比較できません。学習データが足りない場合 (PDFのフォントが1種類で、
「その他」に使えるフォントも無い場合など) は、自動的にこちらを使います。

### 比較 (評価用の日本語フォント16種によるベンチマーク)

2つの文書 (PDFのフォント2種ずつ) の画像内の126行 (文字の高さ 16/22/32px、JPEG圧縮) で評価しました。
画像内の文章は本文に無い文字を多く含み、未知フォントは学習に一切使っていません。

| 方式 | PDFのフォントの行を正しく判定 | 未知フォントの行を除外 |
|---|---|---|
| 字形照合 (0.85) | 25% | 100% |
| 機械学習 (0.55) | **100%** | 65% |

機械学習は、同じ文字が本文に無くても判定できる点で大きく上回ります。一方、PDFのフォントと
よく似た書体 (Zen Kaku Gothic に対する丸ゴシック版の Zen Maru Gothic など) は誤って
受け入れることがあります。「その他」の学習に使うフォントが多い環境 (Windows 標準の日本語フォント等)
ほど除外の精度は上がります。しきい値を上げると除外が増え、検出は減ります。

### 共通

- Tesseract の文字位置は不正確なため、行の文字列と確信度だけを使い、1文字ずつの区切りは
  画像のインクの切れ目から求めます。確信度の低い文字 (60未満) は判定に使いません。
- 画像はまとめて1回の Tesseract 起動で処理します。OCR・学習の結果はドキュメントごとにキャッシュし、
  フォントの選択・判定方式・しきい値の変更は即座に反映されます。

## 制限事項

- OCRで付与された透明テキストは判定から除外
- 「本文のみ」は、本文サイズ (文字数最多のサイズ) ±1pt、かつ上下6%の余白外にある行 (テキストレイヤーのみ)
- 画像内判定の対象は横書きのみ。見本が無いフォント (未埋め込みで、OSにも無い) は候補になりません
- 文字高が 20px 未満の低解像度画像では、明朝・ゴシックのような大きな違いは判別できますが、
  似た書体同士 (ゴシック体の別フォントなど) の区別は難しくなります
