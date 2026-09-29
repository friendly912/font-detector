"""コマンドライン入口。

    python -m font_detector serve [--host 127.0.0.1] [--port 8000] [--open]
    python -m font_detector check-ocr
    python -m font_detector download-tessdata [jpn eng]
    python -m font_detector fonts input.pdf
    python -m font_detector scan input.pdf --font "MS 明朝" [--body-only] [--images] [-o out.pdf]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf

from . import annotate, matcher
from .analyzer import analyze


def _cmd_serve(args: argparse.Namespace) -> None:
    import threading
    import webbrowser

    import uvicorn

    from .server import create_app

    try:
        from .ocr import OcrUnavailableError, resolve_tesseract

        resolve_tesseract()
    except ImportError:
        print("注意: 画像判定用のライブラリが無いため、画像内の文字判定は使えません。", file=sys.stderr)
    except OcrUnavailableError as e:
        print(f"注意: 画像内の文字判定は使えません。{e}", file=sys.stderr)

    url = f"http://{args.host}:{args.port}/"
    print(f"Font Detector: {url}  (終了は Ctrl+C)", flush=True)
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


def _cmd_check_ocr(args: argparse.Namespace) -> None:
    """画像内の文字判定に使う Tesseract の状態を表示する。"""
    from .ocr import OcrUnavailableError, resolve_tesseract

    try:
        cfg = resolve_tesseract()
    except OcrUnavailableError as e:
        sys.exit(f"NG: {e}")
    print(f"Tesseract : {cfg.exe}")
    print(f"バージョン: {cfg.version}")
    print(f"言語データ: {cfg.tessdata or '(Tesseract の既定の場所)'}")
    print(f"使用言語  : {cfg.lang}  (利用可能: {', '.join(cfg.languages)})")
    print("OK: 画像内の文字判定を使えます。")


TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata/raw/main/{}.traineddata"


def _cmd_download_tessdata(args: argparse.Namespace) -> None:
    """Tesseract の言語データをプロジェクトの tessdata/ に取得する (セットアップ時に1回だけ)。"""
    import urllib.request

    from .ocr import PROJECT_TESSDATA_DIR

    PROJECT_TESSDATA_DIR.mkdir(exist_ok=True)
    for lang in args.langs:
        dest = PROJECT_TESSDATA_DIR / f"{lang}.traineddata"
        if dest.is_file() and not args.force:
            print(f"取得済み: {dest}")
            continue
        url = TESSDATA_URL.format(lang)
        print(f"取得中: {url}", flush=True)
        tmp = dest.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            sys.exit(f"言語データを取得できませんでした: {e}")
        tmp.replace(dest)
        print(f"  → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _cmd_fonts(args: argparse.Namespace) -> None:
    with pymupdf.open(args.pdf) as doc:
        a = analyze(doc)
    print(f"本文サイズ推定: {a.body_size}pt / 画像領域: {len(a.images)} / 透明テキスト除外: {a.invisible_chars}文字")
    for f in a.fonts.values():
        emb = {True: "埋込", False: "未埋込", None: "不明"}[f.embedded]
        print(f"{f.char_count:>8}文字  {f.display:<24} [{emb}]  {', '.join(sorted(f.raw_names))}")


def _cmd_scan(args: argparse.Namespace) -> None:
    data = Path(args.pdf).read_bytes()
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        a = analyze(doc)
        criteria = matcher.Criteria(
            query=args.font, body_only=args.body_only, min_score=args.min_score, method=args.method
        )
        image_matches = []
        if args.images:
            from . import imagescan

            image_matches = matcher.search_images(
                imagescan.scan(doc, a, use_ml=args.method == "ml"), a, criteria
            )
    matches = matcher.search(a, criteria)
    if not matches and not image_matches:
        print(f"'{args.font}' に該当する箇所はありません。", file=sys.stderr)
    for s in matches:
        print(f"p{s.page + 1}\t{a.fonts[s.font_key].display}\t{s.size}pt\t{s.text.strip()}")
    for m in image_matches:
        print(f"p{m.line.page + 1}\t{a.fonts[m.score.key].display}\t画像({m.method}) {m.score.score:.2f}\t{m.line.text}")
    if args.output:
        Path(args.output).write_bytes(annotate.annotate(data, a, matches, image_matches))
        total = len(matches) + len(image_matches)
        print(f"{total}件を注釈して {args.output} に保存しました。", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="font_detector")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="Web UI を起動する")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--open", action="store_true", help="起動後にブラウザで開く")
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("check-ocr", help="画像内の文字判定に使う Tesseract を確認する")
    p.set_defaults(func=_cmd_check_ocr)

    p = sub.add_parser("download-tessdata", help="Tesseract の言語データを tessdata/ に取得する")
    p.add_argument("langs", nargs="*", default=["jpn", "eng"])
    p.add_argument("--force", action="store_true", help="取得済みでも取り直す")
    p.set_defaults(func=_cmd_download_tessdata)

    p = sub.add_parser("fonts", help="PDF内のフォント一覧を表示する")
    p.add_argument("pdf")
    p.set_defaults(func=_cmd_fonts)

    p = sub.add_parser("scan", help="指定フォントの箇所を検出する")
    p.add_argument("pdf")
    p.add_argument("--font", required=True, help="フォント名 (カンマ区切りで複数可)")
    p.add_argument("--body-only", action="store_true", help="本文と推定される箇所のみ")
    p.add_argument("--images", action="store_true", help="画像内の文字もOCRで判定する")
    p.add_argument("--method", choices=["ml", "template"], default=matcher.DEFAULT_METHOD,
                   help="画像内の判定方式: ml=機械学習 (既定), template=字形照合")
    p.add_argument("--min-score", type=float, default=None,
                   help="画像内判定のしきい値 (既定: ml は %(ml).2f, template は %(template).2f)"
                   % matcher.DEFAULT_MIN_SCORES)
    p.add_argument("-o", "--output", help="注釈付きPDFの出力先")
    p.set_defaults(func=_cmd_scan)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
