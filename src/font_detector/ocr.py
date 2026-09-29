"""Tesseract OCR のラッパー。行ごとの文字列と、1文字単位の位置を返す。

Tesseract (LSTM) の文字枠は位置がずれることが多いため、Tesseract からは行の枠・文字列・
文字ごとの確信度だけを使い、1文字ごとの区切りは行画像のインクの切れ目から求める。

Tesseract 本体はローカルにインストールされたものを使う。探す順序:
  1. 環境変数 FONT_DETECTOR_TESSERACT (tesseract 実行ファイルのパス)
  2. PATH 上の tesseract
  3. Windows の標準インストール先 (C:\\Program Files\\Tesseract-OCR など)
言語データ (jpn.traineddata 等) は次の順で探す:
  1. 環境変数 FONT_DETECTOR_TESSDATA
  2. プロジェクト直下の tessdata/
  3. Tesseract 本体の既定の場所
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 (画素)

DEFAULT_LANG = "jpn+eng"
PSM_SPARSE_TEXT = "11"   # 図やスクリーンショット内のまばらな文字にも対応するページ分割モード
TIMEOUT_SEC = 120          # 1回の起動あたりの基本の制限時間
TIMEOUT_PER_IMAGE_SEC = 20
BATCH_SIZE = 50            # 1回の起動でまとめて処理する画像の数 (起動と言語データ読込の回数を減らす)
PROJECT_TESSDATA_DIR = Path(__file__).resolve().parents[2] / "tessdata"
HALF_WIDTH_ADVANCE = 0.55  # 全角1文字に対する半角文字の送り幅の目安


class OcrUnavailableError(RuntimeError):
    """Tesseract 本体または言語データが見つからない。"""


@dataclass
class OcrChar:
    text: str
    box: Box
    conf: float = 100.0  # 0..100


@dataclass
class OcrLine:
    text: str
    box: Box
    score: float  # 0..1
    chars: list[OcrChar] = field(default_factory=list)


# ---------------------------------------------------------------- Tesseract の場所


@dataclass(frozen=True)
class TesseractConfig:
    exe: str
    tessdata: str | None
    lang: str
    version: str
    languages: tuple[str, ...]


def _candidate_executables() -> list[str]:
    found: list[str] = []
    env = os.environ.get("FONT_DETECTOR_TESSERACT")
    if env:
        found.append(env)
    which = shutil.which("tesseract")
    if which:
        found.append(which)
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                     os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
            if base:
                found.append(os.path.join(base, "Tesseract-OCR", "tesseract.exe"))
    return found


def _run(args: list[str], input: bytes | None = None, timeout: float = TIMEOUT_SEC) -> subprocess.CompletedProcess:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    return subprocess.run(args, input=input, capture_output=True, timeout=timeout, creationflags=flags)


def _tessdata_dir(lang_files: list[str]) -> str | None:
    env = os.environ.get("FONT_DETECTOR_TESSDATA")
    if env:
        return env
    if all((PROJECT_TESSDATA_DIR / f).is_file() for f in lang_files):
        return str(PROJECT_TESSDATA_DIR)
    return None


@lru_cache(maxsize=1)
def resolve_tesseract() -> TesseractConfig:
    """使用する Tesseract と言語データを確認する。使えなければ OcrUnavailableError。"""
    lang = os.environ.get("FONT_DETECTOR_OCR_LANG", DEFAULT_LANG)
    needed = [p for p in lang.split("+") if p]
    exe = next((c for c in _candidate_executables() if os.path.isfile(c)), None)
    if exe is None:
        raise OcrUnavailableError(
            "Tesseract が見つかりません。Tesseract をインストールするか、"
            "環境変数 FONT_DETECTOR_TESSERACT に tesseract の実行ファイルのパスを指定してください。"
        )
    try:
        version = _run([exe, "--version"]).stdout.decode(errors="replace").splitlines()[0].strip()
        tessdata = _tessdata_dir([f"{p}.traineddata" for p in needed])
        args = [exe, "--list-langs"] + (["--tessdata-dir", tessdata] if tessdata else [])
        out = _run(args).stdout.decode(errors="replace").splitlines()
    except (OSError, subprocess.SubprocessError, IndexError) as e:
        raise OcrUnavailableError(f"Tesseract ({exe}) を実行できません: {e}")
    languages = tuple(line.strip() for line in out[1:] if line.strip())
    missing = [p for p in needed if p not in languages]
    if missing:
        raise OcrUnavailableError(
            f"Tesseract の言語データ {', '.join(m + '.traineddata' for m in missing)} がありません。"
            f"Tesseract のインストール時に追加言語 (Japanese) を選ぶか、"
            f"{PROJECT_TESSDATA_DIR} に配置してください。"
        )
    return TesseractConfig(exe, tessdata, lang, version, languages)


# ---------------------------------------------------------------- hOCR の解析


def _title_props(title: str) -> dict[str, list[str]]:
    props: dict[str, list[str]] = {}
    for part in title.split(";"):
        items = part.split()
        if items:
            props[items[0]] = items[1:]
    return props


class _HocrParser(HTMLParser):
    """ocr_line / ocrx_word / ocrx_cinfo (文字ごとの確信度) を取り出す。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[list[dict]] = []
        self.lines: list[dict] = []  # 現在のページの行
        self._stack: list[str | None] = []
        self._cinfo: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class")
        props = _title_props(a.get("title", ""))
        if cls == "ocr_page":
            self.lines = []
            self.pages.append(self.lines)
        elif cls == "ocr_line" or cls in ("ocr_caption", "ocr_textfloat", "ocr_header"):
            bbox = [int(v) for v in props.get("bbox", ["0"] * 4)]
            self.lines.append({"bbox": tuple(bbox), "words": []})
        elif cls == "ocrx_word" and self.lines:
            conf = float(props.get("x_wconf", ["0"])[0])
            self.lines[-1]["words"].append({"conf": conf, "chars": [], "text": ""})
        elif cls == "ocrx_cinfo" and self.lines and self.lines[-1]["words"]:
            self._cinfo = {"conf": float(props.get("x_conf", ["0"])[0]), "text": ""}
        if tag not in ("br", "meta"):
            self._stack.append(cls)

    def handle_endtag(self, tag):
        if not self._stack:
            return
        cls = self._stack.pop()
        if cls == "ocrx_cinfo" and self._cinfo is not None:
            if self._cinfo["text"]:
                self.lines[-1]["words"][-1]["chars"].append(self._cinfo)
            self._cinfo = None

    def handle_data(self, data):
        if self._cinfo is not None:
            self._cinfo["text"] += data.strip()
        elif self._stack and self._stack[-1] == "ocrx_word" and self.lines and self.lines[-1]["words"]:
            self.lines[-1]["words"][-1]["text"] += data.strip()


HocrLine = tuple[Box, list[tuple[str, float]], str]


def parse_hocr_pages(hocr: str) -> list[list[HocrLine]]:
    """hOCR → ページ (入力画像) ごとの [(行の枠, [(文字, 確信度)], 行の文字列)]"""
    parser = _HocrParser()
    parser.feed(hocr)
    return [_page_lines(lines) for lines in parser.pages]


def parse_hocr(hocr: str) -> list[HocrLine]:
    """1ページ分の hOCR を解析する。"""
    pages = parse_hocr_pages(hocr)
    return pages[0] if pages else []


def _page_lines(lines: list[dict]) -> list[HocrLine]:
    result = []
    for line in lines:
        chars: list[tuple[str, float]] = []
        text = ""
        for word in line["words"]:
            wchars = word["chars"] or [{"text": c, "conf": word["conf"]} for c in word["text"]]
            wtext = "".join(c["text"] for c in wchars)
            if not wtext:
                continue
            # Tesseract は日本語の文字間にも空白を入れるので、英数字同士の間だけ空白を残す
            if text and text[-1].isascii() and text[-1].isalnum() and wtext[0].isascii() and wtext[0].isalnum():
                text += " "
            text += wtext
            for c in wchars:
                for ch in c["text"]:
                    chars.append((ch, c["conf"]))
        if chars:
            result.append((line["bbox"], chars, text))
    return result


# ---------------------------------------------------------------- 1文字ずつの区切り


def ink_segments(gray: np.ndarray, threshold: int = 160) -> list[tuple[int, int]]:
    """インクのある列の連続区間 [(開始, 終了)]。"""
    cols = (gray < threshold).any(axis=0)
    segments: list[tuple[int, int]] = []
    start = None
    for x, on in enumerate(cols):
        if on and start is None:
            start = x
        elif not on and start is not None:
            segments.append((start, x))
            start = None
    if start is not None:
        segments.append((start, len(cols)))
    return segments


def _advance(ch: str) -> float:
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else HALF_WIDTH_ADVANCE


def segment_line(gray: np.ndarray, line_box: Box, chars: list[tuple[str, float]]) -> list[OcrChar]:
    """行画像を認識結果の文字に対応づけて、1文字ずつの枠を求める。

    インクの切れ目の数が文字数と一致すれば1対1で対応させる。一致しない場合
    (「川」のように画が離れた字、接触した文字) は、全角・半角の送り幅から各文字の
    位置を見積もり、インクの塊を中心位置で割り当てる。塊が割り当たらない文字は除く。
    """
    x0, y0, x1, y1 = line_box
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0 or not chars:
        return []
    segs = ink_segments(crop)
    if not segs:
        return []
    if len(segs) == len(chars):
        return [OcrChar(c, (x0 + a, y0, x0 + b, y1), conf) for (c, conf), (a, b) in zip(chars, segs)]

    weights = [_advance(c) for c, _ in chars]
    left, right = segs[0][0], segs[-1][1]
    unit = (right - left) / sum(weights)
    bounds = np.cumsum([0.0] + weights) * unit + left
    groups: list[list[tuple[int, int]]] = [[] for _ in chars]
    for a, b in segs:
        idx = int(np.searchsorted(bounds, (a + b) / 2, side="right")) - 1
        groups[min(max(idx, 0), len(chars) - 1)].append((a, b))

    result: list[OcrChar] = []
    for (c, conf), w, group in zip(chars, weights, groups):
        if not group:
            continue
        a, b = group[0][0], group[-1][1]
        if b - a > 1.4 * w * unit:  # 隣の文字と接触して分けられない
            continue
        result.append(OcrChar(c, (x0 + a, y0, x0 + b, y1), conf))
    return result


# ---------------------------------------------------------------- 実行


class OcrEngine:
    def __init__(self, config: TesseractConfig | None = None) -> None:
        self._config = config

    @property
    def config(self) -> TesseractConfig:
        if self._config is None:
            self._config = resolve_tesseract()
        return self._config

    def recognize(self, gray: np.ndarray, min_score: float = 0.5) -> list[OcrLine]:
        return self.recognize_many([gray], min_score)[0]

    def recognize_many(self, images: list[np.ndarray], min_score: float = 0.5) -> list[list[OcrLine]]:
        """複数の画像を、少ない起動回数でまとめてOCRする。戻り値は入力と同じ順序。"""
        results: list[list[OcrLine]] = []
        for i in range(0, len(images), BATCH_SIZE):
            results.extend(self._recognize_batch(images[i : i + BATCH_SIZE], min_score))
        return results

    def _recognize_batch(self, images: list[np.ndarray], min_score: float) -> list[list[OcrLine]]:
        cfg = self.config
        with tempfile.TemporaryDirectory(prefix="font-detector-") as tmp:
            paths = []
            for i, gray in enumerate(images):
                path = os.path.join(tmp, f"{i:04d}.png")
                cv2.imwrite(path, gray)
                paths.append(path)
            listfile = os.path.join(tmp, "images.txt")
            with open(listfile, "w", encoding="utf-8") as f:
                f.write("\n".join(paths) + "\n")

            args = [cfg.exe, listfile, "stdout"]
            if cfg.tessdata:
                args += ["--tessdata-dir", cfg.tessdata]
            # 出力形式は "hocr" 設定ファイルではなく変数で指定する
            # (設定ファイルは tessdata/configs/ にあり、言語データだけを置いた環境には無いため)
            args += ["-l", cfg.lang, "--psm", PSM_SPARSE_TEXT, "--dpi", "300",
                     "-c", "tessedit_create_hocr=1", "-c", "hocr_font_info=0", "-c", "hocr_char_boxes=1"]
            proc = _run(args, timeout=TIMEOUT_SEC + TIMEOUT_PER_IMAGE_SEC * len(images))
        if proc.returncode != 0:
            raise RuntimeError(f"Tesseract がエラーで終了しました: {proc.stderr.decode(errors='replace')[:500]}")

        pages = parse_hocr_pages(proc.stdout.decode("utf-8", errors="replace"))
        if len(pages) != len(images):
            stderr = proc.stderr.decode(errors="replace").strip()[-500:]
            raise RuntimeError(
                f"Tesseract の出力ページ数 ({len(pages)}) が入力画像数 ({len(images)}) と一致しません。{stderr}"
            )
        results = []
        for gray, page in zip(images, pages):
            lines: list[OcrLine] = []
            for box, chars, text in page:
                score = float(np.mean([conf for _, conf in chars])) / 100
                if score < min_score or not text.strip():
                    continue
                lines.append(OcrLine(text, box, score, segment_line(gray, box, chars)))
            results.append(lines)
        return results


default_engine = OcrEngine()
