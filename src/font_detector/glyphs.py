"""字形の正規化・比較と、フォントごとの見本字形ライブラリ。

見本は次の順で集める。
  1. PDFのテキストレイヤー: 埋め込みフォントで描かれた文字をページ画像から切り出す
  2. システムフォント: 同名のフォントがOSにあれば、不足する文字をそれで描画する
非埋め込みフォントの文字はビューアの代替フォントで描かれるため、1 の見本には使わない。
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np
import pymupdf

from .analyzer import Analysis
from . import fontname
from .sysfonts import SystemFont, SystemFontIndex, cmap

GLYPH_SIZE = 32
REF_SCALE = 6.0          # 見本を切り出すときのページ描画倍率 (11pt ≒ 66px)
SYSTEM_RENDER_PX = 96
MIN_INK_PX = 6           # これより小さい字形は比較しない
# 形がフォントによってほとんど変わらず、判定の役に立たない文字
_UNINFORMATIVE = set("ー一－-‐_=~〜|｜lIi1!！'\"`・.,:;")


def informative(ch: str) -> bool:
    if ch in _UNINFORMATIVE or not ch.strip():
        return False
    return unicodedata.category(ch)[0] not in ("P", "Z", "C")


def ink_crop(gray: np.ndarray) -> np.ndarray | None:
    """グレースケール画像 (0=黒) から、インク量 (0..1) の外接矩形を切り出す。"""
    if gray.size == 0:
        return None
    ink = 1.0 - gray.astype(np.float32) / 255.0
    lo = float(np.percentile(ink, 5))
    ink = ink - lo
    hi = float(ink.max())
    if hi < 0.25:
        return None
    ink = np.clip(ink / hi, 0.0, 1.0)
    ys, xs = np.nonzero(ink > 0.35)
    if len(xs) == 0:
        return None
    return ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def isolate_glyph(gray: np.ndarray, box: tuple[int, int, int, int], line_box: tuple[int, int, int, int]) -> np.ndarray | None:
    """OCRの文字枠から字形を切り出す。

    OCRの文字枠は概略なので、枠を左右に広げたうえで、重心が元の枠内にある連結成分だけを残す。
    (枠からはみ出した自分の画は拾い、枠に入り込んだ隣の文字の画は捨てる)
    """
    x0, _, x1, _ = box
    _, ly0, _, ly1 = line_box
    pad = max(2, (x1 - x0) // 3)
    wx0, wx1 = max(0, x0 - pad), min(gray.shape[1], x1 + pad)
    wy0, wy1 = max(0, ly0 - 2), min(gray.shape[0], ly1 + 2)
    window = gray[wy0:wy1, wx0:wx1]
    if window.size == 0:
        return None
    mask = (window < 160).astype(np.uint8)
    n, labels, _stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        cx = centroids[i][0] + wx0
        keep[i] = x0 <= cx < x1
    if not keep.any():
        return None
    selected = cv2.dilate(keep[labels].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    isolated = np.where(selected, window, 255).astype(np.uint8)
    return ink_crop(isolated)


def normalize(ink: np.ndarray) -> np.ndarray:
    """縦横比を保って GLYPH_SIZE 四方に収め、零平均・単位長のベクトルにする。"""
    h, w = ink.shape
    s = (GLYPH_SIZE - 2) / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    small = cv2.resize(ink, (nw, nh), interpolation=interp)
    canvas = np.zeros((GLYPH_SIZE, GLYPH_SIZE), np.float32)
    y, x = (GLYPH_SIZE - nh) // 2, (GLYPH_SIZE - nw) // 2
    canvas[y : y + nh, x : x + nw] = small
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
    v = canvas.ravel() - canvas.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def similarity(observed: np.ndarray, reference: np.ndarray) -> float:
    """観測字形と見本字形の類似度 (-1..1)。

    見本は高解像度なので、観測と同じ画素数まで縮小してから比べる (解像度の劣化をそろえる)。
    """
    s = max(observed.shape) / max(reference.shape)
    if s < 1:
        rh, rw = reference.shape
        reference = cv2.resize(
            reference, (max(1, round(rw * s)), max(1, round(rh * s))), interpolation=cv2.INTER_AREA
        )
    return float(np.dot(normalize(observed), normalize(reference)))


# ---------------------------------------------------------------- reference library


@dataclass
class FontReferences:
    glyphs: dict[str, np.ndarray] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)  # "pdf" / "system"

    def to_json(self) -> dict:
        return {"glyphs": len(self.glyphs), "sources": sorted(self.sources)}


def _page_gray(page: pymupdf.Page, scale: float) -> np.ndarray:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def collect_pdf_references(
    doc: pymupdf.Document, analysis: Analysis, needed: set[str]
) -> dict[str, FontReferences]:
    """テキストレイヤーから、必要な文字の見本を埋め込みフォントごとに切り出す。"""
    return collect_pdf_glyphs(doc, analysis, lambda key, c: c in needed)


def collect_pdf_glyphs(
    doc: pymupdf.Document, analysis: Analysis, want: Callable[[str, str], bool]
) -> dict[str, FontReferences]:
    """テキストレイヤーから、want(フォントキー, 文字) が真の文字の字形を1つずつ切り出す。"""
    wanted: dict[int, list[tuple[str, str, pymupdf.Rect]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for pno, page in enumerate(doc):
        if page.rotation != 0:
            continue
        raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_TEXT)
        for block in raw["blocks"]:
            for line in block.get("lines", []):
                if abs(line["dir"][1]) > 1e-3:  # 横書き以外は対象外
                    continue
                for span in line["spans"]:
                    if span.get("alpha", 255) == 0:
                        continue
                    key = fontname.parse(span["font"]).key
                    info = analysis.fonts.get(key)
                    if info is None or not info.embedded:
                        continue
                    for ch in span["chars"]:
                        c = ch["c"]
                        if (key, c) not in seen and want(key, c):
                            seen.add((key, c))
                            wanted[pno].append((key, c, pymupdf.Rect(ch["bbox"])))

    refs: dict[str, FontReferences] = defaultdict(FontReferences)
    for pno, items in wanted.items():
        gray = _page_gray(doc[pno], REF_SCALE)
        for key, c, rect in items:
            r = (rect * REF_SCALE).irect & pymupdf.IRect(0, 0, gray.shape[1], gray.shape[0])
            if r.is_empty:
                continue
            ink = ink_crop(gray[r.y0 : r.y1, r.x0 : r.x1])
            if ink is not None:
                refs[key].glyphs[c] = ink
                refs[key].sources.add("pdf")
    return dict(refs)


@lru_cache(maxsize=32)
def _pil_font(path: str, index: int):
    from PIL import ImageFont

    return ImageFont.truetype(path, SYSTEM_RENDER_PX, index=index)


def render_system_glyph(path: str, index: int, ch: str) -> np.ndarray | None:
    from PIL import Image, ImageDraw

    font = _pil_font(path, index)
    img = Image.new("L", (SYSTEM_RENDER_PX * 2, SYSTEM_RENDER_PX * 2), 255)
    ImageDraw.Draw(img).text((SYSTEM_RENDER_PX // 2, SYSTEM_RENDER_PX // 2), ch, font=font, fill=0)
    return ink_crop(np.asarray(img))


def add_system_references(
    refs: dict[str, FontReferences], analysis: Analysis, needed: set[str], index: SystemFontIndex
) -> None:
    """PDFから集められなかった文字を、同名のシステムフォントで補う。"""
    for key in analysis.fonts:
        entry = refs.get(key) or FontReferences()
        missing = needed - entry.glyphs.keys()
        if not missing:
            continue
        sf = index.lookup(key)
        if sf is None:
            continue
        covered = cmap(sf)
        for c in missing:
            if ord(c) not in covered:
                continue
            ink = render_system_glyph(sf.path, sf.index, c)
            if ink is not None:
                entry.glyphs[c] = ink
                entry.sources.add("system")
        if entry.glyphs:
            refs[key] = entry


def render_font_glyphs(font: SystemFont, chars: set[str]) -> dict[str, np.ndarray]:
    """システムフォントで文字を描画する (フォントに無い文字は除く)。"""
    covered = cmap(font)
    out: dict[str, np.ndarray] = {}
    for c in sorted(chars):
        if ord(c) in covered:
            ink = render_system_glyph(font.path, font.index, c)
            if ink is not None:
                out[c] = ink
    return out
