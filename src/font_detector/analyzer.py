"""PDFのテキストレイヤーと画像領域の抽出。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from . import fontname

# フォント一覧とPDF注釈で共通に使う色 (RGB 0-255)
PALETTE = [
    (255, 196, 0),
    (0, 170, 255),
    (255, 90, 120),
    (60, 200, 110),
    (170, 110, 255),
    (255, 140, 40),
    (0, 200, 200),
    (200, 200, 40),
]

Rect = tuple[float, float, float, float]


@dataclass
class Span:
    page: int
    bbox: Rect        # 未回転ページ座標 (PDF注釈用)
    view_bbox: Rect   # 表示上のページ座標 (回転適用後)
    text: str
    font_key: str
    size: float


@dataclass
class ImageRegion:
    page: int
    bbox: Rect
    view_bbox: Rect
    xref: int


@dataclass
class FontInfo:
    key: str
    display: str
    family_id: str
    color: tuple[int, int, int]
    parsed: fontname.FontName
    raw_names: set[str] = field(default_factory=set)
    embedded: bool | None = None
    span_count: int = 0
    char_count: int = 0
    pages: set[int] = field(default_factory=set)

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "display": self.display,
            "family": self.family_id,
            "color": "#%02x%02x%02x" % self.color,
            "raw_names": sorted(self.raw_names),
            "embedded": self.embedded,
            "span_count": self.span_count,
            "char_count": self.char_count,
            "pages": sorted(p + 1 for p in self.pages),
        }


@dataclass
class Analysis:
    page_sizes: list[tuple[float, float]]
    spans: list[Span]
    images: list[ImageRegion]
    fonts: dict[str, FontInfo]
    body_size: float | None
    invisible_chars: int  # OCRの透明テキスト等、判定から除外した文字数

    def to_json(self) -> dict:
        return {
            "page_count": len(self.page_sizes),
            "page_sizes": [{"width": w, "height": h} for w, h in self.page_sizes],
            "fonts": [f.to_json() for f in self.fonts.values()],
            "images": [
                {"page": im.page + 1, "bbox": im.view_bbox} for im in self.images
            ],
            "body_size": self.body_size,
            "invisible_chars": self.invisible_chars,
        }


def _view_rect(rect: pymupdf.Rect, page: pymupdf.Page) -> Rect:
    r = rect * page.rotation_matrix
    r.normalize()
    return (r.x0, r.y0, r.x1, r.y1)


def _embedded_names(page: pymupdf.Page) -> dict[str, bool]:
    """ページで使われるフォントの埋め込み有無 (キー: 正規化名)。"""
    result: dict[str, bool] = {}
    for _xref, ext, _type, basefont, _name, _enc in page.get_fonts():
        key = fontname.parse(basefont).key
        result[key] = result.get(key, False) or ext != "n/a"
    return result


def analyze(doc: pymupdf.Document) -> Analysis:
    spans: list[Span] = []
    images: list[ImageRegion] = []
    fonts: dict[str, FontInfo] = {}
    parsed_cache: dict[str, fontname.FontName] = {}
    size_weights: Counter[float] = Counter()
    invisible = 0

    for pno, page in enumerate(doc):
        embedded = _embedded_names(page)
        text = page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)
        for block in text["blocks"]:
            for line in block.get("lines", []):
                merged: list[Span] = []
                for s in line["spans"]:
                    if not s["text"].strip():
                        continue
                    if s.get("alpha", 255) == 0:
                        invisible += len(s["text"].strip())
                        continue
                    raw = s["font"]
                    parsed = parsed_cache.get(raw) or parsed_cache.setdefault(raw, fontname.parse(raw))
                    info = fonts.get(parsed.key)
                    if info is None:
                        info = fonts[parsed.key] = FontInfo(
                            key=parsed.key,
                            display=parsed.display,
                            family_id=parsed.family_id,
                            color=PALETTE[len(fonts) % len(PALETTE)],
                            parsed=parsed,
                        )
                    info.raw_names.add(raw)
                    if parsed.key in embedded:
                        info.embedded = bool(info.embedded) or embedded[parsed.key]

                    rect = pymupdf.Rect(s["bbox"])
                    size = round(s["size"], 1)
                    prev = merged[-1] if merged else None
                    # Wordは同一書式の行を細かく分割するので、同じフォント・サイズの隣接spanを結合する
                    if prev and prev.font_key == parsed.key and prev.size == size:
                        r = pymupdf.Rect(prev.bbox) | rect
                        prev.bbox = tuple(r)
                        prev.view_bbox = _view_rect(r, page)
                        prev.text += s["text"]
                    else:
                        merged.append(
                            Span(pno, tuple(rect), _view_rect(rect, page), s["text"], parsed.key, size)
                        )
                for sp in merged:
                    n = len(sp.text.strip())
                    info = fonts[sp.font_key]
                    info.span_count += 1
                    info.char_count += n
                    info.pages.add(pno)
                    size_weights[round(sp.size * 2) / 2] += n
                spans.extend(merged)

        for img in page.get_images(full=True):
            for rect in page.get_image_rects(img[0]):
                if rect.is_empty or rect.is_infinite:
                    continue
                images.append(ImageRegion(pno, tuple(rect), _view_rect(rect, page), img[0]))

    # 未埋め込み判定がつかなかったフォント (Type3 等) は None のまま
    body_size = size_weights.most_common(1)[0][0] if size_weights else None
    ordered = dict(sorted(fonts.items(), key=lambda kv: -kv[1].char_count))
    for i, info in enumerate(ordered.values()):
        info.color = PALETTE[i % len(PALETTE)]

    return Analysis(
        page_sizes=[(p.rect.width, p.rect.height) for p in doc],
        spans=spans,
        images=images,
        fonts=ordered,
        body_size=body_size,
        invisible_chars=invisible,
    )
