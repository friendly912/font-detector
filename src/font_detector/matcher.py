"""指定フォントに該当するspanの検索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import fontname
from .analyzer import Analysis, Span

if TYPE_CHECKING:
    from .imagescan import ImageLine, ImageScan

# 本文判定: 本文サイズとの許容差 (pt) と、ヘッダー・フッターとみなす上下余白の割合
BODY_SIZE_TOLERANCE = 1.0
MARGIN_RATIO = 0.06
# 画像内の行: 最上位候補の類似度がこれ以上なら、そのフォントとみなす
DEFAULT_MIN_SCORE = 0.85


@dataclass
class Criteria:
    font_keys: frozenset[str] = frozenset()
    query: str = ""
    body_only: bool = False
    min_score: float = DEFAULT_MIN_SCORE


def target_keys(analysis: Analysis, criteria: Criteria) -> set[str]:
    """一覧で選択されたフォントと、自由入力に該当するフォントのキー集合。"""
    keys = {k for k in criteria.font_keys if k in analysis.fonts}
    for q in (s for s in criteria.query.split(",") if s.strip()):
        keys |= {k for k, f in analysis.fonts.items() if fontname.query_matches(q, f.parsed)}
    return keys


def is_body(span: Span, analysis: Analysis) -> bool:
    if analysis.body_size is None:
        return True
    if abs(span.size - analysis.body_size) > BODY_SIZE_TOLERANCE:
        return False
    _, h = analysis.page_sizes[span.page]
    _, y0, _, y1 = span.view_bbox
    return y0 >= h * MARGIN_RATIO and y1 <= h * (1 - MARGIN_RATIO)


def search(analysis: Analysis, criteria: Criteria) -> list[Span]:
    keys = target_keys(analysis, criteria)
    if not keys:
        return []
    return [
        s
        for s in analysis.spans
        if s.font_key in keys and (not criteria.body_only or is_body(s, analysis))
    ]


def search_images(scan: ImageScan | None, analysis: Analysis, criteria: Criteria) -> list[ImageLine]:
    """画像内の行のうち、最上位候補が指定フォントでしきい値以上のもの。"""
    if scan is None:
        return []
    keys = target_keys(analysis, criteria)
    return [
        ln
        for ln in scan.lines
        if ln.best is not None and ln.best.key in keys and ln.best.score >= criteria.min_score
    ]
