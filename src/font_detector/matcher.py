"""指定フォントに該当するspanの検索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import fontname
from .analyzer import Analysis, Span

if TYPE_CHECKING:
    from .imagescan import FontScore, ImageLine, ImageScan

# 本文判定: 本文サイズとの許容差 (pt) と、ヘッダー・フッターとみなす上下余白の割合
BODY_SIZE_TOLERANCE = 1.0
MARGIN_RATIO = 0.06
# 画像内の行: 最上位候補のスコアがこれ以上なら、そのフォントとみなす (判定方式ごと)
DEFAULT_METHOD = "ml"
DEFAULT_MIN_SCORES = {"ml": 0.55, "template": 0.85}
DEFAULT_MIN_SCORE = DEFAULT_MIN_SCORES["template"]


@dataclass
class Criteria:
    font_keys: frozenset[str] = frozenset()
    query: str = ""
    body_only: bool = False
    min_score: float | None = None      # None なら判定方式の既定値
    method: str = DEFAULT_METHOD

    def threshold(self) -> float:
        return self.min_score if self.min_score is not None else DEFAULT_MIN_SCORES.get(self.method, 0.5)


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


@dataclass
class ImageMatch:
    line: ImageLine
    score: FontScore  # 該当と判定した候補
    method: str


def search_images(scan: ImageScan | None, analysis: Analysis, criteria: Criteria) -> list[ImageMatch]:
    """画像内の行のうち、最上位候補が指定フォントでしきい値以上のもの。

    判定方式が使えない (機械学習の学習ができなかった等) 場合は、字形照合に切り替える。
    """
    if scan is None:
        return []
    method = effective_method(scan, criteria)
    keys = target_keys(analysis, criteria)
    threshold = criteria.threshold() if method == criteria.method else DEFAULT_MIN_SCORES[method]
    result = []
    for ln in scan.lines:
        best = ln.best(method)
        if best is not None and best.key in keys and best.score >= threshold:
            result.append(ImageMatch(ln, best, method))
    return result


def effective_method(scan: ImageScan, criteria: Criteria) -> str:
    return criteria.method if criteria.method in scan.methods else scan.methods[-1]
