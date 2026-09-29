"""画像領域の文字をOCRし、PDF内のフォントのどれで書かれているかを推定する。

判定方式は2つ:
  - "ml":       文書ごとに学習した分類器 (mlclassifier) による確率。PDFに無い文字も判定でき、
                PDFで使われていない書体は「その他」に分類される。
  - "template": 同じ文字の見本字形との類似度 (glyphs.similarity)。

結果 (行ごとの候補フォントとスコア) は指定フォントに依存しないので、ドキュメントごとに
一度だけ計算し、検索時は判定方式・しきい値・指定フォントで絞り込むだけにする。
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from typing import ContextManager

import numpy as np
import pymupdf

from . import glyphs
from .analyzer import Analysis, Rect
from .ocr import OcrEngine, OcrLine, default_engine
from .sysfonts import SystemFontIndex, default_index

MIN_SCALE, MAX_SCALE = 2.0, 5.0   # 画像領域の描画倍率 (画像の実解像度に合わせる)
MIN_REGION_PT = 20                # これより小さい画像は対象外
MIN_CHARS_COMPARED = 2            # 判定に必要な比較文字数
TEXT_OVERLAP_RATIO = 0.5          # テキストレイヤーと重なる行は重複として除外
MIN_CHAR_CONF = 60                # OCRの文字ごとの確信度 (0-100) がこれ未満の文字は比較しない

ML, TEMPLATE = "ml", "template"
log = logging.getLogger(__name__)


@dataclass
class FontScore:
    key: str
    score: float
    compared: int

    def to_json(self) -> dict:
        return {"font": self.key, "score": round(self.score, 4), "compared": self.compared}


@dataclass
class ImageLine:
    page: int
    bbox: Rect          # 未回転ページ座標
    view_bbox: Rect     # 表示座標
    text: str
    ocr_score: float
    usable_chars: int   # 形の比較に使える文字数
    rankings: dict[str, list[FontScore]] = field(default_factory=dict)  # 判定方式 → 候補 (スコア順)

    def best(self, method: str) -> FontScore | None:
        ranking = self.rankings.get(method)
        return ranking[0] if ranking else None

    def margin(self, method: str) -> float:
        ranking = self.rankings.get(method, [])
        if len(ranking) < 2:
            return 1.0
        return ranking[0].score - ranking[1].score

    def to_json(self) -> dict:
        return {
            "page": self.page + 1,
            "bbox": self.view_bbox,
            "text": self.text,
            "ocr_score": round(self.ocr_score, 4),
            "usable_chars": self.usable_chars,
            "rankings": {m: [r.to_json() for r in rs[:5]] for m, rs in self.rankings.items()},
        }


@dataclass
class ImageScan:
    lines: list[ImageLine]
    references: dict[str, glyphs.FontReferences]
    regions: int
    methods: list[str]                  # 利用できた判定方式
    training: dict | None = None        # 機械学習の学習データの概要

    def to_json(self) -> dict:
        return {
            "regions": self.regions,
            "lines": [ln.to_json() for ln in self.lines],
            "references": {k: v.to_json() for k, v in self.references.items()},
            "methods": self.methods,
            "training": self.training,
        }


@dataclass
class _Observed:
    line: OcrLine
    page: int
    origin: tuple[float, float]  # 切り出し画像の左上 (表示座標)
    scale: float
    chars: list[tuple[str, np.ndarray]]


def _native_scale(page: pymupdf.Page, xref: int, rect: pymupdf.Rect) -> float:
    try:
        width_px = page.parent.xref_get_key(xref, "Width")[1]
        return float(width_px) / max(rect.width, 1)
    except (ValueError, TypeError):
        return MIN_SCALE


def _overlaps_text(view_bbox: Rect, text_boxes: list[Rect]) -> bool:
    a = pymupdf.Rect(view_bbox)
    if a.is_empty:
        return False
    return any((a & pymupdf.Rect(b)).get_area() >= TEXT_OVERLAP_RATIO * a.get_area() for b in text_boxes)


def _score_line(obs: _Observed, refs: dict[str, glyphs.FontReferences]) -> list[FontScore]:
    sims: dict[str, list[float]] = {}
    for ch, ink in obs.chars:
        for key, ref in refs.items():
            ref_ink = ref.glyphs.get(ch)
            if ref_ink is not None:
                sims.setdefault(key, []).append(glyphs.similarity(ink, ref_ink))
    # 比較できた文字が少ない候補は、偶然の一致を避けるため除外する
    need = max(MIN_CHARS_COMPARED, len(obs.chars) // 2)
    ranking = [FontScore(k, float(np.mean(v)), len(v)) for k, v in sims.items() if len(v) >= need]
    ranking.sort(key=lambda r: -r.score)
    return ranking


def _train_classifier(doc, analysis, font_index, lock):
    """機械学習の分類器を学習する。学習できない (ライブラリが無い・クラス不足) 場合は None。"""
    try:
        from . import mlclassifier
    except ImportError as e:
        log.warning("機械学習による判定は使えません (%s がありません)", e.name)
        return None
    return mlclassifier.train(doc, analysis, font_index, mupdf_lock=lock)


def _ml_ranking(classifier, obs: _Observed) -> list[FontScore]:
    if classifier is None or len(obs.chars) < MIN_CHARS_COMPARED:
        return []
    return [FontScore(k, p, n) for k, p, n in classifier.rank_line(obs.chars)]


def scan(
    doc: pymupdf.Document,
    analysis: Analysis,
    mupdf_lock: ContextManager | None = None,
    engine: OcrEngine = default_engine,
    font_index: SystemFontIndex = default_index,
    use_ml: bool = True,
) -> ImageScan:
    lock = mupdf_lock or contextlib.nullcontext()

    # 大きい領域から採用し、既に採用した領域にほぼ含まれる領域 (重ねて配置された画像など) は省く。
    # ページ描画から切り出すので、重なった画像も合成後の見た目で一度だけOCRされる。
    by_page: dict[int, list] = {}
    for im in sorted(analysis.images, key=lambda i: -pymupdf.Rect(i.view_bbox).get_area()):
        r = pymupdf.Rect(im.view_bbox)
        if r.width < MIN_REGION_PT or r.height < MIN_REGION_PT:
            continue
        kept = by_page.setdefault(im.page, [])
        if any((r & pymupdf.Rect(k.view_bbox)).get_area() >= 0.9 * r.get_area() for k in kept):
            continue
        kept.append(im)

    text_boxes: dict[int, list[Rect]] = {}
    for s in analysis.spans:
        text_boxes.setdefault(s.page, []).append(s.view_bbox)

    # 全領域を切り出してから、まとめてOCRする (OCRの起動回数を減らす)
    crops: list[tuple[int, tuple[float, float], float, np.ndarray]] = []
    for pno, regions in by_page.items():
        with lock:
            page = doc[pno]
            scale = min(MAX_SCALE, max(MIN_SCALE, max(
                _native_scale(page, im.xref, pymupdf.Rect(im.bbox)) for im in regions)))
            gray = glyphs._page_gray(page, scale)
        for im in regions:
            x0, y0, x1, y1 = (round(v * scale) for v in im.view_bbox)
            x0, y0 = max(0, x0), max(0, y0)
            crop = gray[y0:y1, x0:x1]
            if crop.size:
                crops.append((pno, (x0 / scale, y0 / scale), scale, crop))

    observed: list[_Observed] = []
    ocr_results = engine.recognize_many([c[3] for c in crops])
    for (pno, origin, scale, crop), lines in zip(crops, ocr_results):
        for line in lines:
            chars = []
            for oc in line.chars:
                if oc.conf < MIN_CHAR_CONF or not glyphs.informative(oc.text):
                    continue
                ink = glyphs.isolate_glyph(crop, oc.box, line.box)
                if ink is not None and max(ink.shape) >= glyphs.MIN_INK_PX:
                    chars.append((oc.text, ink))
            observed.append(_Observed(line, pno, origin, scale, chars))

    needed = {c for o in observed for c, _ in o.chars}
    with lock:
        refs = glyphs.collect_pdf_references(doc, analysis, needed)
    glyphs.add_system_references(refs, analysis, needed, font_index)
    classifier = _train_classifier(doc, analysis, font_index, lock) if use_ml and observed else None

    lines: list[ImageLine] = []
    for o in observed:
        bx0, by0, bx1, by1 = o.line.box
        ox, oy = o.origin
        view = (ox + bx0 / o.scale, oy + by0 / o.scale, ox + bx1 / o.scale, oy + by1 / o.scale)
        if _overlaps_text(view, text_boxes.get(o.page, [])):
            continue
        with lock:
            r = pymupdf.Rect(view) * doc[o.page].derotation_matrix
        r.normalize()
        lines.append(ImageLine(
            page=o.page,
            bbox=tuple(r),
            view_bbox=view,
            text=o.line.text,
            ocr_score=o.line.score,
            usable_chars=len(o.chars),
            rankings={TEMPLATE: _score_line(o, refs), **({ML: _ml_ranking(classifier, o)} if classifier else {})},
        ))
    return ImageScan(
        lines=lines,
        references=refs,
        regions=sum(len(v) for v in by_page.values()),
        methods=([ML] if classifier else []) + [TEMPLATE],
        training=classifier.summary.to_json() if classifier else None,
    )
