"""検出結果をハイライト注釈としてPDFに書き込む。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymupdf

from .analyzer import Analysis, Span

if TYPE_CHECKING:
    from .imagescan import ImageLine

ANNOT_TITLE = "font-detector"


def annotate(
    pdf_bytes: bytes,
    analysis: Analysis,
    matches: list[Span],
    image_matches: list[ImageLine] = (),
) -> bytes:
    """元PDFのコピーに注釈を付けて返す (元データは変更しない)。

    テキストレイヤーの一致はハイライト、画像内の推定一致は破線の枠で示す。
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        for span in matches:
            page = doc[span.page]
            font = analysis.fonts[span.font_key]
            annot = page.add_highlight_annot(pymupdf.Rect(span.bbox))
            annot.set_colors(stroke=[c / 255 for c in font.color])
            annot.set_info(title=ANNOT_TITLE, content=f"{font.display} / {span.size}pt")
            annot.update()
        for line in image_matches:
            page = doc[line.page]
            font = analysis.fonts[line.best.key]
            annot = page.add_rect_annot(pymupdf.Rect(line.bbox))
            annot.set_colors(stroke=[c / 255 for c in font.color])
            annot.set_border(width=1.5, dashes=[3, 2])
            annot.set_info(
                title=ANNOT_TITLE,
                content=f"画像内推定: {font.display} (類似度 {line.best.score:.2f})\n{line.text}",
            )
            annot.update()
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()
