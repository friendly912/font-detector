import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from conftest import NOTO, SANS, SERIF
from font_detector import annotate, glyphs, matcher
from font_detector.analyzer import analyze
from font_detector.ocr import OcrUnavailableError, resolve_tesseract, segment_line
from font_detector.server import create_app
from font_detector.sysfonts import SystemFontIndex

SERIF_KEY, SANS_KEY = "notoserifcjkjp", "notosanscjkjp"


def _require_tesseract():
    try:
        resolve_tesseract()
    except OcrUnavailableError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def scanned(image_pdf):
    from font_detector import imagescan

    _require_tesseract()
    doc = pymupdf.open(stream=image_pdf, filetype="pdf")
    a = analyze(doc)
    # システムフォントを使わず、PDF内の見本だけで判定できることを確かめる
    return a, imagescan.scan(doc, a, font_index=SystemFontIndex([]))


def _line(scan, prefix):
    return next(ln for ln in scan.lines if ln.text.startswith(prefix))


def test_image_lines_are_attributed_to_pdf_fonts(scanned):
    a, scan = scanned
    assert set(scan.references) == {SERIF_KEY, SANS_KEY}
    assert all(r.sources == {"pdf"} for r in scan.references.values())

    serif = _line(scan, "画像の中")
    sans = _line(scan, "この行は")
    assert serif.best.key == SERIF_KEY and serif.best.score >= matcher.DEFAULT_MIN_SCORE
    assert sans.best.key == SANS_KEY and sans.best.score >= matcher.DEFAULT_MIN_SCORE


def test_font_not_in_pdf_stays_below_threshold(scanned):
    _, scan = scanned
    other = _line(scan, "別の書体")
    assert other.best.score < matcher.DEFAULT_MIN_SCORE


def test_search_images_respects_selection_and_threshold(scanned):
    a, scan = scanned
    hits = matcher.search_images(scan, a, matcher.Criteria(font_keys=frozenset({SERIF_KEY})))
    assert [ln.text[:4] for ln in hits] == ["画像の中"]
    assert matcher.search_images(scan, a, matcher.Criteria(font_keys=frozenset({SERIF_KEY}), min_score=0.99)) == []


def test_image_line_coordinates_fall_inside_image(scanned):
    a, scan = scanned
    img = pymupdf.Rect(a.images[0].view_bbox)
    for ln in scan.lines:
        assert img.contains(pymupdf.Rect(ln.view_bbox))


def test_annotate_marks_image_lines(image_pdf, scanned):
    a, scan = scanned
    hits = matcher.search_images(scan, a, matcher.Criteria(font_keys=frozenset({SERIF_KEY, SANS_KEY})))
    out = annotate.annotate(image_pdf, a, [], hits)
    with pymupdf.open(stream=out, filetype="pdf") as doc:
        types = [an.type[1] for an in doc[0].annots()]
    assert types == ["Square"] * len(hits) and len(hits) == 2


# ---------------------------------------------------------------- 部品


def _draw(text, font, px=40, width=400):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (width, px * 2), 255)
    ImageDraw.Draw(img).text((10, px // 2), text, font=ImageFont.truetype(font, px), fill=0)
    return np.asarray(img)


def test_isolate_glyph_drops_neighbour_strokes():
    gray = _draw("画像", SERIF)
    cols = np.nonzero((gray < 128).any(axis=0))[0]
    mid = (cols.min() + cols.max()) // 2
    # わざと右の文字に食い込んだ枠を与えても、右の文字の画は拾わない
    ink = glyphs.isolate_glyph(gray, (cols.min(), 0, mid + 6, gray.shape[0]), (0, 0, gray.shape[1], gray.shape[0]))
    alone = glyphs.ink_crop(_draw("画", SERIF))
    assert ink is not None
    assert glyphs.similarity(ink, alone) > 0.95


def _full(gray):
    return (0, 0, gray.shape[1], gray.shape[0])


def test_segment_line_one_to_one():
    gray = _draw("Font", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    chars = segment_line(gray, _full(gray), [(c, 90.0) for c in "Font"])
    assert [c.text for c in chars] == list("Font")
    assert all(a.box[2] <= b.box[0] for a, b in zip(chars, chars[1:]))


def test_segment_line_groups_split_glyphs_by_position():
    # 「川」「八」は画が離れているので、インクの切れ目は文字数より多い
    text = "川の八文字"
    gray = _draw(text, SERIF, width=300)
    chars = segment_line(gray, _full(gray), [(c, 90.0) for c in text])
    assert [c.text for c in chars] == list(text)
    for oc in chars:
        ink = glyphs.isolate_glyph(gray, oc.box, _full(gray))
        alone = glyphs.ink_crop(_draw(oc.text, SERIF))
        assert glyphs.similarity(ink, alone) > 0.95, oc.text


def test_similarity_separates_mincho_and_gothic():
    # 1文字では差が小さい字もあるので、行の判定と同じく複数文字の平均で比べる
    def mean_sim(observed_font, ref_font):
        sims = []
        for ch in "書のは文字ますで":
            ref = glyphs.ink_crop(_draw(ch, ref_font, px=96, width=200))
            observed = glyphs.ink_crop(_draw(ch, observed_font, px=20))
            sims.append(glyphs.similarity(observed, ref))
        return float(np.mean(sims))

    assert mean_sim(SERIF, SERIF) > mean_sim(SERIF, SANS) + 0.05
    assert mean_sim(SANS, SANS) > mean_sim(SANS, SERIF) + 0.05


def test_system_font_index_resolves_pdf_font_keys():
    index = SystemFontIndex([__import__("pathlib").Path(NOTO)])
    sf = index.lookup(SERIF_KEY)
    assert sf is not None and sf.path.endswith("NotoSerifCJK-Regular.ttc")
    assert glyphs.render_system_glyph(sf.path, sf.index, "書") is not None


def test_api_image_scan(image_pdf):
    _require_tesseract()
    client = TestClient(create_app())
    doc = client.post("/api/documents", files={"file": ("img.pdf", image_pdf, "application/pdf")}).json()
    body = {"font_keys": [SERIF_KEY]}

    r = client.post(f"/api/documents/{doc['id']}/search", json={**body, "include_images": True})
    assert all(m["source"] == "text" for m in r.json()["matches"])  # スキャン前は画像の結果なし

    scan = client.post(f"/api/documents/{doc['id']}/image-scan").json()
    assert scan["regions"] == 1 and len(scan["lines"]) == 3

    r = client.post(f"/api/documents/{doc['id']}/search", json={**body, "include_images": True})
    image_hits = [m for m in r.json()["matches"] if m["source"] == "image"]
    assert len(image_hits) == 1 and image_hits[0]["font"] == SERIF_KEY

    r = client.post(f"/api/documents/{doc['id']}/search", json={**body, "include_images": False})
    assert all(m["source"] == "text" for m in r.json()["matches"])
