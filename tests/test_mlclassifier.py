import random
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from conftest import DROID, SANS, SERIF
from font_detector import glyphs, matcher, mlclassifier
from font_detector.analyzer import analyze
from font_detector.ocr import OcrUnavailableError, resolve_tesseract
from font_detector.sysfonts import SystemFontIndex

SERIF_KEY, SANS_KEY = "notoserifcjkjp", "notosanscjkjp"


def _glyph(ch, font, px=96):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (px * 2, px * 2), 255)
    ImageDraw.Draw(img).text((px // 2, px // 2), ch, font=ImageFont.truetype(font, px), fill=0)
    return glyphs.ink_crop(np.asarray(img))


def test_hog_and_features_shape():
    ink = _glyph("永", SERIF)
    f = mlclassifier.features(ink)
    assert f.shape == mlclassifier.features(_glyph("a", SANS)).shape
    assert np.isfinite(f).all()
    assert mlclassifier.hog(np.zeros((32, 32), np.float32)).shape == (324,)


def test_degrade_produces_small_observed_glyph():
    rng = random.Random(0)
    ink = _glyph("書", SERIF)
    sizes = [max(mlclassifier.degrade(ink, rng).shape) for _ in range(20)]
    assert min(sizes) >= mlclassifier.MIN_OBSERVED_PX - 4
    assert max(sizes) <= mlclassifier.MAX_OBSERVED_PX + 4


def test_mincho_contrast_feature():
    # 明朝体は縦画が横画より太い (特徴量の縦横比が大きい)
    def contrast(font):
        return np.mean([mlclassifier.features(_glyph(c, font))[-4] for c in "書十日本"])

    assert contrast(SERIF) > contrast(SANS) + 0.1


@pytest.mark.parametrize("a, b", [
    ("msmincho", "mspmincho"),
    ("msgothic", "mspgothic"),
    ("bizudgothic", "bizudpgothic"),
    ("yugothic", "yugothicui"),
    ("meiryo", "meiryoui"),
    ("notoserifcjkjp", "notoserifcjksc"),
    ("notosanscjkjp", "notosansmonocjkjp"),
])
def test_family_root_merges_glyph_sharing_variants(a, b):
    assert mlclassifier.family_root(a) == mlclassifier.family_root(b)


def test_family_root_keeps_different_designs_apart():
    roots = {mlclassifier.family_root(f) for f in ("msmincho", "msgothic", "yumincho", "yugothic", "meiryo")}
    assert len(roots) == 5


def test_other_fonts_exclude_variants_of_pdf_fonts(image_pdf):
    doc = pymupdf.open(stream=image_pdf, filetype="pdf")
    a = analyze(doc)
    index = SystemFontIndex([Path(SERIF).parent, Path(DROID).parent])
    others = mlclassifier._other_fonts(a, index, list("本文の文章は画像"), random.Random(0))
    paths = {sf.path for sf, _ in others}
    # Noto Serif/Sans CJK の他地域版 (同じ TTC) は「その他」に入らない
    assert paths == {DROID}


def _classifier(image_pdf, dirs):
    doc = pymupdf.open(stream=image_pdf, filetype="pdf")
    a = analyze(doc)
    return a, mlclassifier.train(doc, a, SystemFontIndex(dirs))


@pytest.fixture(scope="module")
def trained(image_pdf):
    # 「その他」の学習用に Droid Sans Fallback だけを置いたフォント環境
    return _classifier(image_pdf, [Path(DROID).parent])


def test_training_uses_pdf_glyphs_and_other_fonts(trained):
    _, clf = trained
    assert set(clf.classes) == {SERIF_KEY, SANS_KEY, mlclassifier.OTHER}
    assert clf.summary.sources[SERIF_KEY] == {"pdf"}
    assert clf.summary.other_fonts == 1


def test_classifies_characters_not_in_the_pdf(trained):
    """学習に使っていない文字 (PDFの本文に無い字) でも書体を判定できる。"""
    a, clf = trained
    body = {c for s in a.spans for c in s.text}
    unseen = [c for c in "旅行計画家族夏休図書館地図探桜並木撮影" if c not in body]
    assert len(unseen) >= 10
    for font, key in ((SERIF, SERIF_KEY), (SANS, SANS_KEY)):
        rng = random.Random(1)
        chars = [(c, mlclassifier.degrade(_glyph(c, font), rng)) for c in unseen]
        assert clf.rank_line(chars)[0][0] == key


def test_no_classifier_without_two_classes(image_pdf):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="serif", fontfile=SERIF)
    page.insert_text((50, 80), "本文" * 30, fontname="serif")
    a = analyze(pymupdf.open(stream=doc.tobytes()))
    assert mlclassifier.train(pymupdf.open(stream=doc.tobytes()), a, SystemFontIndex([])) is None


def test_scan_with_ml(image_pdf):
    from font_detector import imagescan

    try:
        resolve_tesseract()
    except OcrUnavailableError as e:
        pytest.skip(str(e))
    doc = pymupdf.open(stream=image_pdf, filetype="pdf")
    a = analyze(doc)
    scan = imagescan.scan(doc, a, font_index=SystemFontIndex([Path(DROID).parent]))
    assert scan.methods == ["ml", "template"]
    best = {ln.text[:4]: ln.best("ml").key for ln in scan.lines}
    assert best == {"画像の中": SERIF_KEY, "この行は": SANS_KEY, "別の書体": mlclassifier.OTHER}
    hits = matcher.search_images(scan, a, matcher.Criteria(font_keys=frozenset({SERIF_KEY, SANS_KEY})))
    assert sorted(m.line.text[:4] for m in hits) == ["この行は", "画像の中"]
