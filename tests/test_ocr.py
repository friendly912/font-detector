import pytest

from font_detector import ocr

HOCR = """<?xml version="1.0" encoding="UTF-8"?>
<html><body><div class='ocr_page'><div class='ocr_carea'><p class='ocr_par'>
 <span class='ocr_line' id='line_1_1' title="bbox 10 20 200 60; baseline 0 0">
  <span class='ocrx_word' title='bbox 10 20 40 60; x_wconf 96'>
   <span class='ocrx_cinfo' title='x_bboxes 10 20 40 60; x_conf 99.5'>日</span>
   <span class='ocrx_cinfo' title='x_bboxes 40 20 70 60; x_conf 42.0'>本</span>
  </span>
  <span class='ocrx_word' title='bbox 80 20 120 60; x_wconf 90'>
   <span class='ocrx_cinfo' title='x_bboxes 80 20 120 60; x_conf 90'>A</span>
   <span class='ocrx_cinfo' title='x_bboxes 80 20 120 60; x_conf 91'>&gt;</span>
  </span>
  <span class='ocrx_word' title='bbox 130 20 150 60; x_wconf 88'>
   <span class='ocrx_cinfo' title='x_bboxes 130 20 150 60; x_conf 88'>B</span>
  </span>
 </span>
 <span class='ocr_line' id='line_1_2' title="bbox 10 70 50 90">
  <span class='ocrx_word' title='bbox 10 70 50 90; x_wconf 80'>語</span>
 </span>
</p></div></div></body></html>"""


def test_parse_hocr_lines_chars_and_confidence():
    lines = ocr.parse_hocr(HOCR)
    assert len(lines) == 2
    box, chars, text = lines[0]
    assert box == (10, 20, 200, 60)
    assert text == "日本A>B"  # 日本語の後には空白を入れない
    assert chars[1] == ("本", 42.0)
    # 文字ごとの確信度が無い hOCR (古い Tesseract) は単語の確信度を使う
    assert lines[1][1] == [("語", 80.0)]


def test_parse_hocr_keeps_space_between_latin_words():
    hocr = HOCR.replace("&gt;", "C")
    assert ocr.parse_hocr(hocr)[0][2] == "日本AC B"


def test_missing_tesseract_is_reported(monkeypatch, tmp_path):
    ocr.resolve_tesseract.cache_clear()
    monkeypatch.setattr(ocr, "_candidate_executables", lambda: [str(tmp_path / "nope.exe")])
    try:
        with pytest.raises(ocr.OcrUnavailableError, match="Tesseract が見つかりません"):
            ocr.resolve_tesseract()
    finally:
        ocr.resolve_tesseract.cache_clear()


def test_missing_language_is_reported(monkeypatch):
    ocr.resolve_tesseract.cache_clear()
    monkeypatch.setenv("FONT_DETECTOR_OCR_LANG", "jpn+xyz")
    try:
        ocr.resolve_tesseract()
    except ocr.OcrUnavailableError as e:
        assert "xyz.traineddata" in str(e) or "Tesseract が見つかりません" in str(e)
    else:
        pytest.fail("存在しない言語で成功した")
    finally:
        ocr.resolve_tesseract.cache_clear()
