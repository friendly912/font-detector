import pymupdf
from fastapi.testclient import TestClient

from font_detector import annotate, matcher
from font_detector.analyzer import analyze
from font_detector.server import create_app


def _analyze(data):
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return analyze(doc)


def test_font_inventory(sample_pdf):
    a = _analyze(sample_pdf)
    assert list(a.fonts)[0] == "times"  # 文字数が最多のフォントが先頭
    assert {"times", "helvetica"} <= set(a.fonts)
    assert "courier" not in a.fonts          # 透明テキストは除外
    assert a.invisible_chars == len("invisible ocr")
    assert a.body_size == 11.0
    assert len(a.images) == 1


def test_search_body_only_excludes_heading_and_header(sample_pdf):
    a = _analyze(sample_pdf)
    all_helv = matcher.search(a, matcher.Criteria(query="Helvetica"))
    assert len(all_helv) == 2
    assert matcher.search(a, matcher.Criteria(query="Helvetica", body_only=True)) == []

    times = matcher.search(a, matcher.Criteria(font_keys=frozenset({"times"}), body_only=True))
    assert len(times) == 11
    assert {s.page for s in times} == {0, 1}


def test_rotated_page_view_bbox_matches_render(sample_pdf):
    a = _analyze(sample_pdf)
    span = next(s for s in a.spans if s.page == 1)
    w, h = a.page_sizes[1]
    assert w > h  # 90度回転で横長
    x0, y0, x1, y1 = span.view_bbox
    assert y1 - y0 > x1 - x0  # 横書きの行が縦長の箱になる

    with pymupdf.open(stream=sample_pdf, filetype="pdf") as doc:
        pix = doc[1].get_pixmap()
    # 表示座標の箱の内側にインク (暗い画素) がある
    dark = [
        (x, y)
        for x in range(int(x0), int(x1))
        for y in range(int(y0), int(y1))
        if sum(pix.pixel(x, y)) < 300
    ]
    assert dark


def test_annotate_adds_highlights(sample_pdf):
    a = _analyze(sample_pdf)
    matches = matcher.search(a, matcher.Criteria(query="times"))
    out = annotate.annotate(sample_pdf, a, matches)
    with pymupdf.open(stream=out, filetype="pdf") as doc:
        types = [an.type[1] for page in doc for an in page.annots()]
        assert types == ["Highlight"] * len(matches)
        # 回転ページの注釈も対象テキストを覆っている
        page = doc[1]
        an = next(page.annots())
        assert "Rotated" in page.get_textbox(an.rect)


def test_api_flow(sample_pdf):
    client = TestClient(create_app())
    r = client.post("/api/documents", files={"file": ("sample.pdf", sample_pdf, "application/pdf")})
    assert r.status_code == 200
    doc = r.json()
    assert doc["page_count"] == 2

    r = client.post(f"/api/documents/{doc['id']}/search", json={"query": "times", "body_only": True})
    assert r.status_code == 200
    assert len(r.json()["matches"]) == 11

    r = client.get(f"/api/documents/{doc['id']}/pages/1.png?scale=1")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"

    r = client.post(f"/api/documents/{doc['id']}/annotated", json={"font_keys": ["times"]})
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"

    assert client.post("/api/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}).status_code == 400
    assert client.get("/").status_code == 200
