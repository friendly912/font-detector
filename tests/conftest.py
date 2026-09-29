import pymupdf
import pytest


def build_sample_pdf() -> bytes:
    """検証用PDF: 本文 (Times) / 見出し (Helvetica) / 日本語 / 透明テキスト / 画像 / 回転ページ。"""
    doc = pymupdf.open()

    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 30), "Header text", fontname="helv", fontsize=9)
    page.insert_text((72, 100), "Chapter Title", fontname="helv", fontsize=20)
    for i in range(10):
        page.insert_text((72, 150 + i * 16), f"Body line {i} in Times.", fontname="tiro", fontsize=11)
    page.insert_text((72, 340), "日本語の本文です。", fontname="japan", fontsize=11)
    page.insert_text((72, 400), "invisible ocr", fontname="cour", fontsize=11, render_mode=3)

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20), False)
    pix.set_rect(pix.irect, (200, 50, 50))
    page.insert_image(pymupdf.Rect(300, 450, 500, 550), pixmap=pix)

    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((72, 150), "Rotated body in Times.", fontname="tiro", fontsize=11)
    page2.set_rotation(90)

    return doc.tobytes()


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return build_sample_pdf()


NOTO = "/usr/share/fonts/opentype/noto"
SERIF = f"{NOTO}/NotoSerifCJK-Regular.ttc"
SANS = f"{NOTO}/NotoSansCJK-Regular.ttc"
DROID = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"

BODY = (
    "本文の文章は画像の中にある文字と比較するための見本になります。"
    "私たちは新しい方法で書類を作成し、会議の資料を確認しました。"
    "この報告書では調査の結果と今後の計画について説明します。"
)
IMAGE_LINES = [
    (SERIF, "画像の中の文字は明朝体で書かれています"),
    (SANS, "この行はゴシック体の文章です確認しました"),
    (DROID, "別の書体で作成した資料の説明になります"),
]


def _render_lines(lines, px=40) -> bytes:
    import io

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (px * 24, px * 2 * len(lines) + px), "white")
    d = ImageDraw.Draw(img)
    for i, (path, text) in enumerate(lines):
        d.text((px // 2, px // 2 + i * px * 2), text, font=ImageFont.truetype(path, px), fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def build_image_pdf() -> bytes:
    """本文 (Noto Serif / Noto Sans を埋め込み) と、3書体の文字を含む画像のPDF。"""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="serif", fontfile=SERIF)
    page.insert_font(fontname="sans", fontfile=SANS)
    for i in range(0, len(BODY), 30):
        page.insert_text((50, 80 + i), BODY[i : i + 30], fontname="serif", fontsize=10.5)
        page.insert_text((50, 300 + i), BODY[i : i + 30], fontname="sans", fontsize=10.5)
    page.insert_image(pymupdf.Rect(50, 500, 545, 640), stream=_render_lines(IMAGE_LINES))
    return doc.tobytes(garbage=3, deflate=True)


@pytest.fixture(scope="session")
def image_pdf() -> bytes:
    for path in (SERIF, SANS, DROID):
        if not __import__("os").path.exists(path):
            pytest.skip(f"テスト用フォントがありません: {path}")
    return build_image_pdf()
