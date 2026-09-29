"""Web UI 用 API サーバー。"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

import pymupdf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import annotate, matcher
from .analyzer import Analysis, analyze

if TYPE_CHECKING:
    from .imagescan import ImageScan

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DOCUMENTS = 20

# PyMuPDF はスレッドセーフではないため、全操作をこのロックで直列化する
_mupdf_lock = threading.Lock()


@dataclass
class StoredDocument:
    filename: str
    data: bytes
    doc: pymupdf.Document
    analysis: Analysis
    image_scan: ImageScan | None = None
    scan_lock: threading.Lock = field(default_factory=threading.Lock)


class DocumentStore:
    """アップロードされたPDFをメモリに保持する (古いものから破棄)。"""

    def __init__(self, capacity: int = MAX_DOCUMENTS) -> None:
        self._items: OrderedDict[str, StoredDocument] = OrderedDict()
        self._capacity = capacity

    def add(self, item: StoredDocument) -> str:
        doc_id = uuid.uuid4().hex
        self._items[doc_id] = item
        while len(self._items) > self._capacity:
            _, old = self._items.popitem(last=False)
            old.doc.close()
        return doc_id

    def get(self, doc_id: str) -> StoredDocument:
        item = self._items.get(doc_id)
        if item is None:
            raise HTTPException(404, "ドキュメントが見つかりません。再度アップロードしてください。")
        self._items.move_to_end(doc_id)
        return item


class SearchRequest(BaseModel):
    font_keys: list[str] = []
    query: str = ""
    body_only: bool = False
    include_images: bool = False
    min_score: float | None = None
    method: str = matcher.DEFAULT_METHOD

    def criteria(self) -> matcher.Criteria:
        return matcher.Criteria(
            frozenset(self.font_keys), self.query, self.body_only, self.min_score, self.method
        )


def _image_matches(item: StoredDocument, req: SearchRequest):
    if not req.include_images:
        return []
    return matcher.search_images(item.image_scan, item.analysis, req.criteria())


def create_app() -> FastAPI:
    # 自動生成ドキュメント (/docs) はCDNからスクリプトを読み込むため無効にし、完全にローカルで動かす
    app = FastAPI(title="Font Detector", docs_url=None, redoc_url=None, openapi_url=None)
    store = DocumentStore()

    @app.post("/api/documents")
    def upload(file: UploadFile = File(...)) -> dict:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "ファイルサイズが上限 (100MB) を超えています。")
        with _mupdf_lock:
            try:
                doc = pymupdf.open(stream=data, filetype="pdf")
            except Exception:
                raise HTTPException(400, "PDFとして読み込めませんでした。")
            if doc.needs_pass:
                doc.close()
                raise HTTPException(400, "パスワード保護されたPDFには対応していません。")
            analysis = analyze(doc)
        doc_id = store.add(StoredDocument(file.filename or "document.pdf", data, doc, analysis))
        return {"id": doc_id, "filename": file.filename, **analysis.to_json()}

    @app.get("/api/documents/{doc_id}/pages/{page_no}.png")
    def page_image(doc_id: str, page_no: int, scale: float = Query(1.5, ge=0.25, le=4.0)) -> Response:
        item = store.get(doc_id)
        if not 1 <= page_no <= len(item.analysis.page_sizes):
            raise HTTPException(404, "ページが存在しません。")
        with _mupdf_lock:
            pix = item.doc[page_no - 1].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            png = pix.tobytes("png")
        return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/documents/{doc_id}/search")
    def search(doc_id: str, req: SearchRequest) -> dict:
        item = store.get(doc_id)
        matches = matcher.search(item.analysis, req.criteria())
        return {
            "matches": [
                {
                    "page": s.page + 1,
                    "bbox": s.view_bbox,
                    "text": s.text,
                    "font": s.font_key,
                    "size": s.size,
                    "body": matcher.is_body(s, item.analysis),
                    "source": "text",
                }
                for s in matches
            ]
            + [
                {
                    "page": m.line.page + 1,
                    "bbox": m.line.view_bbox,
                    "text": m.line.text,
                    "font": m.score.key,
                    "score": round(m.score.score, 4),
                    "margin": round(m.line.margin(m.method), 4),
                    "method": m.method,
                    "source": "image",
                }
                for m in _image_matches(item, req)
            ],
            "font_keys": sorted(matcher.target_keys(item.analysis, req.criteria())),
        }

    @app.post("/api/documents/{doc_id}/annotated")
    def annotated(doc_id: str, req: SearchRequest) -> Response:
        item = store.get(doc_id)
        matches = matcher.search(item.analysis, req.criteria())
        with _mupdf_lock:
            pdf = annotate.annotate(item.data, item.analysis, matches, _image_matches(item, req))
        name = Path(item.filename).stem + "_highlighted.pdf"
        return Response(
            pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"},
        )

    @app.post("/api/documents/{doc_id}/image-scan")
    def image_scan(doc_id: str) -> dict:
        """画像領域のOCRとフォント推定 (初回のみ実行し、結果はキャッシュする)。"""
        item = store.get(doc_id)
        with item.scan_lock:
            if item.image_scan is None:
                try:
                    from . import imagescan
                    from .ocr import OcrUnavailableError
                except ImportError as e:
                    raise HTTPException(501, f"画像判定用のライブラリがありません: {e.name}")
                try:
                    item.image_scan = imagescan.scan(item.doc, item.analysis, mupdf_lock=_mupdf_lock)
                except OcrUnavailableError as e:
                    raise HTTPException(503, str(e))
        return item.image_scan.to_json()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
