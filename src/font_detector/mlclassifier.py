"""文書ごとに学習するフォント分類器 (機械学習による画像内文字の判定)。

PDFのテキストレイヤーは「どのフォントでどの字形が描かれているか」の正解付きデータになる。
これを学習データにして、画像内の1文字ごとに、どのフォントで書かれたかを分類する。

  - 学習データ: テキストレイヤーから切り出した字形 (足りなければ同名のシステムフォント) を、
    縮小・ぼかし・JPEG圧縮・ノイズ・太さの変化で水増しし、画像内の文字の見え方に近づける。
  - 「その他」クラス: PDFで使われていないインストール済みフォントの字形。
    PDFに無い書体の文字を、PDF内のどれかのフォントと誤判定しないために使う。
  - 特徴量: 形の傾向 (HOG) と線の太さの分布・縦横の画の太さの比など、書体の特徴を表す量。
    文字そのものではなく書体の特徴を学習するので、PDFに出てこない文字も分類できる。
"""

from __future__ import annotations

import contextlib
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import ContextManager

import cv2
import numpy as np
import pymupdf

from . import glyphs
from .analyzer import Analysis
from .sysfonts import SystemFontIndex, cmap

OTHER = "__other__"             # PDFで使われていないフォントのクラス
FEATURE_SIZE = 32
STROKE_SIZE = 64
MAX_CHARS_PER_FONT = 300        # PDFのフォントごとに学習に使う文字数 (出現頻度の高い順)
AUGMENT_PER_GLYPH = 2
MAX_OTHER_FONTS = 12
OTHER_CHARS_PER_FONT = 120
MIN_TRAIN_GLYPHS = 20           # これより字形が少ないフォントは学習しない
MIN_OBSERVED_PX, MAX_OBSERVED_PX = 12, 48  # 水増しで想定する画像内の文字の大きさ
SEED = 0

HOG_CELL, HOG_BINS = 8, 9
_WIDTH_BINS = np.linspace(0, 0.3, 9)


def script_of(ch: str) -> str:
    return "cjk" if unicodedata.east_asian_width(ch) in ("W", "F") else "latin"


# ---------------------------------------------------------------- 特徴量


def _fit(ink: np.ndarray, size: int) -> np.ndarray:
    """縦横比を保って size 四方に収める。"""
    h, w = ink.shape
    s = (size - 2) / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    canvas = np.zeros((size, size), np.float32)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y : y + nh, x : x + nw] = cv2.resize(ink, (nw, nh), interpolation=interp)
    return canvas


def hog(img: np.ndarray) -> np.ndarray:
    """HOG特徴 (8px セル・9方向・2x2セルのブロック正規化)。

    OpenCV 5 では HOGDescriptor が本体から外れたため、NumPy で実装する。
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    bins = np.minimum((ang / (180.0 / HOG_BINS)).astype(np.int32), HOG_BINS - 1)
    n = img.shape[0] // HOG_CELL
    cells = np.zeros((n, n, HOG_BINS), np.float32)
    for b in range(HOG_BINS):
        m = np.where(bins == b, mag, 0.0)[: n * HOG_CELL, : n * HOG_CELL]
        cells[:, :, b] = m.reshape(n, HOG_CELL, n, HOG_CELL).sum(axis=(1, 3))
    blocks = []
    for y in range(n - 1):
        for x in range(n - 1):
            v = cells[y : y + 2, x : x + 2].ravel()
            blocks.append(v / (np.linalg.norm(v) + 1e-5))
    return np.concatenate(blocks)


def _run_lengths(b: np.ndarray, axis: int) -> np.ndarray:
    """2値画像のインクの連続長 (axis=0: 縦方向の連続 = 横画の太さ)。"""
    arr = b if axis == 0 else b.T
    padded = np.pad(arr.astype(np.int8), ((1, 1), (0, 0)))
    diff = np.diff(padded, axis=0)
    starts = np.nonzero(diff == 1)
    ends = np.nonzero(diff == -1)
    return (ends[0] - starts[0]).astype(np.float32)


def features(ink: np.ndarray) -> np.ndarray:
    """字形 (インク量 0..1 の外接矩形) の特徴ベクトル。"""
    h, w = ink.shape
    shape = _fit(ink, FEATURE_SIZE)
    shape_hog = hog(shape)

    big = _fit(ink, STROKE_SIZE)
    b = (big > 0.5).astype(np.uint8)
    density = float(b.mean())
    if b.any():
        dt = cv2.distanceTransform(b, cv2.DIST_L2, 3)
        ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (b > 0)
        widths = 2 * dt[ridge] / STROKE_SIZE
        pct = np.percentile(widths, [10, 25, 50, 75, 90]) if widths.size else np.zeros(5)
        hist = np.histogram(widths, bins=_WIDTH_BINS)[0] / max(1, widths.size)
        vruns, hruns = _run_lengths(b, 0), _run_lengths(b, 1)
        # 連続長には画の太さ (短い) と画に沿った長さ (長い) が混ざるので、短い側の代表値を使う
        v_med = float(np.percentile(vruns, 25)) if vruns.size else 0.0  # 横画の太さ
        h_med = float(np.percentile(hruns, 25)) if hruns.size else 0.0  # 縦画の太さ
        contrast = math.log((h_med + 1) / (v_med + 1))           # 明朝体は縦画が太く正になりやすい
    else:
        pct, hist, v_med, h_med, contrast = np.zeros(5), np.zeros(8), 0.0, 0.0, 0.0
    stroke = np.concatenate([
        pct, hist,
        [pct[4] / (pct[0] + 1e-3), density, math.log(w / h), v_med / STROKE_SIZE,
         h_med / STROKE_SIZE, contrast],
    ])
    # HOG (324次元) に埋もれないよう、書体を表す量を重み付けして結合する
    return np.concatenate([shape_hog, np.repeat(stroke.astype(np.float32), 4)]).astype(np.float32)


# ---------------------------------------------------------------- 水増し


def degrade(ink: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """高解像度の字形を、画像内の文字のように劣化させる (縮小・ぼかし・圧縮・ノイズ・太さ)。"""
    g = ink
    r = rng.random()
    if r < 0.15:
        g = cv2.erode(g, np.ones((2, 2), np.uint8))
    elif r < 0.3:
        g = cv2.dilate(g, np.ones((2, 2), np.uint8))
    target = math.exp(rng.uniform(math.log(MIN_OBSERVED_PX), math.log(MAX_OBSERVED_PX)))
    s = min(1.0, target / max(g.shape))
    small = cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    small = np.pad(small, 3)
    gray = 255.0 - small * 255.0 * rng.uniform(0.75, 1.0)
    sigma = rng.uniform(0.0, 0.7)
    if sigma > 0.05:
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    gray = gray + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 6), gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, enc = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(40, 95)])
        if ok:
            gray = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return glyphs.ink_crop(gray)


# ---------------------------------------------------------------- 分類器


@dataclass
class TrainingSummary:
    glyphs: dict[str, int] = field(default_factory=dict)       # クラス → 元の字形数
    sources: dict[str, set[str]] = field(default_factory=dict)  # クラス → "pdf" / "system"
    other_fonts: int = 0
    samples: int = 0

    def to_json(self) -> dict:
        return {
            "glyphs": self.glyphs,
            "sources": {k: sorted(v) for k, v in self.sources.items()},
            "other_fonts": self.other_fonts,
            "samples": self.samples,
        }


@dataclass
class FontClassifier:
    model: object                 # sklearn Pipeline
    classes: list[str]
    scripts: dict[str, set[str]]  # クラスが学習した文字種 (cjk / latin)
    summary: TrainingSummary

    def char_proba(self, chars: list[tuple[str, np.ndarray]]) -> np.ndarray:
        """1文字ごとのクラス確率。その文字種を学習していないクラスは除外して正規化する。"""
        x = np.stack([features(ink) for _, ink in chars])
        proba = self.model.predict_proba(x)
        for i, (c, _) in enumerate(chars):
            sc = script_of(c)
            mask = np.array([sc in self.scripts[k] for k in self.classes], dtype=np.float32)
            p = proba[i] * mask
            proba[i] = p / p.sum() if p.sum() > 0 else proba[i]
        return proba

    def rank_line(self, chars: list[tuple[str, np.ndarray]]) -> list[tuple[str, float, int]]:
        """行の各文字の確率を平均し、[(クラス, 確率, 文字数)] を確率の高い順に返す。"""
        if not chars:
            return []
        mean = self.char_proba(chars).mean(axis=0)
        order = np.argsort(-mean)
        return [(self.classes[i], float(mean[i]), len(chars)) for i in order]


def _pdf_chars(analysis: Analysis) -> dict[str, list[str]]:
    """PDFのフォントごとに、学習に使う文字 (出現頻度の高い順)。"""
    counts: dict[str, Counter] = {}
    for s in analysis.spans:
        counts.setdefault(s.font_key, Counter()).update(c for c in s.text if glyphs.informative(c))
    return {k: [c for c, _ in cnt.most_common(MAX_CHARS_PER_FONT)] for k, cnt in counts.items()}


_VARIANT_PATTERNS = [
    (re.compile(r"cjk(jp|sc|tc|kr|hk)$"), "cjk"),   # Noto CJK などの地域版
    (re.compile(r"p(mincho|gothic)"), r"\1"),       # MS P明朝 / BIZ UDPゴシック などのプロポーショナル版
    (re.compile(r"udp"), "ud"),
    (re.compile(r"(ui|mono)$"), ""),                 # Yu Gothic UI / Meiryo UI などのUI版
    (re.compile(r"^(notosans|notoserif)mono"), r"\1"),
]


def family_root(family_id: str) -> str:
    """字形が共通の派生版 (P版・UI版・地域版) を同一視するためのファミリー名。"""
    root = family_id
    for pattern, repl in _VARIANT_PATTERNS:
        root = pattern.sub(repl, root)
    return root


def _other_fonts(
    analysis: Analysis, index: SystemFontIndex, sample: list[str], rng: random.Random
) -> list[tuple[object, dict[str, np.ndarray]]]:
    """PDFで使われていないインストール済みフォントの字形。文字種ごとに偏らないよう集める。

    PDFのフォントと字形が共通のフォント (同じファイルに入った別名、P版・UI版・地域版) を
    「その他」として学習すると、同じ字形に別のラベルが付いて判定が乱れるため除外する。
    """
    pdf_roots = {family_root(f.family_id) for f in analysis.fonts.values()}
    pdf_files = {sf.path for key in analysis.fonts if (sf := index.lookup(key)) is not None}
    candidates = [
        sf for sf, fams in index.fonts().items()
        if sf.path not in pdf_files and not ({family_root(f) for f in fams} & pdf_roots)
    ]
    rng.shuffle(candidates)
    by_script: dict[str, list[str]] = {}
    for c in sample:
        by_script.setdefault(script_of(c), []).append(c)
    quota = {sc: max(1, MAX_OTHER_FONTS * len(cs) // len(sample)) for sc, cs in by_script.items()}
    taken: dict[str, int] = Counter()
    result = []
    for sf in candidates:
        if all(taken[sc] >= quota[sc] for sc in quota):
            break
        covered = cmap(sf)
        for sc, cs in by_script.items():
            if taken[sc] >= quota[sc]:
                continue
            hit = [c for c in cs if ord(c) in covered]
            if len(hit) >= 0.6 * len(cs):
                result.append((sf, glyphs.render_font_glyphs(sf, set(hit[:OTHER_CHARS_PER_FONT]))))
                taken[sc] += 1
                break
    return result


Labelled = list[tuple[str, str, np.ndarray]]  # (クラス, 文字, 高解像度の字形)


def collect_training_glyphs(
    doc: pymupdf.Document,
    analysis: Analysis,
    index: SystemFontIndex,
    mupdf_lock: ContextManager | None = None,
) -> tuple[Labelled, TrainingSummary] | None:
    """学習用の字形を集める。学習できるクラスが2つ未満なら None。"""
    rng = random.Random(SEED)
    lock = mupdf_lock or contextlib.nullcontext()
    wanted = _pdf_chars(analysis)
    with lock:
        refs = glyphs.collect_pdf_glyphs(doc, analysis, lambda k, c: c in wanted.get(k, ()))

    summary = TrainingSummary()
    labelled: Labelled = []
    for key, chars in wanted.items():
        found = dict(refs[key].glyphs) if key in refs else {}
        sources = set(refs[key].sources) if key in refs else set()
        # 埋め込まれていないフォントや、字形が少ないフォントは同名のシステムフォントで補う
        sf = index.lookup(key)
        if sf is not None and len(found) < len(chars):
            extra = glyphs.render_font_glyphs(sf, set(chars) - found.keys())
            if extra:
                found.update(extra)
                sources.add("system")
        if len(found) >= MIN_TRAIN_GLYPHS:
            labelled += [(key, c, ink) for c, ink in found.items()]
            summary.sources[key] = sources

    sample = sorted({c for _, c, _ in labelled})
    rng.shuffle(sample)
    other_list = _other_fonts(analysis, index, sample[: OTHER_CHARS_PER_FONT * 2], rng)
    other = [(OTHER, c, ink) for _, g in other_list for c, ink in g.items()]
    if len(other) >= MIN_TRAIN_GLYPHS:
        labelled += other
        summary.other_fonts = len(other_list)
        summary.sources[OTHER] = {"system"}

    if len({k for k, _, _ in labelled}) < 2:
        return None
    for key, _, _ in labelled:
        summary.glyphs[key] = summary.glyphs.get(key, 0) + 1
    return labelled, summary


def make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        # 強めの正則化 (C=0.01) は、ベンチマークで精度を保ったまま学習時間が最も短かった
        LogisticRegression(C=0.01, max_iter=3000, tol=1e-3, class_weight="balanced"),
    )


def fit(labelled: Labelled, summary: TrainingSummary, model=None) -> FontClassifier:
    """字形を水増しして特徴量にし、分類器を学習する。"""
    rng = random.Random(SEED)
    x, y = [], []
    scripts: dict[str, set[str]] = {OTHER: {"cjk", "latin"}}
    for key, c, ink in labelled:
        scripts.setdefault(key, set()).add(script_of(c))
        for _ in range(AUGMENT_PER_GLYPH):
            d = degrade(ink, rng)
            if d is not None and max(d.shape) >= glyphs.MIN_INK_PX:
                x.append(features(d))
                y.append(key)
    summary.samples = len(y)
    model = model if model is not None else make_model()
    model.fit(np.stack(x), np.array(y))
    labels = list(model.classes_)
    return FontClassifier(model, labels, {k: scripts.get(k, set()) for k in labels}, summary)


def train(
    doc: pymupdf.Document,
    analysis: Analysis,
    index: SystemFontIndex,
    mupdf_lock: ContextManager | None = None,
) -> FontClassifier | None:
    """文書のフォントを見分ける分類器を学習する。学習できるクラスが2つ未満なら None。"""
    collected = collect_training_glyphs(doc, analysis, index, mupdf_lock)
    if collected is None:
        return None
    return fit(*collected)
