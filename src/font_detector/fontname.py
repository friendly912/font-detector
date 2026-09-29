"""PDF内のフォント名の正規化。

PDFに記録されるフォント名は同じ書体でも表記が揺れる。
  - サブセット接頭辞:   "ABCDEF+MS-Mincho"
  - スタイル接尾辞:     "MS-Mincho,Bold" / "YuGothic-Regular"
  - 全角・空白の揺れ:   "ＭＳ 明朝" / "MS明朝" / "MS-Mincho"
  - 文字化け:           Shift_JIS のバイト列が Latin-1 として解釈された "‚l‚r –¾’©"
これらを吸収し、照合用のキーと表示名を作る。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_HEX_ESCAPE = re.compile(r"#([0-9A-Fa-f]{2})")
_STYLE_SUFFIX = re.compile(
    r"[-_ ]?(regular|roman|book|normal|medium|light|semibold|demibold|bold|"
    r"extrabold|heavy|black|italic|oblique|w\d{1,2})$",
    re.IGNORECASE,
)

# 別名表: compact() 後の表記 → 正規ファミリーID
_ALIASES: dict[str, str] = {}
# 正規ファミリーID → 表示名
_DISPLAY: dict[str, str] = {}


def _register(family_id: str, display: str, *aliases: str) -> None:
    _DISPLAY[family_id] = display
    for name in (family_id, display, *aliases):
        _ALIASES[compact(name)] = family_id


def compact(name: str) -> str:
    """照合用に NFKC 正規化・小文字化し、空白と区切り記号を除く。"""
    s = unicodedata.normalize("NFKC", name).lower()
    return re.sub(r"[\s\-_・]", "", s)


_register("msmincho", "MS 明朝", "ＭＳ 明朝")
_register("mspmincho", "MS P明朝", "ＭＳ Ｐ明朝")
_register("msgothic", "MS ゴシック", "ＭＳ ゴシック")
_register("mspgothic", "MS Pゴシック", "ＭＳ Ｐゴシック")
_register("msuigothic", "MS UI Gothic")
_register("yumincho", "游明朝", "Yu Mincho", "YuMincho")
_register("yugothic", "游ゴシック", "Yu Gothic", "YuGothic")
_register("yugothicui", "Yu Gothic UI")
_register("meiryo", "メイリオ", "Meiryo")
_register("meiryoui", "Meiryo UI")
_register("bizudmincho", "BIZ UD明朝", "BIZ UDMincho")
_register("bizudpmincho", "BIZ UDP明朝", "BIZ UDPMincho")
_register("bizudgothic", "BIZ UDゴシック", "BIZ UDGothic")
_register("bizudpgothic", "BIZ UDPゴシック", "BIZ UDPGothic")
_register("hiraminpron", "ヒラギノ明朝 ProN", "HiraMinProN", "Hiragino Mincho ProN")
_register("hirakakupron", "ヒラギノ角ゴ ProN", "HiraKakuProN", "Hiragino Kaku Gothic ProN")
_register("timesnewroman", "Times New Roman", "TimesNewRomanPSMT", "TimesNewRomanPS")
_register("arial", "Arial", "ArialMT")
_register("century", "Century")


@dataclass(frozen=True)
class FontName:
    raw: str         # PDFに記録された名前
    name: str        # 接頭辞除去・文字化け修復・NFKC 後の名前
    key: str         # フォント一覧の単位 (スタイル込み)
    family_id: str   # ファミリー単位の照合キー
    display: str     # UI 表示名
    style: str       # "Bold" などのスタイル (無ければ "")


def decode_hex_escapes(name: str) -> str:
    """PDF名前オブジェクトの #xx エスケープをデコードする。"""
    if not _HEX_ESCAPE.search(name):
        return name
    data = _HEX_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), name).encode("latin-1", "replace")
    for enc in ("utf-8", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


def repair_mojibake(name: str) -> str:
    """Shift_JIS 等のバイト列が 1 バイト文字として解釈された名前を復元する。"""
    if all(ord(c) < 0x80 for c in name):
        return name
    for single in ("cp1252", "latin-1"):
        try:
            data = name.encode(single)
        except UnicodeEncodeError:
            continue
        for enc in ("utf-8", "cp932"):
            try:
                fixed = data.decode(enc)
            except UnicodeDecodeError:
                continue
            if any(_is_cjk(c) for c in fixed):
                return fixed
    return name


def _is_cjk(c: str) -> bool:
    cp = ord(c)
    return 0x3000 <= cp <= 0x9FFF or 0xFF00 <= cp <= 0xFFEF


def parse(raw: str) -> FontName:
    name = _SUBSET_PREFIX.sub("", raw.strip())
    name = repair_mojibake(decode_hex_escapes(name))
    name = unicodedata.normalize("NFKC", name).strip()

    # "MS-Mincho,Bold" 形式のスタイル指定
    base, _, style = name.partition(",")
    base, style = base.strip(), style.strip()

    family = base
    suffixes: list[str] = []
    while True:
        m = _STYLE_SUFFIX.search(family)
        if not m or m.start() == 0:
            break
        suffixes.insert(0, m.group(1).capitalize())
        family = family[: m.start()]
    if not style:
        style = " ".join(suffixes)

    family_id = _ALIASES.get(compact(family), compact(family))
    display = _DISPLAY.get(family_id, family)
    if style and style.lower() not in ("regular", "roman", "book", "normal"):
        display = f"{display} {style}"
        key = f"{family_id}|{compact(style)}"
    else:
        key = family_id
    return FontName(raw=raw, name=name, key=key, family_id=family_id, display=display, style=style)


def query_matches(query: str, font: FontName) -> bool:
    """自由入力のフォント指定がこのフォントに該当するか。

    別名表で解決できればファミリー一致、できなければ部分一致で判定する。
    """
    q = compact(query)
    if not q:
        return False
    alias = _ALIASES.get(q)
    if alias is not None:
        return alias == font.family_id
    return any(q in compact(s) for s in (font.name, font.display, font.family_id))
