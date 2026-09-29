"""OSにインストールされたフォントの索引。

PDFに埋め込まれていない (またはサブセットに無い) 文字の見本を描画するために使う。
環境変数 FONT_DETECTOR_FONT_DIRS (os.pathsep 区切り) で検索先を追加できる。
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import fontname

FONT_SUFFIXES = {".ttf", ".otf", ".ttc", ".otc"}
_NAME_IDS = (1, 2, 4, 6, 16, 17)


@dataclass(frozen=True)
class SystemFont:
    path: str
    index: int  # TTC 内のフォント番号


def font_dirs() -> list[Path]:
    dirs: list[Path] = []
    extra = os.environ.get("FONT_DETECTOR_FONT_DIRS")
    if extra:
        dirs += [Path(p) for p in extra.split(os.pathsep) if p]
    home = Path.home()
    if sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        dirs.append(windir / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [Path("/System/Library/Fonts"), Path("/Library/Fonts"), home / "Library" / "Fonts"]
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                 home / ".local" / "share" / "fonts", home / ".fonts"]
    return [d for d in dirs if d.is_dir()]


def _names(font) -> set[str]:
    """name テーブルから照合に使う名前を集める (全言語)。"""
    table = font["name"]
    by_id: dict[int, set[str]] = {}
    for rec in table.names:
        if rec.nameID not in _NAME_IDS:
            continue
        try:
            by_id.setdefault(rec.nameID, set()).add(rec.toUnicode().strip())
        except UnicodeDecodeError:
            continue
    names = by_id.get(4, set()) | by_id.get(6, set())
    for fam_id, sub_id in ((1, 2), (16, 17)):
        for fam in by_id.get(fam_id, ()):
            for sub in by_id.get(sub_id, {""}):
                names.add(f"{fam} {sub}".strip())
    return {n for n in names if n}


@lru_cache(maxsize=128)
def cmap(font: SystemFont) -> frozenset[int]:
    """フォントが持つ文字 (Unicode コードポイント) の集合。"""
    from fontTools.ttLib import TTFont

    try:
        return frozenset((TTFont(font.path, fontNumber=font.index, lazy=True).getBestCmap() or {}).keys())
    except Exception:
        return frozenset()


class SystemFontIndex:
    """フォントキー (fontname.parse().key) → システムフォントファイル。初回参照時に走査する。"""

    def __init__(self, dirs: list[Path] | None = None) -> None:
        self._dirs = dirs
        self._index: dict[str, SystemFont] | None = None
        self._families: dict[SystemFont, set[str]] = {}
        self._lock = threading.Lock()

    def lookup(self, key: str) -> SystemFont | None:
        return self._load().get(key)

    def fonts(self) -> dict[SystemFont, set[str]]:
        """インストールされている全フォント → そのフォントの全ファミリーID (名前の言語違いを含む)。"""
        self._load()
        return dict(self._families)

    def _load(self) -> dict[str, SystemFont]:
        with self._lock:
            if self._index is None:
                self._index = self._scan(self._dirs if self._dirs is not None else font_dirs(), self._families)
            return self._index

    @staticmethod
    def _scan(dirs: list[Path], families: dict[SystemFont, set[str]]) -> dict[str, SystemFont]:
        from fontTools.ttLib import TTCollection, TTFont

        index: dict[str, SystemFont] = {}
        for d in dirs:
            for path in sorted(d.rglob("*")):
                if path.suffix.lower() not in FONT_SUFFIXES:
                    continue
                try:
                    if path.suffix.lower() in (".ttc", ".otc"):
                        fonts = TTCollection(str(path), lazy=True).fonts
                    else:
                        fonts = [TTFont(str(path), lazy=True)]
                    for i, font in enumerate(fonts):
                        sf = SystemFont(str(path), i)
                        for name in sorted(_names(font)):
                            parsed = fontname.parse(name)
                            index.setdefault(parsed.key, sf)
                            families.setdefault(sf, set()).add(parsed.family_id)
                except Exception:
                    continue  # 壊れたフォントや非対応形式は無視
        return index


default_index = SystemFontIndex()
