import pytest

from font_detector import fontname


@pytest.mark.parametrize(
    "raw, key, display",
    [
        ("ABCDEF+MS-Mincho", "msmincho", "MS 明朝"),
        ("MSMincho", "msmincho", "MS 明朝"),
        ("ＭＳ 明朝", "msmincho", "MS 明朝"),
        ("MS-Mincho,Bold", "msmincho|bold", "MS 明朝 Bold"),
        ("YuGothic-Regular", "yugothic", "游ゴシック"),
        ("YuGothic-Bold", "yugothic|bold", "游ゴシック Bold"),
        ("TimesNewRomanPSMT", "timesnewroman", "Times New Roman"),
        ("Arial-BoldItalicMT", "arialbolditalicmt", "Arial-BoldItalicMT"),
        ("Century", "century", "Century"),
    ],
)
def test_parse(raw, key, display):
    f = fontname.parse(raw)
    assert (f.key, f.display) == (key, display)


def test_hex_escaped_shift_jis():
    # "ＭＳ 明朝" を Shift_JIS で #xx エスケープした名前
    raw = "".join(f"#{b:02X}" for b in "ＭＳ 明朝".encode("cp932"))
    assert fontname.parse(raw).family_id == "msmincho"


def test_mojibake_shift_jis():
    raw = "ＭＳ ゴシック".encode("cp932").decode("cp1252", errors="ignore")
    assert fontname.parse(raw).family_id == "msgothic"


@pytest.mark.parametrize("query", ["MS 明朝", "ms mincho", "ＭＳ明朝", "mincho"])
def test_query_matches_mincho(query):
    assert fontname.query_matches(query, fontname.parse("ABCDEF+MS-Mincho"))


def test_query_alias_does_not_match_proportional_variant():
    assert not fontname.query_matches("MS 明朝", fontname.parse("MS-PMincho"))
