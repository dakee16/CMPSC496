"""test_editor_colors.py - every kind of Python token in the code editor is
readable, in the light theme and the dark one.

THE REPORT (30 Sep). A student's screenshot showed `self` almost invisible and
`.expressions.split` in pale cyan. The editor uses CodeMirror's DARK "dracula"
theme recoloured for our cream background (frontend/ui.css), but only some
token kinds were recoloured - `self` (cm-variable-2) kept dracula's white:
1.07:1 on cream, where text needs 4.5:1.

No browser: the colours are read from the CSS, and each must reach 4.5:1
contrast (WCAG AA for text) on the code background in both themes.
"""
import pathlib
import re

HERE = pathlib.Path(__file__).parent
# What CodeMirror 5's python mode marks up: keywords, builtins, def names,
# variables, `self`/`cls` (variable-2), strings, numbers, comments, operators,
# attributes after a dot (property), decorators (meta), True/False/None (atom).
PYTHON_TOKENS = ["keyword", "builtin", "def", "variable", "variable-2", "string",
                 "number", "comment", "operator", "property", "meta", "atom"]


def _palettes():
    found = {}
    for name, hexa in re.findall(r"--code-([\w-]+):\s*(#[0-9a-fA-F]{6})",
                                 (HERE / "frontend/tokens.css").read_text()):
        found.setdefault(name, []).append(hexa)
    return found                          # name -> [light, dark]


def _token_colors():
    out = {}
    for selectors, var in re.findall(r"([^{}]+)\{color:var\(--code-([\w-]+)\)",
                                     (HERE / "frontend/ui.css").read_text()):
        for cls in re.findall(r"\.cm-s-dracula span\.cm-([\w-]+)", selectors):
            out[cls] = var
    return out


def _contrast(a, b):
    def lum(h):
        rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_every_python_token_has_a_colour_of_ours():
    missing = [t for t in PYTHON_TOKENS if t not in _token_colors()]
    assert not missing, f"left on dracula's dark-theme colours: {missing}"


def test_every_token_colour_is_readable_in_both_themes():
    pal, colors = _palettes(), _token_colors()
    bg = pal["bg"]
    for token in PYTHON_TOKENS:
        var = colors.get(token)
        if var is None:
            continue                      # the test above reports it
        for theme, fg, back in zip(("light", "dark"), pal[var], bg):
            ratio = _contrast(fg, back)
            assert ratio >= 4.5, f"cm-{token} ({var}) is {ratio:.1f}:1 in the {theme} theme"
