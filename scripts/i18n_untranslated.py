#!/usr/bin/env python3
"""Report translation blocks that still look like the English source.

Usage: python scripts/i18n_untranslated.py [lang ...]

A block counts as untranslated when its prose (text outside code blocks, inline code,
URLs, and HTML tags) is identical to the English cell, or when it contains Latin letters
but almost no characters of the target script. Blocks whose English prose is empty
(image-only or code-only cells) are ignored.
"""
import re
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_i18n as b  # noqa: E402

SCRIPT_RE = {
    "ko": re.compile(r"[가-힣]"),
    "ja": re.compile(r"[぀-ヿ一-鿿]"),
    "zh": re.compile(r"[一-鿿]"),
}


def prose(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"~~~.*?~~~", " ", text, flags=re.S)
    text = re.sub(r"<pre.*?</pre>", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"[#|*_>\-\d.:()\[\]/,;=+%&]+", " ", text)
    return " ".join(text.split())


def main(langs):
    problems = 0
    for lang in langs:
        for md in sorted((b.I18N_DIR / lang).rglob("*.md")):
            rel = md.relative_to(b.I18N_DIR / lang).with_suffix(".ipynb")
            notebook = b.NOTEBOOKS_DIR / rel
            nb = nbformat.read(notebook, as_version=4)
            english = {c.id: c.source for c in b.markdown_cells(nb)}
            blocks = b.parse_translation(md)
            bad = []
            for cid, (_, text) in blocks.items():
                en = prose(english.get(cid, ""))
                if len(en) < 8:
                    continue
                tr = prose(text)
                latin = len(re.findall(r"[A-Za-z]", tr))
                script = len(SCRIPT_RE[lang].findall(text))
                if tr == en or (script < 3 and latin > 20):
                    bad.append(cid)
            status = "OK" if not bad else f"{len(bad)} untranslated"
            print(f"[{lang}] {md.relative_to(b.ROOT)}: {len(blocks)} blocks, {status}")
            for cid in bad:
                print(f"       {cid}")
            problems += len(bad)
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["ko", "ja", "zh"]))
