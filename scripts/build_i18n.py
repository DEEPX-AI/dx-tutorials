#!/usr/bin/env python3
"""Generate translated copies of a tutorial notebook from sidecar translation files.

Layout
------
notebooks/T01-Getting-Started/getting_started.ipynb      English source (single source of truth)
i18n/ko/T01-Getting-Started/getting_started.md           Korean translations of the Markdown cells
notebooks/T01-Getting-Started/getting_started.ko.ipynb   generated: same code cells, Korean Markdown

A translation file is plain Markdown. Each Markdown cell of the source notebook is one block:

    <!-- cell: <cell id> src: <hash of the English cell when it was translated> -->
    translated Markdown ...

Code cells are copied verbatim, so one verification run of the English notebook covers
every language. Cells without a translation keep the English text; cells whose English
source changed since the translation (hash mismatch) are reported as stale.

Usage
-----
    python scripts/build_i18n.py init  notebooks/T01-Getting-Started/getting_started.ipynb --lang ko
    python scripts/build_i18n.py build notebooks/T01-Getting-Started/getting_started.ipynb
    python scripts/build_i18n.py stamp notebooks/T01-Getting-Started/getting_started.ipynb --lang ko
    python scripts/build_i18n.py check [notebook ...]

`build` also makes sure the English source has the language-switcher cell at the top.
`stamp` records the current English hash for every translated cell (run it after updating a
translation). `check` exits non-zero when a generated notebook is out of date or a
translation is missing or stale; use it in CI or a pre-commit hook.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
I18N_DIR = ROOT / "i18n"

SOURCE_LANG = "en"
LANGUAGE_NAMES = {
    "en": "English",
    "ko": "한국어",
    "ja": "日本語",
    "zh": "中文",
}
SWITCHER_ID = "i18n-language-switcher"
CELL_MARKER = re.compile(r"^<!--\s*cell:\s*(?P<id>\S+)(?:\s+src:\s*(?P<src>[0-9a-f]+))?\s*-->\s*$")


def cell_hash(source: str) -> str:
    return hashlib.sha1(source.strip().encode("utf-8")).hexdigest()[:10]


def translation_path(notebook: Path, lang: str) -> Path:
    relative = notebook.resolve().relative_to(NOTEBOOKS_DIR.resolve())
    return I18N_DIR / lang / relative.with_suffix(".md")


def generated_path(notebook: Path, lang: str) -> Path:
    return notebook.with_name(f"{notebook.stem}.{lang}.ipynb")


def available_languages(notebook: Path) -> list[str]:
    langs = []
    if I18N_DIR.is_dir():
        for lang_dir in I18N_DIR.iterdir():
            if lang_dir.is_dir() and translation_path(notebook, lang_dir.name).is_file():
                langs.append(lang_dir.name)
    # Known languages in LANGUAGE_NAMES order (the switcher order), then any others alphabetically.
    order = {code: index for index, code in enumerate(LANGUAGE_NAMES)}
    return sorted(langs, key=lambda code: (order.get(code, len(order)), code))


def markdown_cells(nb) -> list:
    return [cell for cell in nb.cells if cell.cell_type == "markdown" and cell.get("id") != SWITCHER_ID]


def parse_translation(path: Path) -> dict[str, tuple[str | None, str]]:
    """Return {cell id: (recorded source hash, translated markdown)}."""
    blocks: dict[str, tuple[str | None, str]] = {}
    current_id = None
    current_hash = None
    buffer: list[str] = []

    def flush():
        if current_id is not None:
            blocks[current_id] = (current_hash, "\n".join(buffer).strip("\n"))

    for line in path.read_text(encoding="utf-8").splitlines():
        match = CELL_MARKER.match(line)
        if match:
            flush()
            current_id, current_hash, buffer = match.group("id"), match.group("src"), []
        elif current_id is not None:
            buffer.append(line)
    flush()
    return blocks


def write_translation(path: Path, notebook: Path, lang: str, blocks: dict[str, tuple[str | None, str]], order: list) -> None:
    lines = [
        f"<!-- i18n source: {notebook.resolve().relative_to(ROOT).as_posix()} -->",
        f"<!-- i18n lang: {lang} -->",
        "<!-- One block per Markdown cell of the source notebook. Keep the markers; translate the text. -->",
        "",
    ]
    for cell in order:
        src_hash, text = blocks[cell.id]
        marker = f"<!-- cell: {cell.id}" + (f" src: {src_hash}" if src_hash else "") + " -->"
        lines += [marker, text, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def switcher_source(notebook: Path, lang: str, langs: list[str]) -> str:
    parts = []
    for code in [SOURCE_LANG, *langs]:
        name = LANGUAGE_NAMES.get(code, code)
        target = notebook.name if code == SOURCE_LANG else generated_path(notebook, code).name
        parts.append(f"**{name}**" if code == lang else f"[{name}]({target})")
    return "🌐 " + " | ".join(parts)


def switcher_cell(notebook: Path, lang: str, langs: list[str]):
    cell = nbformat.v4.new_markdown_cell(switcher_source(notebook, lang, langs))
    cell.id = SWITCHER_ID
    cell.metadata["i18n"] = {"role": "switcher"}
    return cell


def ensure_switcher(nb, notebook: Path, lang: str, langs: list[str]) -> bool:
    """Insert or refresh the switcher cell at the top. Returns True when the notebook changed."""
    wanted = switcher_source(notebook, lang, langs)
    existing = [index for index, cell in enumerate(nb.cells) if cell.get("id") == SWITCHER_ID]
    if existing:
        index = existing[0]
        if nb.cells[index].source == wanted and index == 0:
            return False
        cell = nb.cells.pop(index)
        cell.source = wanted
        nb.cells.insert(0, cell)
        return True
    nb.cells.insert(0, switcher_cell(notebook, lang, langs))
    return True


def strip_outputs(nb) -> None:
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None


def render(nb_source, notebook: Path, lang: str, langs: list[str], blocks: dict[str, tuple[str | None, str]]):
    """Build the translated notebook object and return (notebook, missing ids, stale ids)."""
    nb = nbformat.from_dict(nb_source)
    missing, stale = [], []
    for cell in markdown_cells(nb):
        if cell.id not in blocks or not blocks[cell.id][1].strip():
            missing.append(cell.id)
            continue
        src_hash, text = blocks[cell.id]
        if src_hash != cell_hash(cell.source):
            stale.append(cell.id)
        cell.source = text
    ensure_switcher(nb, notebook, lang, langs)
    nb.metadata["i18n"] = {
        "lang": lang,
        "source": notebook.resolve().relative_to(ROOT).as_posix(),
        "generator": "scripts/build_i18n.py",
    }
    strip_outputs(nb)
    return nb, missing, stale


def notebook_arg(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        sys.exit(f"Notebook not found: {path}")
    if re.search(r"\.[a-z]{2}\.ipynb$", path.name):
        sys.exit(f"{path.name} looks like a generated notebook; pass the English source instead.")
    return path


def cmd_init(args) -> int:
    notebook = notebook_arg(args.notebook)
    nb = nbformat.read(notebook, as_version=4)
    path = translation_path(notebook, args.lang)
    if path.exists() and not args.force:
        sys.exit(f"{path.relative_to(ROOT)} exists; use --force to overwrite it with the English template.")
    cells = markdown_cells(nb)
    blocks = {cell.id: (cell_hash(cell.source), cell.source.strip("\n")) for cell in cells}
    write_translation(path, notebook, args.lang, blocks, cells)
    print(f"Template written: {path.relative_to(ROOT)} ({len(cells)} cells). Translate the text under each marker.")
    return 0


def cmd_stamp(args) -> int:
    notebook = notebook_arg(args.notebook)
    nb = nbformat.read(notebook, as_version=4)
    langs = [args.lang] if args.lang else available_languages(notebook)
    cells = markdown_cells(nb)
    by_id = {cell.id: cell for cell in cells}
    for lang in langs:
        path = translation_path(notebook, lang)
        blocks = parse_translation(path)
        unknown = sorted(set(blocks) - set(by_id))
        stamped = {cid: (cell_hash(by_id[cid].source), text) for cid, (_, text) in blocks.items() if cid in by_id}
        order = [cell for cell in cells if cell.id in stamped]
        write_translation(path, notebook, lang, stamped, order)
        print(f"[{lang}] stamped {len(stamped)} cells in {path.relative_to(ROOT)}")
        if unknown:
            print(f"[{lang}]   dropped {len(unknown)} block(s) whose cell id no longer exists: {', '.join(unknown)}")
    return 0


def build_one(notebook: Path, write: bool) -> int:
    """Build every language for one notebook. Returns the number of problems found."""
    langs = available_languages(notebook)
    if not langs:
        print(f"{notebook.relative_to(ROOT)}: no translation files under i18n/<lang>/, nothing to do")
        return 0

    nb = nbformat.read(notebook, as_version=4)
    problems = 0

    if ensure_switcher(nb, notebook, SOURCE_LANG, langs):
        if write:
            nbformat.write(nb, notebook)
            print(f"[en] updated language switcher in {notebook.relative_to(ROOT)}")
        else:
            print(f"[en] language switcher is missing or outdated in {notebook.relative_to(ROOT)}")
            problems += 1

    for lang in langs:
        blocks = parse_translation(translation_path(notebook, lang))
        generated, missing, stale = render(nb, notebook, lang, langs, blocks)
        target = generated_path(notebook, lang)
        status = f"{len(markdown_cells(generated)) - len(missing)}/{len(markdown_cells(generated))} cells translated"
        if missing:
            status += f", {len(missing)} missing (English kept)"
        if stale:
            status += f", {len(stale)} stale (source changed after translation)"
        if write:
            nbformat.write(generated, target)
            print(f"[{lang}] wrote {target.relative_to(ROOT)}: {status}")
        else:
            current = nbformat.read(target, as_version=4) if target.is_file() else None
            up_to_date = current is not None and nbformat.writes(current) == nbformat.writes(generated)
            print(f"[{lang}] {target.relative_to(ROOT)}: {'up to date' if up_to_date else 'OUT OF DATE'}; {status}")
            problems += 0 if up_to_date else 1
        problems += len(missing) + len(stale)
        for cid in stale:
            print(f"[{lang}]   stale: {cid}")
        for cid in missing:
            print(f"[{lang}]   missing: {cid}")
    return problems


def all_source_notebooks() -> list[Path]:
    sources = set()
    if I18N_DIR.is_dir():
        for md in I18N_DIR.glob("*/**/*.md"):
            lang = md.relative_to(I18N_DIR).parts[0]
            relative = md.relative_to(I18N_DIR / lang).with_suffix(".ipynb")
            source = NOTEBOOKS_DIR / relative
            if source.is_file():
                sources.add(source)
    return sorted(sources)


def cmd_build(args) -> int:
    notebooks = [notebook_arg(n) for n in args.notebooks] or all_source_notebooks()
    problems = sum(build_one(nb, write=True) for nb in notebooks)
    return 0 if problems == 0 or not args.strict else 1


def cmd_check(args) -> int:
    notebooks = [notebook_arg(n) for n in args.notebooks] or all_source_notebooks()
    problems = sum(build_one(nb, write=False) for nb in notebooks)
    print("OK: all generated notebooks are up to date" if problems == 0 else f"{problems} problem(s) found")
    return 0 if problems == 0 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write an English template translation file for a language")
    p.add_argument("notebook")
    p.add_argument("--lang", required=True, choices=sorted(set(LANGUAGE_NAMES) - {SOURCE_LANG}))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("stamp", help="record the current English hash for every translated cell")
    p.add_argument("notebook")
    p.add_argument("--lang")
    p.set_defaults(func=cmd_stamp)

    p = sub.add_parser("build", help="generate <stem>.<lang>.ipynb next to the source notebook")
    p.add_argument("notebooks", nargs="*")
    p.add_argument("--strict", action="store_true", help="exit 1 when any translation is missing or stale")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("check", help="verify generated notebooks and translations without writing")
    p.add_argument("notebooks", nargs="*")
    p.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
