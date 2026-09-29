"""Convert the 2024 edition's HTML sources into one Markdown file per chapter.

The book text is CC BY-NC-SA 4.0 and stays outside this repository. Images and example
screenshots aren't copied: the Markdown links to them in the source ``content/`` directory
with relative paths, so keep both directories where they are.

Usage:
    uv run tools/html_to_md.py \\
        --content ../nature-of-code-private/2024_p5js/noc-book-2-main/content \\
        --out ../nature-of-code-private/book-md
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["beautifulsoup4>=4.12"]
# ///

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

# Code annotations for the print layout, e.g. "//{!1 .bold} Comment" -> "// Comment".
ANNOTATION = re.compile(r"//\{[^}]*\}\s?")

# Cross-references use the website's slugs ("/vectors#anchor"); map them to our files.
SlugMap = dict[str, str]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class Converter:
    """Turns one chapter's HTML into Markdown."""

    def __init__(self, content: Path, assets: str, slugs: SlugMap) -> None:
        self.content = content
        self.assets = assets  # relative path from the output dir to the source content dir
        self.slugs = slugs

    # --- inline -----------------------------------------------------------
    def inline(self, node: Tag | NavigableString) -> str:
        if isinstance(node, NavigableString):
            return re.sub(r"\s+", " ", str(node))
        name = node.name
        inner = "".join(self.inline(c) for c in node.children)
        if name == "br":
            return " "
        if name == "img":
            return self.image(node)
        if node.get("data-type") == "equation":
            return f"${node.get_text().strip()}$"
        if name in ("i", "em"):
            return f"*{inner.strip()}*" if inner.strip() else inner
        if name in ("b", "strong"):
            return f"**{inner.strip()}**" if inner.strip() else inner
        if name == "s":
            return f"~~{inner.strip()}~~"
        if name == "code":
            return f"`{node.get_text()}`"
        if name == "a" and (href := node.get("href")):
            return f"[{inner.strip()}]({self.link(str(href))})"
        if name == "span" and "highlight" in (node.get("class") or []):
            return f"**{inner.strip()}**"
        return inner

    def text(self, node: Tag) -> str:
        return re.sub(r"[ \t]+", " ", "".join(self.inline(c) for c in node.children)).strip()

    def link(self, href: str) -> str:
        if href.startswith("/"):
            slug, _, anchor = href[1:].partition("#")
            target = self.slugs.get(slug, slug)
            return f"{target}#{anchor}" if anchor else target
        return href

    def asset(self, src: str) -> str:
        return src if src.startswith("http") else f"{self.assets}/{src}"

    def image(self, img: Tag) -> str:
        return f"![{img.get('alt', '')}]({self.asset(str(img.get('src', '')))})"

    # --- blocks -----------------------------------------------------------
    def blocks(self, node: Tag) -> list[str]:
        out: list[str] = []
        for child in node.children:
            if isinstance(child, NavigableString):
                if child.strip():
                    out.append(child.strip())
                continue
            out.extend(self.block(child))
        return out

    def block(self, node: Tag) -> list[str]:
        name = node.name
        kind = node.get("data-type")
        if m := re.fullmatch(r"h([1-6])", name or ""):
            return [f"{'#' * int(m[1])} {self.text(node)}"]
        if name == "p":
            t = self.text(node)
            return [t] if t else []
        if name == "pre":
            return [self.code(node)]
        if name in ("ul", "ol"):
            return [self.list_(node)]
        if name == "table":
            return [self.table(node)]
        if name == "figure":
            return [self.figure(node)]
        if kind == "equation":
            return [f"$$\n{node.get_text().strip()}\n$$"]
        if kind == "embed":
            return [self.embed(node)]
        if kind == "video-link":
            return [f"Video: [{node.get('data-title', '')}]({node.get('href', '')})"]
        if kind in ("example", "exercise", "note", "project"):
            return [quote("\n\n".join(self.blocks(node)))]
        if name == "blockquote":
            return [quote("\n\n".join(self.blocks(node)))]
        if name == "img":
            return [self.image(node)]
        if name in ("div", "section", "span", "a"):
            return self.blocks(node)
        t = self.text(node)
        return [t] if t else []

    def code(self, pre: Tag) -> str:
        lang = pre.get("data-code-language", "javascript")
        lines = []
        for line in pre.get_text().replace("\r", "").strip("\n").split("\n"):
            stripped = ANNOTATION.sub("// ", line).rstrip()
            if stripped.strip() == "//" and line.strip() != "//":
                continue  # the line only carried a layout annotation
            lines.append(stripped)
        return f"```{lang}\n" + "\n".join(lines) + "\n```"

    def embed(self, node: Tag) -> str:
        path = node.get("data-example-path", "")
        parts = [self.image(img) for img in node.find_all("img")]
        if path:
            sketch = f"{path}/sketch.js"
            target = sketch if (self.content / sketch).exists() else path
            parts.append(f"Source: [`{path}`]({self.asset(target)})")
        if editor := node.get("data-p5-editor"):
            parts.append(f"([p5.js editor]({editor}))")
        return "\n\n".join(parts)

    def figure(self, node: Tag) -> str:
        body = [b for child in node.children if isinstance(child, Tag) for b in self.block(child)]
        caption = node.find("figcaption")
        cap = self.text(caption) if caption else ""
        body = [b for b in body if b and b != cap]
        return "\n\n".join(body + ([f"*{cap}*"] if cap else []))

    def list_(self, node: Tag, depth: int = 0) -> str:
        lines = []
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            marker = f"{i}." if node.name == "ol" else "-"
            parts: list[str] = []
            nested: list[str] = []
            inline: list[str] = []
            for child in li.children:
                if isinstance(child, Tag) and child.name in ("ul", "ol"):
                    nested.append(self.list_(child, depth + 1))
                elif isinstance(child, Tag) and child.name in ("p", "pre", "figure", "div"):
                    parts.extend(self.block(child))
                else:
                    inline.append(self.inline(child))
            if text := re.sub(r"\s+", " ", "".join(inline)).strip():
                parts.insert(0, text)
            indent = "  " * depth
            body = "\n\n".join(parts).replace("\n", "\n" + indent + "  ")
            lines.append(f"{indent}{marker} {body}")
            lines.extend(nested)
        return "\n".join(lines)

    def table(self, node: Tag) -> str:
        rows = [
            [self.text(cell).replace("|", r"\|") for cell in tr.find_all(["th", "td"])]
            for tr in node.find_all("tr")
        ]
        rows = [r for r in rows if r]
        if not rows:
            return ""
        width = max(map(len, rows))
        rows = [r + [""] * (width - len(r)) for r in rows]
        md = [f"| {' | '.join(rows[0])} |", f"|{'---|' * width}"]
        md += [f"| {' | '.join(r)} |" for r in rows[1:]]
        return "\n".join(md)


def quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def convert(path: Path, conv: Converter) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    md = "\n\n".join(b for b in conv.blocks(soup) if b.strip())
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--content", type=Path, required=True, help="the book's content/ dir")
    parser.add_argument(
        "--out", type=Path, required=True, help="output directory (outside the repo!)"
    )
    args = parser.parse_args()

    content: Path = args.content.resolve()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads((content / "content.json").read_text(encoding="utf-8"))

    def filename(entry: dict[str, str]) -> str:
        if entry["type"] == "chapter":
            number, _, title = entry["title"].partition(". ")
            return f"{int(number):02d}-{slugify(title)}.md"
        return f"{Path(entry['src']).stem}.md"

    slugs = {e["slug"]: filename(e) for e in entries}
    conv = Converter(content, os.path.relpath(content, out), slugs)
    index = ["# The Nature of Code, 2024 edition (Markdown)", ""]
    for entry in entries:
        name = filename(entry)
        (out / name).write_text(convert(content / entry["src"], conv), encoding="utf-8")
        index.append(f"- [{entry['title'].strip()}]({name})")
        print(name)
    (out / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
