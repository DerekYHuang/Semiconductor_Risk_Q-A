"""
Parse raw documents (PDF, HTML, TXT) from data/raw/* into clean plain-text files.

Run:
    python src/ingest/parse_documents.py

Output:
    One .txt file per source document, written to data/processed/parsed/
    (created automatically), preserving source-folder info in the filename
    so you can trace any chunk back to its origin later.
"""

import re
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader
from tqdm import tqdm

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    """Collapse excess whitespace without destroying paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            # Scanned/garbled PDFs happen a lot with GeoTracker reports — log and move on
            # rather than crashing the whole ingest run.
            print(f"  [warn] could not extract a page from {path.name}: {e}")
    return clean_text("\n".join(pages))


def parse_html(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return clean_text(soup.get_text(separator="\n"))


def parse_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())


def main():
    config = load_config()
    raw_root = Path(__file__).resolve().parents[2] / config["paths"]["raw_data"]
    out_root = Path(__file__).resolve().parents[2] / "data" / "processed" / "parsed"
    out_root.mkdir(parents=True, exist_ok=True)

    handlers = {".pdf": parse_pdf, ".html": parse_html, ".htm": parse_html, ".txt": parse_txt}

    all_files = [p for p in raw_root.rglob("*") if p.suffix.lower() in handlers]

    if not all_files:
        print(f"No documents found under {raw_root}.")
        print("Add source documents first — see the README.md in each data/raw/* subfolder.")
        return

    print(f"Found {len(all_files)} documents to parse.")
    for path in tqdm(all_files):
        try:
            text = handlers[path.suffix.lower()](path)
        except Exception as e:
            print(f"  [error] failed to parse {path}: {e}")
            continue

        if not text.strip():
            print(f"  [warn] no text extracted from {path.name} — skipping")
            continue

        # Preserve source subfolder (superfund/envirostor/etc.) in the output filename
        source_category = path.relative_to(raw_root).parts[0]
        out_name = f"{source_category}__{path.stem}.txt"
        with open(out_root / out_name, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Done. Parsed text written to {out_root}")


if __name__ == "__main__":
    main()
