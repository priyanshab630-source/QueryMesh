"""PDF -> text.

pypdf is the default. If Docling is installed it is used instead, because its
layout-aware parsing keeps tables and section order intact:

    pip install docling
"""

from pathlib import Path


def parse(path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        from docling.document_converter import DocumentConverter   # optional

        result = DocumentConverter().convert(str(path))
        return result.document.export_to_markdown()
    except Exception:                            # noqa: BLE001 - fall through
        pass

    from pypdf import PdfReader                  # default path

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def parse_sections(path) -> dict:
    """Very small heuristic split into abstract / body / references."""
    text = parse(path)
    lowered = text.lower()
    sections = {"abstract": "", "body": text, "references": ""}

    if "abstract" in lowered:
        start = lowered.index("abstract") + len("abstract")
        end = lowered.find("introduction", start)
        sections["abstract"] = text[start:end if end != -1 else start + 2000].strip()

    if "references" in lowered:
        cut = lowered.rindex("references")
        sections["body"] = text[:cut]
        sections["references"] = text[cut:]

    return sections
