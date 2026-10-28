"""PDF -> text.

pypdf is the default. If Docling is installed it is used instead, because its
layout-aware parsing keeps tables and section order intact:

    pip install docling
"""

import os
import sys
from pathlib import Path

import config


def backend() -> str:
    """Which parser will be used: 'docling' (if installed and enabled) or 'pypdf'."""
    if not config.USE_DOCLING:
        return "pypdf"
    try:
        import docling  # noqa: F401

        return "docling"
    except ImportError:
        return "pypdf (docling not installed)"


def parse(path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if not config.USE_DOCLING:
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "")
                            for page in PdfReader(str(path)).pages)

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        # Default pipeline (layout + text). Table structure and OCR pull in extra
        # models, so they are opt-in: DOCLING_TABLES=1 / DOCLING_OCR=1.
        if os.getenv("DOCLING_TABLES") == "1" or os.getenv("DOCLING_OCR") == "1":
            pipeline = PdfPipelineOptions(
                do_table_structure=os.getenv("DOCLING_TABLES") == "1",
                do_ocr=os.getenv("DOCLING_OCR") == "1",
            )
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
            )
        else:
            converter = DocumentConverter()
        result = converter.convert(str(path))
        return result.document.export_to_markdown()
    except Exception as exc:                     # noqa: BLE001 - fall through
        print(f"[pdf] docling failed ({type(exc).__name__}); using pypdf",
              file=sys.stderr)

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
