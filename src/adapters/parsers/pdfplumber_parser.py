from __future__ import annotations

import tempfile
from pathlib import Path

from ...core.models.elements import Figure, Formula, Table
from ...core.registry import register_structured_parser
from ...utils.ids import generate_element_id


@register_structured_parser("pdfplumber")
class PdfPlumberParser:
    """Structured parser using pdfplumber (tables) and PyMuPDF (figures).

    CPU-native alternative to MinerU2.5. Extracts tables and embedded images
    without any ML models. Formulas are not extracted (returns empty list) —
    use the pix2tex adapter separately if formula recognition is needed.

    Requires: pip install pdfplumber pymupdf
    """

    TOOL_NAME = "pdfplumber"
    TOOL_VERSION = "0.11"

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def parse(
        self, pdf_path: Path, doc_id: str
    ) -> tuple[list[Table], list[Formula], list[Figure]]:
        tables = self._extract_tables(pdf_path, doc_id)
        figures = self._extract_figures(pdf_path, doc_id)
        # Formulas require an ML model; return empty so the pipeline enrichment
        # step is a clean no-op rather than silently skipping.
        return tables, [], figures

    def _extract_tables(self, pdf_path: Path, doc_id: str) -> list[Table]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber not installed. Run: pip install pdfplumber"
            )

        tables: list[Table] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_number, page in enumerate(pdf.pages):
                raw_tables = page.extract_tables()
                for seq, rows in enumerate(raw_tables):
                    if not rows:
                        continue
                    tables.append(
                        self._rows_to_table(rows, doc_id, page_number, str(pdf_path), seq)
                    )
        return tables

    def _rows_to_table(
        self,
        rows: list[list[str | None]],
        doc_id: str,
        page_number: int,
        source_file: str,
        seq: int,
    ) -> Table:
        # Replace None cells (merged cells in pdfplumber) with empty string
        clean_rows = [
            [cell if cell is not None else "" for cell in row] for row in rows
        ]
        headers = clean_rows[0] if clean_rows else []
        return Table(
            object_id=generate_element_id(
                doc_id, page_number, "table", self.TOOL_NAME, seq
            ),
            doc_id=doc_id,
            source_file=source_file,
            page_number=page_number,
            tool_name=self.TOOL_NAME,
            tool_version=self.TOOL_VERSION,
            content=self._rows_to_markdown(clean_rows),
            rows=clean_rows,
            headers=headers,
        )

    def _rows_to_markdown(self, rows: list[list[str]]) -> str:
        if not rows:
            return ""
        lines = [" | ".join(rows[0])]
        lines.append(" | ".join("---" for _ in rows[0]))
        lines.extend(" | ".join(row) for row in rows[1:])
        return "\n".join(lines)

    # Caption prefixes in multiple languages — extend as needed
    _CAPTION_PREFIXES = ("figure", "fig.", "abbildung", "abb.")

    def _extract_figures(self, pdf_path: Path, doc_id: str) -> list[Figure]:
        try:
            import fitz
        except ImportError:
            raise ImportError("pymupdf not installed. Run: pip install pymupdf")

        figures: list[Figure] = []
        doc = fitz.open(str(pdf_path))
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="techpdf_figures_"))
            for page_number, page in enumerate(doc):
                text_blocks = page.get_text("blocks")
                for seq, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    img_data = doc.extract_image(xref)
                    if not img_data:
                        continue
                    img_path = tmp_dir / f"{doc_id}_p{page_number}_f{seq}.png"
                    img_path.write_bytes(img_data["image"])
                    caption = self._find_caption(img_info, text_blocks)
                    figures.append(
                        self._image_to_figure(
                            img_path, doc_id, page_number, str(pdf_path), seq, caption
                        )
                    )
        finally:
            doc.close()
        return figures

    def _find_caption(self, img_info: tuple, text_blocks: list) -> str | None:
        """Search nearby text blocks for a caption line.

        Looks for blocks whose text starts with a known caption prefix.
        Returns the first match or None when nothing is found.
        """
        for block in text_blocks:
            if block[6] != 0:  # skip non-text blocks
                continue
            text = block[4].strip()
            if text.lower().startswith(self._CAPTION_PREFIXES):
                return text
        return None

    def _image_to_figure(
        self,
        img_path: Path,
        doc_id: str,
        page_number: int,
        source_file: str,
        seq: int,
        caption: str | None = None,
    ) -> Figure:
        return Figure(
            object_id=generate_element_id(
                doc_id, page_number, "figure", self.TOOL_NAME, seq
            ),
            doc_id=doc_id,
            source_file=source_file,
            page_number=page_number,
            tool_name=self.TOOL_NAME,
            tool_version=self.TOOL_VERSION,
            image_path=str(img_path),
            caption=caption,
        )
