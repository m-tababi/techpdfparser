from __future__ import annotations

from pathlib import Path

from ...core.models.elements import TextChunk
from ...core.registry import register_text_extractor
from ...utils.ids import generate_element_id


@register_text_extractor("olmocr2")
class OlmOCR2Extractor:
    """Text extractor using olmOCR-2.

    olmOCR-2 produces reading-order-aware markdown output, which is
    critical for technical documents with multi-column layouts, figures,
    and captions interspersed with body text.

    Replace with a native PDF text layer extractor (e.g. via PyMuPDF) by
    registering a new adapter under a different name and updating the config.

    Model: allenai/olmOCR-2
    Requires: pip install olmocr transformers torch
    """

    TOOL_NAME = "olmocr2"
    TOOL_VERSION = "2.0"

    def __init__(
        self, model_name: str = "allenai/olmOCR-2", device: str = "cuda"
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self._model_name, device_map=self._device
            )
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

    @property
    def tool_name(self) -> str:
        return self.TOOL_NAME

    @property
    def tool_version(self) -> str:
        return self.TOOL_VERSION

    def extract_page(
        self, pdf_path: Path, page_number: int, doc_id: str
    ) -> list[TextChunk]:
        """Render page to image, run olmOCR-2, return a single TextChunk."""
        self._load()
        # Import here to avoid circular deps (renderer → adapter → renderer)
        from ..renderers.pymupdf import PyMuPDFRenderer

        image = PyMuPDFRenderer().render_page(pdf_path, page_number)
        text = self._run_ocr(image)

        return [
            TextChunk(
                object_id=generate_element_id(
                    doc_id, page_number, "text_chunk", self.TOOL_NAME
                ),
                doc_id=doc_id,
                source_file=str(pdf_path),
                page_number=page_number,
                tool_name=self.TOOL_NAME,
                tool_version=self.TOOL_VERSION,
                content=text,
            )
        ]

    def extract_all(self, pdf_path: Path, doc_id: str) -> list[TextChunk]:
        from ..renderers.pymupdf import PyMuPDFRenderer

        count = PyMuPDFRenderer().page_count(pdf_path)
        chunks: list[TextChunk] = []
        for page_num in range(count):
            chunks.extend(self.extract_page(pdf_path, page_num, doc_id))
        return chunks

    def _run_ocr(self, image) -> str:
        """Run olmOCR-2 inference and return raw markdown text."""
        import torch

        prompt = "Extract all text from this document page in reading order."
        inputs = self._processor(
            images=[image], text=prompt, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=2048)
        return self._processor.decode(output[0], skip_special_tokens=True)
