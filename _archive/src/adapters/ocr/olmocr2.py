from __future__ import annotations

import base64
import re
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

from ...core.models.elements import TextChunk
from ...core.registry import register_text_extractor
from ...utils.ids import generate_element_id
from ...utils.runtime import is_cuda_oom, release_runtime_resources

_OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document as if you were "
    "reading it naturally.\n"
    "Convert equations to LateX and tables to HTML.\n"
    "If there are any figures or charts, label them with the following markdown "
    "syntax ![Alt text describing the contents of the figure]"
    "(page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top "
    "specifying values for the primary_language, is_rotation_valid, "
    "rotation_correction, is_table, and is_diagram parameters."
)
_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
_TARGET_LONGEST_DIM = 1288


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
        self,
        model_name: str = "allenai/olmOCR-2-7B-1025",
        processor_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ) -> None:
        self._model_name = model_name
        self._processor_name = processor_name
        self._device = device
        self._runtime_device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._processor_name)
            self._load_model(AutoModelForVision2Seq)
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )

    def _load_model(self, model_cls) -> None:
        try:
            self._model = model_cls.from_pretrained(
                self._model_name,
                device_map=self._runtime_device,
                torch_dtype="auto",
            )
        except Exception as exc:
            if self._runtime_device == "cpu" or not is_cuda_oom(exc):
                raise

            warnings.warn(
                f"{self.TOOL_NAME} ran out of GPU memory; retrying on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            processor = self._processor
            self._model = None
            release_runtime_resources()
            self._runtime_device = "cpu"
            self._processor = processor
            self._model = model_cls.from_pretrained(
                self._model_name,
                device_map=self._runtime_device,
                torch_dtype="auto",
            )

    def unload(self) -> None:
        self._model = None
        self._processor = None

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

        image = self._prepare_image(image)
        image_base64 = self._image_to_base64(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OLMOCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._runtime_device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.1,
                do_sample=True,
                num_return_sequences=1,
            )
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = self._processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip()
        return self._strip_front_matter(text_output)

    def _prepare_image(self, image):
        longest_dim = max(image.size)
        if longest_dim == _TARGET_LONGEST_DIM:
            return image
        scale = _TARGET_LONGEST_DIM / longest_dim
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        return image.resize(new_size, resample=PILImage.Resampling.LANCZOS)

    def _image_to_base64(self, image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _strip_front_matter(self, text: str) -> str:
        return _FRONT_MATTER_RE.sub("", text, count=1).strip()
