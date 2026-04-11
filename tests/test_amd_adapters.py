"""Tests for AMD/CPU-compatible adapters.

All tests run without downloading or loading models — heavy imports are mocked.
This makes the suite fast and CI-friendly even without GPU dependencies installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_image() -> PILImage.Image:
    return PILImage.new("RGB", (64, 64), color=(128, 128, 128))


# ---------------------------------------------------------------------------
# Registration tests — no model loading required
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_clip_registered(self):
        from src.core.registry import _VISUAL_EMBEDDERS
        assert "clip" in _VISUAL_EMBEDDERS

    def test_pymupdf_text_registered(self):
        from src.core.registry import _TEXT_EXTRACTORS
        assert "pymupdf_text" in _TEXT_EXTRACTORS

    def test_minilm_registered(self):
        from src.core.registry import _TEXT_EMBEDDERS
        assert "minilm" in _TEXT_EMBEDDERS

    def test_pdfplumber_registered(self):
        from src.core.registry import _STRUCTURED_PARSERS
        assert "pdfplumber" in _STRUCTURED_PARSERS

    def test_pix2tex_registered(self):
        from src.core.registry import _FORMULA_EXTRACTORS
        assert "pix2tex" in _FORMULA_EXTRACTORS

    def test_moondream_registered(self):
        from src.core.registry import _FIGURE_DESCRIPTORS
        assert "moondream" in _FIGURE_DESCRIPTORS


# ---------------------------------------------------------------------------
# Properties before model load (lazy-load guard)
# ---------------------------------------------------------------------------

class TestPropertiesBeforeLoad:
    def test_clip_properties(self):
        from src.adapters.visual.clip import CLIPEmbedder
        a = CLIPEmbedder()
        assert a.tool_name == "clip"
        assert a.embedding_dim == 512
        assert a.is_multi_vector is True
        assert a._model is None

    def test_minilm_properties(self):
        from src.adapters.embedders.minilm import MiniLMEmbedder
        a = MiniLMEmbedder()
        assert a.tool_name == "minilm"
        assert a.embedding_dim == 384
        assert a._model is None

    def test_pymupdf_text_properties(self):
        from src.adapters.ocr.pymupdf_text import PyMuPDFTextExtractor
        a = PyMuPDFTextExtractor()
        assert a.tool_name == "pymupdf_text"
        assert a.tool_version == "1.24"

    def test_pdfplumber_properties(self):
        from src.adapters.parsers.pdfplumber_parser import PdfPlumberParser
        a = PdfPlumberParser()
        assert a.tool_name == "pdfplumber"

    def test_pix2tex_model_none_before_load(self):
        from src.adapters.formula.pix2tex import Pix2TexExtractor
        a = Pix2TexExtractor()
        assert a._model is None

    def test_moondream_model_none_before_load(self):
        from src.adapters.figures.moondream import MoondreamDescriptor
        a = MoondreamDescriptor()
        assert a._model is None
        assert a._tokenizer is None


# ---------------------------------------------------------------------------
# CLIPEmbedder — contract shape tests
# ---------------------------------------------------------------------------

class TestCLIPEmbedder:
    def _mock_clip(self):
        """Return a mock CLIPModel that produces a 512-dim feature vector."""
        import torch
        mock_model = MagicMock()
        mock_model.get_image_features.return_value = torch.ones(1, 512)
        mock_model.get_text_features.return_value = torch.ones(1, 512)
        # Processor returns a MagicMock so .to(device) works without a real BatchEncoding
        mock_processor = MagicMock()
        return mock_model, mock_processor

    def test_embed_page_returns_single_wrapped_vector(self):
        from src.adapters.visual.clip import CLIPEmbedder
        adapter = CLIPEmbedder()
        mock_model, mock_processor = self._mock_clip()
        adapter._model = mock_model
        adapter._processor = mock_processor

        result = adapter.embed_page(_fake_image())
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 512

    def test_embed_query_returns_single_wrapped_vector(self):
        from src.adapters.visual.clip import CLIPEmbedder
        import torch
        adapter = CLIPEmbedder()
        mock_model, mock_processor = self._mock_clip()
        adapter._model = mock_model
        adapter._processor = mock_processor

        result = adapter.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 512

    def test_load_raises_on_missing_transformers(self):
        from src.adapters.visual.clip import CLIPEmbedder
        adapter = CLIPEmbedder()
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                adapter._load()


# ---------------------------------------------------------------------------
# MiniLMEmbedder — contract shape tests
# ---------------------------------------------------------------------------

class TestMiniLMEmbedder:
    def _make_adapter_with_mock(self):
        from src.adapters.embedders.minilm import MiniLMEmbedder
        adapter = MiniLMEmbedder()
        mock_model = MagicMock()
        # encode() returns a 2-D numpy array: (n_texts, 384)
        mock_model.encode.side_effect = lambda batch, **kw: np.zeros(
            (len(batch), 384), dtype=np.float32
        )
        adapter._model = mock_model
        return adapter

    def test_embed_returns_list_of_vectors(self):
        adapter = self._make_adapter_with_mock()
        result = adapter.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 384

    def test_embed_query_returns_flat_vector(self):
        adapter = self._make_adapter_with_mock()
        result = adapter.embed_query("query text")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_batching_respects_batch_size(self):
        from src.adapters.embedders.minilm import MiniLMEmbedder
        adapter = MiniLMEmbedder(batch_size=2)
        mock_model = MagicMock()
        calls = []
        def _encode(batch, **kw):
            calls.append(len(batch))
            return np.zeros((len(batch), 384), dtype=np.float32)
        mock_model.encode.side_effect = _encode
        adapter._model = mock_model

        adapter.embed(["a", "b", "c", "d", "e"])
        # batch_size=2 → batches of [2, 2, 1]
        assert calls == [2, 2, 1]

    def test_load_raises_on_missing_sentence_transformers(self):
        from src.adapters.embedders.minilm import MiniLMEmbedder
        adapter = MiniLMEmbedder()
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                adapter._load()


# ---------------------------------------------------------------------------
# PyMuPDFTextExtractor — logic tests with mocked fitz
# ---------------------------------------------------------------------------

class TestPyMuPDFTextExtractor:
    def _make_blocks(self, texts: list[str]) -> list[tuple]:
        """Build fitz-style block tuples: (x0,y0,x1,y1, text, block_no, type)."""
        return [(0, i * 10, 100, (i + 1) * 10, t, i, 0) for i, t in enumerate(texts)]

    def test_extract_page_returns_text_chunks(self):
        from pathlib import Path
        from src.adapters.ocr.pymupdf_text import PyMuPDFTextExtractor

        adapter = PyMuPDFTextExtractor()
        blocks = self._make_blocks(["Hello world", "Second block"])

        mock_page = MagicMock()
        mock_page.get_text.return_value = blocks
        mock_doc = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("fitz.open", return_value=mock_doc):
            result = adapter.extract_page(Path("test.pdf"), 0, "doc1")

        assert len(result) == 2
        assert result[0].content == "Hello world"
        assert result[1].content == "Second block"

    def test_skips_whitespace_only_blocks(self):
        from pathlib import Path
        from src.adapters.ocr.pymupdf_text import PyMuPDFTextExtractor

        adapter = PyMuPDFTextExtractor()
        blocks = self._make_blocks(["  \n  ", "Real content", "\t"])

        mock_page = MagicMock()
        mock_page.get_text.return_value = blocks
        mock_doc = MagicMock()
        mock_doc.__getitem__.return_value = mock_page

        with patch("fitz.open", return_value=mock_doc):
            result = adapter._blocks_to_chunks(blocks, "doc1", 0, "test.pdf")

        assert len(result) == 1
        assert result[0].content == "Real content"

    def test_skips_image_blocks(self):
        from src.adapters.ocr.pymupdf_text import PyMuPDFTextExtractor

        adapter = PyMuPDFTextExtractor()
        # block_type=1 is an image block — should be skipped
        blocks = [
            (0, 0, 100, 50, "text content", 0, 0),
            (0, 50, 100, 100, "image data", 1, 1),
        ]
        result = adapter._blocks_to_chunks(blocks, "doc1", 0, "test.pdf")
        assert len(result) == 1
        assert result[0].content == "text content"

    def test_page_number_and_doc_id_set_correctly(self):
        from src.adapters.ocr.pymupdf_text import PyMuPDFTextExtractor

        adapter = PyMuPDFTextExtractor()
        blocks = self._make_blocks(["Some text"])
        result = adapter._blocks_to_chunks(blocks, "mydoc", 5, "file.pdf")

        assert result[0].doc_id == "mydoc"
        assert result[0].page_number == 5
        assert result[0].source_file == "file.pdf"


# ---------------------------------------------------------------------------
# PdfPlumberParser — formula list is always empty
# ---------------------------------------------------------------------------

class TestPdfPlumberParser:
    def test_parse_formulas_always_empty(self):
        from pathlib import Path
        from src.adapters.parsers.pdfplumber_parser import PdfPlumberParser

        adapter = PdfPlumberParser()

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([]))

        with patch("pdfplumber.open", return_value=mock_pdf), \
             patch("fitz.open", return_value=mock_doc):
            tables, formulas, figures = adapter.parse(Path("test.pdf"), "doc1")

        assert formulas == []
        assert isinstance(tables, list)
        assert isinstance(figures, list)

    def test_rows_to_markdown_format(self):
        from src.adapters.parsers.pdfplumber_parser import PdfPlumberParser

        adapter = PdfPlumberParser()
        rows = [["Name", "Value"], ["alpha", "1"], ["beta", "2"]]
        md = adapter._rows_to_markdown(rows)
        lines = md.splitlines()
        assert "Name" in lines[0]
        assert "---" in lines[1]
        assert "alpha" in lines[2]

    def test_rows_to_markdown_empty(self):
        from src.adapters.parsers.pdfplumber_parser import PdfPlumberParser

        adapter = PdfPlumberParser()
        assert adapter._rows_to_markdown([]) == ""

    def test_load_raises_on_missing_pdfplumber(self):
        from pathlib import Path
        from src.adapters.parsers.pdfplumber_parser import PdfPlumberParser

        adapter = PdfPlumberParser()
        with patch.dict(sys.modules, {"pdfplumber": None}):
            with pytest.raises(ImportError, match="pdfplumber"):
                adapter._extract_tables(Path("test.pdf"), "doc1")


# ---------------------------------------------------------------------------
# Pix2TexExtractor
# ---------------------------------------------------------------------------

class TestPix2TexExtractor:
    def test_extract_returns_formula_list(self):
        from src.adapters.formula.pix2tex import Pix2TexExtractor

        adapter = Pix2TexExtractor()
        mock_model = MagicMock(return_value=r"E = mc^2")
        adapter._model = mock_model

        result = adapter.extract(_fake_image(), doc_id="doc1", page_number=0)

        assert len(result) == 1
        assert result[0].latex == r"E = mc^2"

    def test_extract_returns_empty_on_exception(self):
        from src.adapters.formula.pix2tex import Pix2TexExtractor

        adapter = Pix2TexExtractor()
        adapter._model = MagicMock(side_effect=RuntimeError("not a formula"))

        result = adapter.extract(_fake_image())
        assert result == []

    def test_extract_returns_empty_on_blank_latex(self):
        from src.adapters.formula.pix2tex import Pix2TexExtractor

        adapter = Pix2TexExtractor()
        adapter._model = MagicMock(return_value="   ")

        result = adapter.extract(_fake_image())
        assert result == []

    def test_crop_applied_when_bbox_given(self):
        from src.adapters.formula.pix2tex import Pix2TexExtractor
        from src.core.models.document import BoundingBox

        adapter = Pix2TexExtractor()
        captured = []
        def _capture(img):
            captured.append(img.size)
            return r"\alpha"
        adapter._model = _capture

        bbox = BoundingBox(x0=0, y0=0, x1=32, y1=32)
        adapter.extract(_fake_image(), bbox=bbox, doc_id="d", page_number=1)
        assert captured[0] == (32, 32)

    def test_load_raises_on_missing_pix2tex(self):
        from src.adapters.formula.pix2tex import Pix2TexExtractor

        adapter = Pix2TexExtractor()
        with patch.dict(sys.modules, {"pix2tex": None, "pix2tex.cli": None}):
            with pytest.raises(ImportError, match="pix2tex"):
                adapter._load()


# ---------------------------------------------------------------------------
# MoondreamDescriptor
# ---------------------------------------------------------------------------

class TestMoondreamDescriptor:
    def test_describe_returns_string(self):
        from src.adapters.figures.moondream import MoondreamDescriptor

        adapter = MoondreamDescriptor()
        mock_model = MagicMock()
        mock_model.encode_image.return_value = MagicMock()
        mock_model.answer_question.return_value = "A bar chart showing quarterly sales."
        adapter._model = mock_model
        adapter._tokenizer = MagicMock()

        result = adapter.describe(_fake_image())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_load_raises_on_missing_transformers(self):
        from src.adapters.figures.moondream import MoondreamDescriptor

        adapter = MoondreamDescriptor()
        with patch.dict(sys.modules, {"transformers": None}):
            with pytest.raises(ImportError, match="transformers"):
                adapter._load()
