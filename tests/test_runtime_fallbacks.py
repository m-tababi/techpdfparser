from __future__ import annotations

import sys
import types

from PIL import Image

from src.adapters.figures.qwen25vl import Qwen25VLDescriptor
from src.adapters.ocr.olmocr2 import OlmOCR2Extractor


def test_olmocr_falls_back_to_cpu_on_cuda_oom(monkeypatch):
    calls: list[str] = []

    class FakeProcessor:
        @staticmethod
        def from_pretrained(_model_name):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(_model_name, *, device_map, torch_dtype):
            calls.append(device_map)
            assert torch_dtype == "auto"
            if device_map == "cuda":
                raise RuntimeError("CUDA out of memory")
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModelForVision2Seq=FakeModel,
            AutoProcessor=FakeProcessor,
        ),
    )

    extractor = OlmOCR2Extractor(model_name="fake/olmocr", device="cuda")
    extractor._load()

    assert calls == ["cuda", "cpu"]
    assert extractor._runtime_device == "cpu"
    assert extractor._model is not None
    assert extractor._processor is not None


def test_qwen_falls_back_to_cpu_on_cuda_oom(monkeypatch):
    calls: list[str] = []

    class FakeProcessor:
        @staticmethod
        def from_pretrained(_model_name):
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(_model_name, *, device_map, torch_dtype):
            calls.append(device_map)
            assert torch_dtype == "auto"
            if device_map == "cuda":
                raise RuntimeError("CUDA out of memory")
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoProcessor=FakeProcessor,
            Qwen2_5_VLForConditionalGeneration=FakeModel,
        ),
    )

    descriptor = Qwen25VLDescriptor(model_name="fake/qwen", device="cuda")
    descriptor._load()

    assert calls == ["cuda", "cpu"]
    assert descriptor._runtime_device == "cpu"
    assert descriptor._model is not None
    assert descriptor._processor is not None


def test_olmocr_strips_front_matter():
    extractor = OlmOCR2Extractor(model_name="fake/olmocr", device="cpu")

    output = extractor._strip_front_matter(
        "---\nprimary_language: en\nis_table: false\n---\nReal text body"
    )

    assert output == "Real text body"


def test_olmocr_resizes_page_to_expected_longest_edge():
    extractor = OlmOCR2Extractor(model_name="fake/olmocr", device="cpu")
    image = Image.new("RGB", (2000, 1000), color="white")

    resized = extractor._prepare_image(image)

    assert max(resized.size) == 1288
    assert resized.size == (1288, 644)
