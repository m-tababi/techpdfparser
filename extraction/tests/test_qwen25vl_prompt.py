"""Unit-Tests für die Prompt-Konstruktion des Qwen-Descriptors.

Lädt das Model nicht — testet nur die Text-Assembly.
"""
from __future__ import annotations

from extraction.adapters.qwen25vl_figure import _build_prompt


def test_prompt_without_caption_is_image_only():
    prompt = _build_prompt(caption=None)
    assert "caption" not in prompt.lower()
    assert "describe this figure" in prompt.lower()


def test_prompt_with_caption_includes_it_as_grounding():
    prompt = _build_prompt(caption="Figure 2. Tensile test specimens.")
    assert "Figure 2. Tensile test specimens." in prompt
    # Das Prompt soll explizit sagen: nichts erfinden, Caption als Kontext.
    assert "do not invent" in prompt.lower() or "nicht erfinden" in prompt.lower()


def test_prompt_with_empty_caption_treated_as_none():
    assert _build_prompt(caption="") == _build_prompt(caption=None)
    assert _build_prompt(caption="   ") == _build_prompt(caption=None)
