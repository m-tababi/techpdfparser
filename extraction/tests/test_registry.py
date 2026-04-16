from extraction.registry import (
    get_renderer,
    get_segmenter,
    get_text_extractor,
    get_table_extractor,
    get_formula_extractor,
    get_figure_descriptor,
    register_renderer,
    register_segmenter,
    register_text_extractor,
    register_table_extractor,
    register_formula_extractor,
    register_figure_descriptor,
)


def test_register_and_get_renderer():
    @register_renderer("test_renderer")
    class TestRenderer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    instance = get_renderer("test_renderer", dpi=150)
    assert instance.kwargs == {"dpi": 150}


def test_register_and_get_segmenter():
    @register_segmenter("test_seg")
    class TestSeg:
        pass

    instance = get_segmenter("test_seg")
    assert instance is not None


def test_unknown_adapter_raises_key_error():
    import pytest

    with pytest.raises(KeyError, match="Unknown renderer"):
        get_renderer("nonexistent_adapter_xyz")


def test_register_all_adapter_types():
    @register_text_extractor("test_ocr")
    class T1:
        pass

    @register_table_extractor("test_table")
    class T2:
        pass

    @register_formula_extractor("test_formula")
    class T3:
        pass

    @register_figure_descriptor("test_fig")
    class T4:
        pass

    assert get_text_extractor("test_ocr") is not None
    assert get_table_extractor("test_table") is not None
    assert get_formula_extractor("test_formula") is not None
    assert get_figure_descriptor("test_fig") is not None
