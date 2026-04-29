from __future__ import annotations

import json
from pathlib import Path

from src.adapters.parsers.mineru25 import MinerU25Parser


def test_parse_reads_current_mineru_middle_json(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    def _fake_do_parse(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        parse_dir = output_dir / pdf_path.name / "auto"
        images_dir = parse_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for name in ["table0.png", "formula0.png", "figure0.png", "chart0.png"]:
            (images_dir / name).write_bytes(b"img")

        middle_json = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "page_size": [1000, 2000],
                    "para_blocks": [
                        {
                            "type": "table",
                            "bbox": [10, 20, 110, 220],
                            "blocks": [
                                {
                                    "type": "table_body",
                                    "lines": [
                                        {
                                            "spans": [
                                                {
                                                    "type": "table",
                                                    "image_path": "table0.png",
                                                    "html": (
                                                        "<table><tr><th>A</th><th>B</th></tr>"
                                                        "<tr><td>1</td><td>2</td></tr></table>"
                                                    ),
                                                }
                                            ]
                                        }
                                    ],
                                },
                                {
                                    "type": "table_caption",
                                    "lines": [
                                        {
                                            "spans": [
                                                {"type": "text", "content": "Table 1: Values"}
                                            ]
                                        }
                                    ],
                                },
                            ],
                        },
                        {
                            "type": "interline_equation",
                            "bbox": [30, 40, 50, 60],
                            "lines": [
                                {
                                    "spans": [
                                        {
                                            "type": "interline_equation",
                                            "image_path": "formula0.png",
                                            "content": r"E = mc^2",
                                        }
                                    ]
                                }
                            ],
                        },
                        {
                            "type": "image",
                            "bbox": [100, 200, 300, 400],
                            "blocks": [
                                {
                                    "type": "image_body",
                                    "lines": [
                                        {
                                            "spans": [
                                                {
                                                    "type": "image",
                                                    "image_path": "figure0.png",
                                                }
                                            ]
                                        }
                                    ],
                                },
                                {
                                    "type": "image_caption",
                                    "lines": [
                                        {
                                            "spans": [
                                                {"type": "text", "content": "Figure 1: Diagram"}
                                            ]
                                        }
                                    ],
                                },
                            ],
                        },
                    ],
                    "discarded_blocks": [
                        {
                            "type": "chart",
                            "bbox": [500, 600, 700, 800],
                            "blocks": [
                                {
                                    "type": "chart_body",
                                    "lines": [
                                        {
                                            "spans": [
                                                {
                                                    "type": "chart",
                                                    "image_path": "chart0.png",
                                                    "content": "",
                                                }
                                            ]
                                        }
                                    ],
                                },
                                {
                                    "type": "chart_caption",
                                    "lines": [
                                        {
                                            "spans": [
                                                {
                                                    "type": "text",
                                                    "content": "Chart 1: Trend",
                                                }
                                            ]
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ]
        }
        (parse_dir / f"{pdf_path.name}_middle.json").write_text(json.dumps(middle_json))

    adapter = MinerU25Parser()
    adapter._do_parse = _fake_do_parse

    tables, formulas, figures = adapter.parse(pdf_path, "doc1")

    assert len(tables) == 1
    assert tables[0].page_number == 0
    assert tables[0].bbox is not None
    assert tables[0].headers == ["A", "B"]
    assert tables[0].rows == [["A", "B"], ["1", "2"]]
    assert "Table 1: Values" in tables[0].content

    assert len(formulas) == 1
    assert formulas[0].latex == r"E = mc^2"
    assert formulas[0].image_path is not None
    assert Path(formulas[0].image_path).exists()

    assert len(figures) == 2
    assert figures[0].caption == "Figure 1: Diagram"
    assert figures[1].caption == "Chart 1: Trend"
    assert all(Path(fig.image_path).exists() for fig in figures)
