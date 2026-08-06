from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.plot_notion_experiment_results import draw_chart, selection_span


class SelectionSpanTest(unittest.TestCase):
    def test_four_group_chart_uses_compact_span(self) -> None:
        self.assertEqual(selection_span(2.0, 4), (1.58, 2.42))

    def test_three_group_chart_uses_readable_span(self) -> None:
        self.assertEqual(selection_span(1.0, 3), (0.55, 1.45))

    def test_draw_chart_exports_svg_and_png(self) -> None:
        rows = [
            (
                "A",
                {
                    "precision": 80.0,
                    "recall": 70.0,
                    "duplicate_rate": 20.0,
                    "mean_l2_m": 0.3,
                    "distance_threshold_m": 1.0,
                },
            ),
            (
                "B",
                {
                    "precision": 84.0,
                    "recall": 70.0,
                    "duplicate_rate": 16.0,
                    "mean_l2_m": 0.3,
                    "distance_threshold_m": 1.0,
                },
            ),
            (
                "C",
                {
                    "precision": 76.0,
                    "recall": 63.0,
                    "duplicate_rate": 24.0,
                    "mean_l2_m": 0.3,
                    "distance_threshold_m": 1.0,
                },
            ),
        ]
        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            draw_chart(
                rows=rows,
                title="Test chart",
                subtitle="Compatibility regression",
                xlabel="Setting",
                selected_index=1,
                source_note="Test source",
                output_stem="chart",
                output_dir=output_dir,
            )
            self.assertTrue((output_dir / "chart.svg").is_file())
            self.assertTrue((output_dir / "chart.png").is_file())


if __name__ == "__main__":
    unittest.main()
