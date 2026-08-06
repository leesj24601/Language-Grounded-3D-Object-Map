from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "outputs" / "gt_aligned_10_label" / "metrics"
DEFAULT_OUTPUT_DIR = Path("/tmp/notion-chart-selection-emphasis")

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT = FontProperties(fname=FONT_PATH)
FONT_BOLD = FontProperties(fname=FONT_PATH, weight="bold")

SERIES = [
    {
        "key": "precision",
        "label": "Precision",
        "color": "#E8EDF3",
        "edge": "#4F6F91",
        "hatch": "///",
    },
    {
        "key": "recall",
        "label": "Recall",
        "color": "#D8A545",
        "edge": "#8A6320",
        "hatch": None,
    },
    {
        "key": "duplicate_rate",
        "label": "Duplicate rate",
        "color": "#F7F8FA",
        "edge": "#6E7781",
        "hatch": "xx",
    },
]

EXPECTED = {
    "config": [
        (53.57, 50.00, 46.43),
        (72.00, 60.00, 28.00),
        (84.00, 70.00, 16.00),
        (75.86, 73.33, 24.14),
    ],
    "threshold": [
        (70.00, 70.00, 30.00),
        (84.00, 70.00, 16.00),
        (80.00, 53.33, 20.00),
    ],
    "association": [
        (80.77, 70.00, 19.23),
        (84.00, 70.00, 16.00),
        (76.92, 66.67, 23.08),
        (76.00, 63.33, 24.00),
    ],
}


def selection_span(center: float, group_count: int) -> tuple[float, float]:
    """Return the x-axis bounds for a selected category highlight."""
    half_width = 0.42 if group_count >= 4 else 0.45
    return round(center - half_width, 2), round(center + half_width, 2)


def load_metrics(filename: str) -> dict[str, float]:
    with (METRICS_DIR / filename).open(encoding="utf-8") as file:
        data = json.load(file)
    return {
        "precision": float(data["precision"]) * 100,
        "recall": float(data["recall"]) * 100,
        "duplicate_rate": float(data["duplicate_rate"]) * 100,
        "mean_l2_m": float(data["mean_l2_m"]),
        "distance_threshold_m": float(data["distance_threshold_m"]),
    }


def build_rows() -> dict[str, list[tuple[str, dict[str, float]]]]:
    return {
        "config": [
            ("20F\nminobs 1", load_metrics("metrics_41098076_20frames_text035_minobs1.json")),
            ("50F\nminobs 2", load_metrics("metrics_41098076_50frames_text035_minobs2.json")),
            ("100F\nminobs 4", load_metrics("metrics_41098076_100frames_text035_minobs4.json")),
            ("200F\nminobs 6", load_metrics("metrics_41098076_200frames_text035_minobs6.json")),
        ],
        "threshold": [
            ("0.30", load_metrics("metrics_41098076_100frames_text030_minobs4.json")),
            ("0.35", load_metrics("metrics_41098076_100frames_text035_minobs4.json")),
            ("0.40", load_metrics("metrics_41098076_100frames_text040_minobs4.json")),
        ],
        "association": [
            ("0.5 m", load_metrics("metrics_41098076_100frames_text035_assoc050_minobs4.json")),
            ("0.6 m", load_metrics("metrics_41098076_100frames_text035_minobs4.json")),
            ("0.7 m", load_metrics("metrics_41098076_100frames_text035_assoc070_minobs4.json")),
            ("0.8 m", load_metrics("metrics_41098076_100frames_text035_assoc080_minobs4.json")),
        ],
    }


def validate_rows(name: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    for index, (_, row) in enumerate(rows):
        actual = (row["precision"], row["recall"], row["duplicate_rate"])
        expected = EXPECTED[name][index]
        if any(abs(actual_value - expected_value) > 0.005 for actual_value, expected_value in zip(actual, expected)):
            raise AssertionError(f"{name}[{index}] expected {expected}, got {actual}")
        if row["distance_threshold_m"] != 1.0:
            raise AssertionError(f"{name}[{index}] does not use the 1.0 m matching threshold")


def draw_chart(
    *,
    rows: list[tuple[str, dict[str, float]]],
    title: str,
    subtitle: str,
    xlabel: str,
    selected_index: int,
    source_note: str,
    output_stem: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    x = np.arange(len(rows), dtype=float)
    width = 0.205 if len(rows) >= 4 else 0.23
    offsets = np.array([-width, 0.0, width])

    left, right = selection_span(float(x[selected_index]), len(rows))
    ax.axvspan(
        left,
        right,
        ymin=0.0,
        ymax=0.96,
        facecolor="#FFF3D6",
        edgecolor="#C58B25",
        linewidth=2.0,
        alpha=0.72,
        zorder=1,
    )

    for series_index, series in enumerate(SERIES):
        values = [row[series["key"]] for _, row in rows]
        bars = ax.bar(
            x + offsets[series_index],
            values,
            width,
            label=series["label"],
            color=series["color"],
            edgecolor=series["edge"],
            linewidth=1.8,
            hatch=series["hatch"],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontproperties=FONT_BOLD,
                fontsize=11.2,
                color="#20262E",
                zorder=4,
            )

    ax.text(
        x[selected_index],
        104,
        "SELECTED",
        ha="center",
        va="center",
        fontproperties=FONT_BOLD,
        fontsize=10.5,
        color="#FFFFFF",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "#C58B25",
            "edgecolor": "#8A6320",
            "linewidth": 1.0,
        },
        zorder=5,
    )

    fig.suptitle(
        title,
        x=0.09,
        y=0.965,
        ha="left",
        fontproperties=FONT_BOLD,
        fontsize=22,
        color="#20262E",
    )
    ax.set_title(
        subtitle,
        loc="left",
        pad=18,
        fontproperties=FONT,
        fontsize=12.5,
        color="#66707C",
    )

    ax.set_xticks(
        x,
        [label for label, _ in rows],
        fontproperties=FONT_BOLD,
        fontsize=12,
    )
    tick_labels = ax.get_xticklabels()
    tick_labels[selected_index].set_color("#8A6320")
    ax.set_xlabel(xlabel, fontproperties=FONT, fontsize=11.5, color="#353C45", labelpad=12)
    ax.set_ylabel("비율 (%)", fontproperties=FONT, fontsize=12, color="#353C45")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontproperties=FONT, fontsize=10.5)
    ax.grid(axis="y", color="#DDE2E8", linewidth=0.9, zorder=0)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", length=0, colors="#5D6672")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AEB6C0")
    ax.spines["bottom"].set_linewidth(1.0)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.57, 1.13),
        ncol=3,
        frameon=False,
        prop=FONT,
        handlelength=1.8,
        columnspacing=1.8,
    )
    for legend_text in legend.get_texts():
        legend_text.set_color("#353C45")

    fig.text(
        0.09,
        0.025,
        source_note,
        ha="left",
        fontproperties=FONT,
        fontsize=10.5,
        color="#6E7781",
    )

    plt.tight_layout(rect=[0.06, 0.08, 0.98, 0.90])
    for extension, kwargs in [
        ("svg", {"format": "svg"}),
        ("png", {"format": "png", "dpi": 180}),
    ]:
        fig.savefig(
            output_dir / f"{output_stem}.{extension}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Notion-ready semantic-map experiment charts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for SVG and PNG outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs("/tmp/semantic-map-mpl-cache", exist_ok=True)

    rows_by_name = build_rows()
    for name, rows in rows_by_name.items():
        validate_rows(name, rows)

    draw_chart(
        rows=rows_by_name["config"],
        title="대표 설정별 성능 비교",
        subtitle="ARKitScenes 41098076 · 설정별 Precision·Recall·Duplicate rate (%)",
        xlabel="Frames · min observations 조합",
        selected_index=2,
        source_note="출처: outputs/gt_aligned_10_label/metrics · Frames와 min observations가 함께 변하는 설정 조합 비교",
        output_stem="semantic_map_config_comparison_selected",
        output_dir=output_dir,
    )
    draw_chart(
        rows=rows_by_name["threshold"],
        title="Text threshold별 성능 비교",
        subtitle="100 frames · association distance 0.6 m · min observations 4 · 비율 (%)",
        xlabel="Text threshold",
        selected_index=1,
        source_note="출처: outputs/gt_aligned_10_label/metrics · 동일 조건에서 text threshold만 변경",
        output_stem="semantic_map_text_threshold_selected",
        output_dir=output_dir,
    )
    draw_chart(
        rows=rows_by_name["association"],
        title="Association distance별 성능 비교",
        subtitle="100 frames · text threshold 0.35 · min observations 4 · 비율 (%)",
        xlabel="Association distance",
        selected_index=1,
        source_note="출처: outputs/gt_aligned_10_label/metrics · 동일 조건에서 association distance만 변경",
        output_stem="semantic_map_association_distance_selected",
        output_dir=output_dir,
    )

    for stem in [
        "semantic_map_config_comparison_selected",
        "semantic_map_text_threshold_selected",
        "semantic_map_association_distance_selected",
    ]:
        print({"svg": str(output_dir / f"{stem}.svg"), "png": str(output_dir / f"{stem}.png")})


if __name__ == "__main__":
    main()
