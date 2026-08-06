# Notion Chart Selection Emphasis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 최종 선택 설정을 금색 강조 사각형과 `SELECTED` 배지로 표시한 실험 차트 3개를 재생성하고, Notion 실험 결과 설명을 중복 없이 정리한다.

**Architecture:** 저장소의 metrics JSON을 읽는 재현 가능한 Matplotlib 스크립트를 `scripts/`에 둔다. 선택 범주 강조는 별도 함수가 계산한 x축 경계로 그리며, 생성된 SVG를 Notion에 업로드한 뒤 기존 이미지 블록과 설명을 표적 교체한다.

**Tech Stack:** Python 3.11, Matplotlib, NumPy, unittest, Notion MCP enhanced Markdown

## Global Constraints

- Precision, Recall, Duplicate rate의 값과 시리즈 인코딩은 변경하지 않는다.
- 세 차트의 y축은 0–112%로 유지한다.
- 선택 상태는 금색 배경과 테두리, `SELECTED` 텍스트를 함께 사용한다.
- 상세 수치와 실험별 해석 토글은 유지한다.
- 차트 산출물은 임시 디렉터리에 생성하고 저장소에는 커밋하지 않는다.

---

### Task 1: 재현 가능한 차트 생성기와 강조 경계 테스트

**Files:**
- Create: `scripts/plot_notion_experiment_results.py`
- Create: `tests/test_plot_notion_experiment_results.py`

**Interfaces:**
- Consumes: `outputs/gt_aligned_10_label/metrics/*.json`
- Produces: `selection_span(center: float, group_count: int) -> tuple[float, float]`, `draw_chart(...) -> None`, `/tmp/notion-chart-selection-emphasis/*.svg|png`

- [ ] **Step 1: 선택 강조 경계 테스트 작성**

```python
class SelectionSpanTest(unittest.TestCase):
    def test_four_group_chart_uses_compact_span(self) -> None:
        self.assertEqual(selection_span(2.0, 4), (1.58, 2.42))

    def test_three_group_chart_uses_readable_span(self) -> None:
        self.assertEqual(selection_span(1.0, 3), (0.55, 1.45))
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

Run: `python -m unittest tests.test_plot_notion_experiment_results -v`
Expected: FAIL because `scripts.plot_notion_experiment_results` does not exist.

- [ ] **Step 3: 차트 생성기 구현**

```python
def selection_span(center: float, group_count: int) -> tuple[float, float]:
    half_width = 0.42 if group_count >= 4 else 0.45
    return round(center - half_width, 2), round(center + half_width, 2)

left, right = selection_span(float(x[selected_index]), len(rows))
ax.axvspan(
    left,
    right,
    facecolor="#FFF3D6",
    edgecolor="#C58B25",
    linewidth=2.0,
    alpha=0.72,
    zorder=1,
)
ax.text(
    x[selected_index],
    104,
    "SELECTED",
    ha="center",
    va="center",
    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#C58B25", "edgecolor": "none"},
    color="#FFFFFF",
    zorder=5,
)
```

스크립트는 기존 기대값 검증을 유지하고 `--output-dir` 인수로 SVG와 PNG를 생성한다.

- [ ] **Step 4: 단위 테스트와 차트 생성을 실행**

Run: `/home/cvr/anaconda3/envs/mqe/bin/python -m unittest tests.test_plot_notion_experiment_results -v`
Expected: 2 tests, OK.

Run: `MPLCONFIGDIR=/tmp/semantic-map-mpl-cache /home/cvr/anaconda3/envs/mqe/bin/python scripts/plot_notion_experiment_results.py --output-dir /tmp/notion-chart-selection-emphasis`
Expected: 3 SVG and 3 PNG paths printed; exit code 0.

- [ ] **Step 5: PNG 시각 검수**

세 PNG에서 선택 사각형이 막대·값 라벨을 가리지 않고, `SELECTED` 배지가 동일한 높이에 보이며, 축과 범례가 잘리지 않는지 확인한다.

- [ ] **Step 6: 구현 커밋**

```bash
git add scripts/plot_notion_experiment_results.py tests/test_plot_notion_experiment_results.py
git commit -m "feat: highlight selected experiment settings"
```

### Task 2: Notion 차트 및 설명 표적 교체

**Files:**
- Modify externally: Notion page `353c937eac2f80e0ad68dca916b46df5`

**Interfaces:**
- Consumes: `/tmp/notion-chart-selection-emphasis/*.svg`
- Produces: 새 이미지 3개와 축약된 결과 설명이 있는 Notion 실험 결과 섹션

- [ ] **Step 1: SVG 3개를 Notion 임시 파일로 업로드**

각 SVG에 대해 Notion file upload를 만들고, 반환된 multipart URL로 파일을 전송한 뒤 `file-upload://...` 소스를 확보한다.

- [ ] **Step 2: 기존 이미지 블록 표적 교체**

세 기존 이미지와 바로 뒤의 중복 캡션을 새 업로드 이미지로 교체한다. 다른 섹션과 상세 토글은 수정하지 않는다.

- [ ] **Step 3: 결과 요약과 출처 문구 축약**

```markdown
<callout icon="📌" color="yellow_bg">
	**결과 요약**
	- 최종 선택: **100 frames · text threshold 0.35 · association distance 0.6m · min observations 4** — Precision **0.840** · Recall **0.700** · Mean L2 Error **0.311m** · Duplicate Rate **0.160**
	- 선택 근거: threshold 0.35와 association distance 0.6m에서 Precision–Recall 균형과 Duplicate 억제가 가장 안정적
	- 해석 주의: Frames와 min observations가 함께 변한 설정 조합 비교이므로 Frame 수의 독립 효과로 해석하지 않음
</callout>
*공통 평가 기준 — ARKitScenes Scene 41098076 · GT-aligned 10 labels · 1.0m matching threshold · [원본 metrics JSON](https://github.com/leesj24601/Language-Grounded-3D-Object-Map/tree/main/outputs/gt_aligned_10_label/metrics)*
```

- [ ] **Step 4: Notion 재조회 검증**

새 SVG 파일명 3개, `SELECTED` 포함 SVG, 축약된 결과 요약 3개 항목, 상세 수치 토글, metrics 링크를 확인한다. 기존 SVG 파일명과 중복 캡션 문구가 없어야 한다.

- [ ] **Step 5: 저장소 상태 확인**

Run: `git status --short`
Expected: 사용자가 만든 변경은 유지되며, 계획 밖 산출물이나 PNG/SVG가 저장소에 추가되지 않는다.
