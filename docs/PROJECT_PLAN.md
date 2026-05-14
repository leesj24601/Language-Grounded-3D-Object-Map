# Language-Grounded 3D Object Mapping
## 컴퓨터 비전 텀 프로젝트 계획서

> **학번/이름**: 21101477 이석준
> **기간**: 5주 (2026-04-24 ~ 2026-05-29)
> **형태**: 개인 프로젝트
> **최종 모호성 점수**: 17.7% (Deep Interview 6라운드 완료)

---

## RALPLAN-DR 요약

### Principles (설계 원칙)
1. **모듈성 우선** — 각 단계(탐지, 세그멘테이션, 3D 투영, 맵 관리, 질의)를 독립 모듈로 분리하여 추후 로봇 파이프라인 재사용 가능
2. **Zero-shot 활용** — 추가 학습 없이 사전 학습 모델(Grounding DINO, SAM)만 사용
3. **검증 가능성 우선** — 공개 RGB-D 데이터셋에서 정량 지표를 산출하여 취업 포트폴리오 설득력 확보
4. **배치 우선** — 실시간 처리 대신 배치 처리로 안정성과 개발 속도를 확보
5. **빠른 완료** — MVP → 평가 → 시각화 순서로 핵심 기능 먼저 완성

### Decision Drivers (핵심 결정 요인)
1. **취업 포트폴리오 가치** — 검증 가능한 ScanNet 단일 scene 수치 + language-grounded 3D mapping 조합으로 설득력 확보
2. **5주 개인 프로젝트 제약** — 실현 가능한 범위로 scope를 제한해야 함
3. **재사용성** — 추후 Isaac Sim / Go2 / 실시간 카메라 파이프라인에 perception 모듈로 연결 가능해야 함

### Options Considered

| 옵션 | 선택 | 이유 |
|------|------|------|
| 데이터: ScanNet 단일 scene | ✅ | 실제/표준 RGB-D sequence + pose + instance annotation을 같은 world 좌표계에서 제공하여 3D object map 평가에 적합 |
| 데이터: Replica 단일 scene | fallback | ScanNet 접근/annotation 처리 실패 시 대체 가능한 RGB-D + pose 데이터 |
| 데이터: Isaac Sim | 후속 확장 | 로봇 환경 데모로는 좋지만, 이번 메인 검증 데이터로는 benchmark 설득력 약함 |
| 탐지: CLIP 기반 | ✗ | bounding box 없어 SAM 연결 불편 |
| 탐지: Grounding DINO + SAM | ✅ | text→bbox→mask 파이프라인 자연스럽게 연결 |
| 3D: TSDF Fusion | ✗ | 복잡도 높음, 텀 프로젝트 과도 |
| 3D: centroid 기반 Object Map | ✅ | 구현 단순, 평가 명확, Non-goal과 일치 |
| 처리: 실시간 | ✗ | Non-goal로 명시, 구현 복잡도 급증 |
| 처리: 배치 처리 | ✅ | 안정적, 빠른 개발, 충분한 성능 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│              ScanNet Single Scene (Primary)                 │
│   one scene, e.g. scene0000_00 / indoor room scan           │
│   → RGB-D frames + Camera Pose + GT Instance Annotation     │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │
┌──────────────────▼──────────────────────────────────────────┐
│          Dataset Adapter (공통 Frame 형식 변환)              │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                  메인 파이프라인 (cv env)                    │
│                                                              │
│  Frame Loader                                                │
│       │                                                      │
│       ▼                                                      │
│  Grounding DINO ──(text prompt)──→ Bounding Box             │
│       │                                                      │
│       ▼                                                      │
│  SAM ────────────(bbox)──────────→ Pixel Mask               │
│       │                                                      │
│       ▼                                                      │
│  Depth Unprojector ─(mask+depth+K)→ 3D Point Cloud          │
│       │                                                      │
│       ▼                                                      │
│  Pose Transformer ─(T_cam2world)──→ World-frame Points      │
│       │                                                      │
│       ▼                                                      │
│  Object Associator ─(dist+label)──→ Object3D (centroid)     │
│       │                                                      │
│       ▼                                                      │
│  Semantic Map ────────────────────→ {label: centroid, ...}  │
│       │                                                      │
│       ▼                                                      │
│  Text Query Engine ──(lookup or expansion)→ Query Result    │
│       │                                                      │
│       ▼                                                      │
│  Evaluator + Visualizer                                      │
└─────────────────────────────────────────────────────────────┘

Optional extension:
┌─────────────────────────────────────────────────────────────┐
│   Isaac Sim Demo → RGB + Depth + Pose → same Frame format   │
│   목적: future robotic-scene 확장성 시연, 메인 평가는 ScanNet │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   └── Dataset Adapter 이후 동일 파이프라인 사용
```

### Single-scene 정의

이 프로젝트에서 **단일 scene**은 이미지 한 장이 아니라, 같은 장소/공간을 여러 카메라 위치에서 촬영한 RGB-D frame sequence를 의미한다. 예를 들어 `scene0000_00`은 하나의 방 또는 실내 공간에 대한 scan이며, 그 안에 여러 RGB frame, depth frame, camera pose, intrinsics, mesh, object annotation이 함께 포함된다.

따라서 map 생성은 여러 dataset이나 여러 장소를 섞는 것이 아니라, 같은 scene 안의 frame들을 하나의 world 좌표계로 변환해 누적하는 방식으로 수행한다.

```
ScanNet scene0000_00
├── frame_0000: RGB + depth + T_cam2world
├── frame_0001: RGB + depth + T_cam2world
├── frame_0002: RGB + depth + T_cam2world
├── ...
└── same-scene object annotations
```

### Semantic Map 생성 방식: base vocabulary + open-vocabulary expansion

이 프로젝트의 메인 결과물은 dense TSDF/point-wise semantic map이 아니라 **object-level semantic map**이다. 즉, 모든 공간 픽셀에 class를 붙이는 것이 아니라, 실내 주요 객체별로 `label`, `centroid`, `observation_count`, `seen_frame_ids`를 저장한다.

메인 파이프라인은 사용자가 질문할 때마다 처음부터 탐지하는 방식이 아니라, 먼저 제한된 실내 객체 vocabulary를 사용해 ScanNet single-scene 전체 frame subset에서 base semantic map을 만든다.

```text
base vocabulary 예시:
chair . table . sofa . bed . cabinet . desk . lamp . vase . plant .
```

이후 사용자 질의는 두 단계로 처리한다.

1. **Map lookup**: 질의 label이 SemanticMap에 이미 있으면 저장된 centroid/count를 즉시 반환한다.
2. **Open-vocabulary expansion**: 질의 label이 map에 없으면 해당 label을 새 text prompt로 Grounding DINO + SAM을 실행하고, 3D centroid를 계산해 SemanticMap에 새 객체로 등록한 뒤 반환한다.

따라서 프로젝트 메시지는 다음과 같이 정의한다.

> 사전 정의한 실내 객체 vocabulary로 3D semantic object map을 만들고, open-vocabulary 질의를 통해 기존 map을 조회하거나 새로운 객체를 동적으로 추가할 수 있는 시스템.

---

## 환경 구성

### 메인 환경 + 선택적 Isaac Sim 환경

| 환경 | conda 이름 | 역할 | 핵심 패키지 |
|------|-----------|------|------------|
| 메인 파이프라인 | `cv` (Python 3.11 권장, 3.12 시도 가능) | 데이터셋 로딩 + 탐지 + 3D 맵핑 + 평가 | Grounding DINO, SAM, Open3D |
| Isaac Sim 환경 (선택) | `lab` (Python 3.11) | 보조 데모 데이터 생성 | Isaac Sim 5.1, PyTorch 2.7, CUDA 12.8 |

> **이유**: 메인 결과는 ScanNet 단일 scene에서 검증한다. 3D semantic map은 여러 장소를 섞는 것이 아니라 하나의 일관된 world 좌표계 안에서 RGB-D frame을 누적해야 하므로, primary 입력은 ScanNet의 한 scene으로 고정한다. Replica는 ScanNet 접근 또는 annotation 처리 실패 시 fallback으로 사용한다. Isaac Sim은 이번 구현의 critical path가 아니라 후속 확장 데모로만 남긴다.

### cv 환경 설치

```bash
conda activate cv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install groundingdino-py
pip install segment-anything
pip install open3d
pip install opencv-python numpy scipy matplotlib
```

> **환경 fallback**: Grounding DINO 또는 SAM 설치가 Python 3.12에서 막히면 `cv` 환경을 Python 3.11 또는 3.10으로 다시 생성한다. 모델 설치 성공이 Week 1의 우선 검증 항목이다.

---

## 프로젝트 디렉토리 구조

```
cv_proj/
├── PROJECT_PLAN.md              # 이 파일
├── data/
│   ├── raw/                     # ScanNet 단일 scene에서 변환한 공통 입력
│   │   ├── frame_0000_rgb.png
│   │   ├── frame_0000_depth.npy
│   │   ├── frame_0000_pose.json
│   │   └── ...
│   ├── scannet/                 # ScanNet primary scene 원본 또는 변환 중간 파일
│   ├── replica/                 # fallback scene 원본 또는 변환 중간 파일
│   ├── intrinsics.json          # 카메라 내부 파라미터 (K matrix)
│   └── gt_objects.json          # GT 객체 위치 (평가용: instance mesh vertices 평균)
├── datasets/
│   ├── scannet_adapter.py       # ScanNet → 공통 Frame 형식 변환
│   └── replica_adapter.py       # Replica → 공통 Frame 형식 변환
├── isaac_sim/
│   └── data_collector.py        # 선택적 Isaac Sim 데모 데이터 생성
├── src/
│   ├── detector.py              # Grounding DINO wrapper
│   ├── segmentor.py             # SAM wrapper
│   ├── projector.py             # depth unprojection + pose transform
│   ├── semantic_map.py          # Object3D + SemanticMap 관리
│   ├── query_engine.py          # 텍스트 질의 처리
│   ├── evaluator.py             # Recall + Precision + Localization Error 계산
│   └── visualizer.py           # Open3D 3D 시각화
├── models/
│   ├── groundingdino/           # 모델 가중치
│   └── sam/                     # 모델 가중치
├── outputs/
│   ├── maps/                    # 생성된 semantic map JSON
│   ├── figures/                 # 시각화 결과 이미지
│   └── metrics.json            # 평가 결과
├── notebooks/
│   └── demo.ipynb               # 시연용 노트북
├── main.py                      # 메인 실행 스크립트
├── query.py                     # 텍스트 질의 인터페이스
├── evaluate.py                  # 평가 실행 스크립트
└── requirements.txt
```

---

## 주차별 구현 계획

### Week 1: 환경 세팅 + ScanNet 단일 scene 로딩 파이프라인
**목표**: ScanNet의 한 scene에서 RGB-D sequence를 공통 Frame 형식으로 읽고 3D 투영 검증

#### 할 일
- [ ] `cv` conda 환경에 Grounding DINO, SAM, Open3D 설치 확인
- [ ] ScanNet 1개 scene을 primary map 공간으로 선정
  - 후보: `scene0000_00` 또는 실내 가구가 5개 이상 보이는 작은 room scene
  - frame subset: 같은 scene의 50-200 RGB-D frames
  - fallback: ScanNet 접근/annotation 처리 실패 시 Replica 1개 scene으로 전환
- [ ] `datasets/scannet_adapter.py` 작성
  - RGB, depth, camera pose, intrinsics를 공통 형식으로 변환
  - frame index 기준으로 파일 경로와 pose를 안정적으로 매칭
  - GT object annotation을 `gt_objects.json`으로 변환
  - GT centroid는 ScanNet instance annotation에 속한 mesh vertices의 XYZ 평균으로 계산
  - 평가 대상 category는 chair, table, sofa, cabinet, bed 등 5개 이하로 제한
- [ ] 저장된 데이터 형식 검증 (shape, dtype, depth scale, 좌표계 확인)
- [ ] `src/projector.py` 기초 구현 및 단일 프레임 테스트

#### Dataset Adapter 핵심 로직
```python
# datasets/base_adapter.py
import numpy as np
import json

class Frame:
    def __init__(self, rgb_path, depth_path, T_cam2world, intrinsics):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.T_cam2world = T_cam2world
        self.intrinsics = intrinsics

def export_frame(frame: Frame, out_dir: str, idx: int):
    depth = load_depth_in_meters(frame.depth_path)
    np.save(f"{out_dir}/frame_{idx:04d}_depth.npy", depth)
    save_rgb_copy(frame.rgb_path, f"{out_dir}/frame_{idx:04d}_rgb.png")
    json.dump(frame.T_cam2world.tolist(),
              open(f"{out_dir}/frame_{idx:04d}_pose.json", "w"))
```

**완료 기준**: ScanNet 1개 scene에서 최소 50프레임 이상의 RGB + depth + pose 로딩 및 단일 3D 투영 성공

---

### Week 2: 탐지 + 세그멘테이션 파이프라인
**목표**: Grounding DINO → SAM → 2D mask 생성까지 완성

#### 할 일
- [ ] `src/detector.py`: Grounding DINO 모델 로드 + 추론 함수 구현
  - 입력: RGB image, text prompt (예: `"chair . table . sofa . bed . cabinet . desk . lamp . vase . plant"`)
  - 출력: bounding boxes, labels, confidence scores
- [ ] `configs/vocabulary.json` 또는 equivalent 상수로 base indoor vocabulary 정의
  - MVP primary category는 ScanNet GT 평가 가능성을 고려해 5개 이하로 제한 가능
  - demo vocabulary는 chair, table, sofa, bed, cabinet, desk, lamp, vase, plant 등으로 확장 가능
- [ ] `src/segmentor.py`: SAM 모델 로드 + bbox → mask 변환
  - 입력: RGB image, bounding boxes
  - 출력: binary masks (H, W)
- [ ] 단일 프레임 파이프라인 테스트
  - 여러 텍스트 프롬프트 실험
  - 탐지 결과 시각화 (bbox + mask overlay)
- [ ] Confidence threshold 튜닝 (기본: 0.35)

#### 핵심 코드 구조
```python
# src/detector.py
class GroundingDINODetector:
    def detect(self, image: np.ndarray, text_prompt: str) -> list[Detection]:
        # returns: [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}]

# src/segmentor.py
class SAMSegmentor:
    def segment(self, image: np.ndarray, bboxes: list) -> list[np.ndarray]:
        # returns: list of binary masks (H, W)
```

**완료 기준**: 실내 장면 이미지에서 base vocabulary prompt로 3개 이상 category의 객체 마스크 생성 성공

---

### Week 3: 3D 투영 + Semantic Map 구성
**목표**: 2D mask → 3D centroid → SemanticMap 완성

#### 할 일
- [ ] `src/projector.py`: depth unprojection + pose transformation 구현
  - `mask_to_pointcloud(mask, depth, K)` → (N, 3) 3D points in camera frame
  - `transform_to_world(points, T_cam2world)` → (N, 3) in world frame
  - centroid 계산: `centroid = points.mean(axis=0)`
- [ ] `src/semantic_map.py`: Object3D + SemanticMap 자료구조 구현
  - `Object3D`: label, centroid, point_cloud, observation_count
  - `SemanticMap.add_object(obj2d, centroid)`: 중복 제거 포함
  - 중복 판단: `same_label AND distance(centroid_a, centroid_b) < 0.5m`
  - 같은 객체로 판단되면 새 객체를 만들지 않고 기존 `centroid`와 `observation_count`를 업데이트
- [ ] TODO: label normalization 개선
  - `"tv monitor"` → `tv_monitor`, `"couch"` → `sofa` 같은 alias를 canonical label로 통일
  - `"cabinet refrigerator"`처럼 여러 class가 섞인 Grounding DINO phrase는 skip 또는 분리 정책 정의
  - base vocabulary와 query-time expansion 결과가 같은 label 체계를 쓰도록 정리
- [ ] 멀티프레임 통합 테스트 (10-20프레임)
- [ ] TODO: label normalization 이후 10-20 frame map build로 duplicate/merge 품질 재확인
- [ ] `outputs/maps/`에 JSON으로 맵 저장

#### 3D 투영 핵심 로직
```python
def mask_to_pointcloud(mask, depth, K):
    fx, fy, cx, cy = K
    v, u = np.where(mask > 0)
    d = depth[v, u]
    valid = d > 0
    X = (u[valid] - cx) * d[valid] / fx
    Y = (v[valid] - cy) * d[valid] / fy
    Z = d[valid]
    return np.stack([X, Y, Z], axis=1)

def transform_to_world(points_cam, T_cam2world):
    ones = np.ones((len(points_cam), 1))
    pts_h = np.hstack([points_cam, ones])   # (N, 4)
    pts_world = (T_cam2world @ pts_h.T).T   # (N, 4)
    return pts_world[:, :3]
```

#### Object association 핵심 로직
같은 object는 여러 frame에서 반복 탐지될 수 있다. 이때 매번 새 객체로 등록하지 않고, 기존 SemanticMap의 object와 매칭한 뒤 누적 업데이트한다.

```python
def add_observation(new_label, new_centroid, semantic_map, threshold=0.5):
    candidates = [
        obj for obj in semantic_map.objects
        if obj.label == new_label
    ]

    for obj in candidates:
        dist = np.linalg.norm(obj.centroid - new_centroid)
        if dist < threshold:
            n = obj.observation_count
            obj.centroid = (obj.centroid * n + new_centroid) / (n + 1)
            obj.observation_count += 1
            obj.seen_frame_ids.append(current_frame_id)
            return obj.id

    return semantic_map.create_object(
        label=new_label,
        centroid=new_centroid,
        observation_count=1,
        seen_frame_ids=[current_frame_id],
    )
```

`observation_count`는 여러 frame에서 반복 관측된 객체의 신뢰도를 나타낸다. threshold가 너무 작으면 같은 객체가 여러 개로 쪼개지고, 너무 크면 가까운 객체 두 개가 하나로 합쳐질 수 있으므로 MVP에서는 `0.5m`로 시작해 평가 결과에 따라 고정된 실험 설정으로 조정한다.

**완료 기준**: 50프레임 처리 후 SemanticMap에 5개 이상 객체 등록, 중복 제거 동작 확인

---

### Week 4: 텍스트 질의 + 평가 + 시각화
**목표**: 질의 시스템 완성 + Object Recall / Precision / 3D Localization Error 계산

#### 할 일
- [ ] `src/query_engine.py`: 3가지 질의 유형 구현
  - `"chair 위치"` → 모든 chair의 centroid 반환
  - `"가장 가까운 chair"` → 쿼리 위치에서 최소 거리 객체
  - `"탐지된 chair 개수"` → count 반환
  - MVP에서는 LLM 없이 규칙 기반 parser로 `label`과 `query_type`을 추출
- [ ] query-time open-vocabulary expansion 구현
  - 질의 label이 SemanticMap에 있으면 map lookup만 수행
  - 질의 label이 SemanticMap에 없으면 해당 label prompt로 추가 탐지 실행
  - 추가 탐지 결과를 3D centroid로 변환하고 SemanticMap에 등록
  - expansion으로 추가된 객체는 `source="query_expansion"` 등으로 base map 객체와 구분 가능하게 저장
- [ ] `gt_objects.json` 작성: ScanNet primary scene의 instance/semantic annotation에서 평가 대상 객체 위치 추출
- [ ] `src/evaluator.py`: 평가 지표 계산
  - **Object Recall** = 탐지된 GT 객체 수 / 전체 GT 객체 수
  - **Object Precision** = matched prediction 수 / 전체 prediction 수
  - **3D Localization Error** = mean L2(pred_centroid, gt_centroid) [미터 단위]
  - **Duplicate Count/Rate** = 이미 matched 된 GT 주변에 중복 생성된 prediction 수
  - Hungarian matching으로 pred ↔ GT 최적 매칭
- [ ] `src/visualizer.py`: Open3D 3D 시각화
  - 카메라 궤적 표시
  - 객체 centroid 구 표시 (색상 = 카테고리)
  - GT vs Predicted 비교 시각화

#### 평가 계산 예시
```python
# 헝가리안 매칭으로 pred-GT 최적 페어링
from scipy.optimize import linear_sum_assignment

def compute_metrics(pred_objects, gt_objects, match_threshold=1.0):
    # distance matrix: pred x gt
    dist_matrix = cdist(pred_centroids, gt_centroids)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    matched = dist_matrix[row_ind, col_ind] < match_threshold
    recall = matched.sum() / len(gt_objects)
    precision = matched.sum() / len(pred_objects)
    loc_error = dist_matrix[row_ind, col_ind][matched].mean()
    return {"recall": recall, "precision": precision, "loc_error": loc_error}
```

#### Query Engine MVP 방식

MVP의 자연어 질의는 full LLM reasoning이 아니라, 제한된 자연어 문장에서 객체 label과 질의 유형을 추출하는 방식으로 구현한다. 우선 지원 label은 base map 생성 때 사용한 indoor vocabulary를 기본으로 하되, map에 없는 label이 들어오면 open-vocabulary expansion 후보로 취급한다.

```python
def parse_query(text, known_labels):
    text_lower = text.lower()
    label = next((l for l in known_labels if l.lower() in text_lower), None)
    if label is None:
        label = extract_unknown_label_candidate(text_lower)

    if "몇" in text or "개" in text or "count" in text_lower:
        return {"type": "count", "label": label}
    if "가까운" in text or "nearest" in text_lower:
        return {"type": "nearest", "label": label}
    if "어디" in text or "위치" in text or "where" in text_lower:
        return {"type": "location", "label": label}

    return {"type": "unknown", "label": label}
```

예상 질의:
- `"chair 어디 있어?"` → 모든 chair centroid 반환
- `"가장 가까운 chair는 어디야?"` → 기준 위치에서 가장 가까운 chair 반환
- `"table 몇 개야?"` → SemanticMap에 등록된 table 개수 반환
- `"microwave 어디 있어?"` → map에 없으면 `"microwave ."` prompt로 추가 탐지 후 map에 등록하고 centroid 반환

색상/재질/형상까지 포함한 attribute query 예: `"빨갛고 울퉁불퉁한 항아리"`는 MVP 범위 밖이며, 후속 확장에서는 object crop 저장 + CLIP embedding 또는 VLM caption을 추가해 지원한다.

**완료 기준**: Object Recall ≥ 0.6, Object Precision ≥ 0.5, 3D Localization Error ≤ 1.0m 달성

---

## 테스트 및 검증 계획

| 단계 | 대상 | 검증 내용 | 완료 기준 |
|------|------|----------|----------|
| Unit | `src/projector.py` | synthetic depth/K/pose로 예상 3D 좌표 검증 | known point 오차 < 1e-5 |
| Unit | `src/semantic_map.py` | same label + 0.5m 이내 object merge, 다른 label/object 분리 | merge/split case 통과 |
| Unit | `src/evaluator.py` | 작은 pred/GT fixture로 recall, precision, loc_error 검증 | 계산값이 hand-labeled expected 값과 일치 |
| Integration | dataset adapter | ScanNet 1 scene에서 RGB/depth/pose/intrinsics shape, dtype, scale 확인 | 5 frame 연속 로딩 성공 |
| Smoke | `main.py` | 5-frame subset end-to-end 실행 | `outputs/maps/scene_map.json` 생성 |
| Full run | `main.py` + `evaluate.py` | 50-frame subset map 생성 및 평가 | `outputs/metrics.json` 생성 |

---

### Week 5: 통합 + 데모 제작 + 보고서
**목표**: 전체 파이프라인 통합, 시연 영상 제작, 최종 정리

#### 할 일
- [ ] `main.py` 통합 실행 스크립트 완성
  ```bash
  python main.py --scene data/raw/ --prompts "chair . table . sofa . TV"
  ```
- [ ] `query.py` 대화형 질의 인터페이스
  ```bash
  python query.py --map outputs/maps/scene_map.json
  # > "가장 가까운 chair는 어디에 있나요?"
  ```
- [ ] `notebooks/demo.ipynb` 작성
  - 단계별 시각화 포함 (bbox → mask → 3D → map → query)
- [ ] **시연 영상 제작**
  - ScanNet 단일 scene + 탐지 결과 overlay
  - 3D map 시각화 (Open3D 스크린샷 또는 영상)
  - 텍스트 질의 응답 데모
- [ ] `outputs/metrics.json` 최종 결과 저장
- [ ] 발표 자료 / 보고서 작성

**완료 기준**: 전체 파이프라인 end-to-end 실행 성공, 시연 영상 완성

---

## 수용 기준 (Acceptance Criteria)

| # | 기준 | 검증 방법 | 목표 임계값 |
|---|------|----------|------------|
| AC1 | Grounding DINO 텍스트 탐지 | 단일 프레임에서 bbox 생성 | confidence ≥ 0.35 |
| AC2 | SAM 마스크 생성 | bbox → binary mask | mask area > 100px |
| AC3 | 단일 프레임 3D 투영 | centroid 좌표 출력 | Z > 0 (카메라 앞) |
| AC4 | 멀티프레임 맵 통합 | 50프레임 처리 | 오류 없이 완료 |
| AC5 | 중복 제거 동작 | 동일 장면 2회 스캔 | 등록 객체 수 변화 없음 |
| AC6 | 텍스트 질의 3종 | query.py 실행 | 정확한 응답 반환 |
| AC7 | **Object Recall** | evaluator.py | **≥ 0.6 (60%)** |
| AC8 | **Object Precision** | evaluator.py | **≥ 0.5 (50%)** |
| AC9 | **3D Localization Error** | evaluator.py | **≤ 1.0m** |
| AC10 | 3D 시각화 | Open3D 뷰어 / 스크린샷 | 모든 객체 표시 |
| AC11 | 시연 영상 | 파이프라인 end-to-end | 영상 파일 완성 |
| AC12 | Open-vocabulary map expansion | map에 없는 label 질의 | 추가 탐지 후 SemanticMap에 신규 객체 등록 또는 "not found" 반환 |

---

## 위험 요소 및 대응 방안

| 위험 | 확률 | 영향 | 대응 방안 |
|------|------|------|----------|
| ScanNet 데이터셋 다운로드 및 포맷 파악 지연 | 중 | 높음 | Week 1에서 primary scene 1개만 먼저 확보, adapter를 최소 형식부터 구현. 실패 시 Replica 1개 scene으로 fallback |
| ScanNet annotation에서 GT object centroid 추출 난이도 | 중 | 중 | 평가 대상 카테고리를 5개 이하로 제한하고 instance mesh vertices 평균으로 centroid 계산 |
| Grounding DINO 탐지율 저조 | 중 | 중 | confidence threshold 낮추기, 프롬프트 수정 |
| Grounding DINO label phrase ambiguity | 중 | 중 | canonical label normalization 추가, mixed phrase는 skip 또는 분리 정책 적용 |
| base vocabulary 누락으로 질의 객체 미등록 | 중 | 중 | 실내 주요 객체 vocabulary를 문서화하고, map에 없는 label은 query-time expansion으로 추가 탐지 |
| 10-20 frame 확장 시 duplicate 증가 | 중 | 중 | label normalization 후 association threshold와 observation_count 기준 재조정 |
| depth scale mismatch | 중 | 높음 | Week 1 검증 단계에서 meter 단위 확인, 스케일 팩터 적용 |
| 3D localization error 과대 | 중 | 중 | primary metric은 mean L2로 고정. median L2는 보조 분석으로만 보고하고, outlier 제거는 사전에 정의한 invalid-depth/object-size 필터에만 적용 |
| 개발 시간 초과 | 낮 | 중 | Week 4 이후 기능 축소 (질의 2종으로 줄이기) |

---

## 평가 지표 상세

### Object Recall
```
Recall = |탐지된 GT 객체| / |전체 GT 객체|
```
- GT 객체 정의: ScanNet primary scene의 instance annotation에 포함된 실내 가구 객체 (5개 이하 primary category)
- GT centroid: 각 GT instance에 속한 mesh vertices의 XYZ 평균
- "탐지됨" 기준: pred centroid가 GT centroid에서 1.0m 이내

### Object Precision
```
Precision = |matched predictions| / |all predicted objects|
```
- pred 객체는 label이 GT category에 포함된 Object3D만 평가에 포함
- Hungarian matching 후 1.0m 이내 matched 된 prediction만 true positive로 계산
- unmatched prediction은 false positive로 계산

### 3D Localization Error
```
Loc_Error = mean(||pred_centroid_i - gt_centroid_i||_2)
```
- Hungarian matching으로 pred ↔ GT 최적 매칭 후 계산
- matched pair만 포함 (unmatched는 recall에서 페널티)
- primary metric은 mean L2로 고정한다. median L2는 결과 해석용 보조 통계로만 보고한다.

### Duplicate Rate
```
Duplicate_Rate = |duplicate predictions| / |all predicted objects|
```
- 같은 label의 여러 prediction이 하나의 GT 주변 1.0m 이내에 몰린 경우, 가장 가까운 하나만 match로 인정하고 나머지는 duplicate으로 기록한다.
- Duplicate Rate는 필수 pass/fail 지표는 아니지만 object association 품질을 설명하는 보조 지표로 보고한다.

---

## 주요 참고자료

| 자료 | URL/경로 | 용도 |
|------|---------|------|
| Grounding DINO | https://github.com/IDEA-Research/GroundingDINO | 탐지 모델 |
| SAM | https://github.com/facebookresearch/segment-anything | 세그멘테이션 |
| ScanNet | http://www.scan-net.org | Primary single-scene RGB-D + pose + annotation 데이터셋 |
| Replica | https://github.com/facebookresearch/Replica-Dataset | ScanNet 실패 시 fallback single-scene RGB-D + pose + semantic 데이터셋 |
| Isaac Sim Python API | Isaac Sim 공식 문서 | 후속 로봇 환경 확장 데모 |
| Open3D | http://www.open3d.org | 3D 시각화 |
| Grounded-SAM | https://github.com/IDEA-Research/Grounded-Segment-Anything | 통합 파이프라인 참고 |

---

## ADR (Architecture Decision Record)

### ADR-001: ScanNet 단일 scene vs Replica/Isaac Sim

- **결정**: ScanNet의 한 scene을 primary source로 사용한다. 3D semantic object map은 하나의 일관된 world 좌표계 안에서 RGB-D frame을 누적해야 하므로, 여러 dataset/scene을 섞지 않고 단일 scene을 기준으로 평가한다. Replica는 ScanNet 접근 또는 annotation 처리 실패 시 fallback으로 둔다. Isaac Sim은 이번 프로젝트의 critical path에서 제외하고 future extension으로 유지한다.
- **이유**: ScanNet은 실제/표준 indoor RGB-D sequence, camera pose, instance-level semantic annotation을 제공하므로 정량 평가와 포트폴리오 설득력이 높다. Replica는 비교적 다루기 쉬운 fallback이지만 실제 benchmark 설득력은 ScanNet이 더 강하다. Isaac Sim 자체 생성 데이터는 로봇 확장 데모로는 좋지만 이번 목표인 공개 데이터 기반 검증에는 불필요하다.
- **대안**: Replica primary (구현 난이도는 낮지만 benchmark 설득력 약함), Isaac Sim primary (통제는 쉽지만 공개 데이터셋 검증 메시지 약함), 여러 scene 통합 (좌표계 일관성이 깨져 object map 목적과 불일치)
- **결과**: 메인 구현은 ScanNet single-scene dataset adapter + 공통 Frame 형식으로 시작한다. `data/raw/`는 하나의 scene에서 나온 frame만 포함하며, `gt_objects.json`도 같은 scene의 object annotation만 포함한다.

### ADR-002: Object-level Map vs Dense TSDF Map

- **결정**: centroid 기반 Object-level Map
- **이유**: 구현 단순, 텍스트 질의에 직접적, 평가 명확, 5주 일정에 적합
- **대안**: TSDF Fusion (고품질이나 구현 복잡도 과도), NeRF (텀 프로젝트 불가)
- **결과**: 평가 지표(Recall, Precision, Localization Error, Duplicate Rate) 직접 계산 가능

### ADR-003: 배치 처리 vs 실시간 처리

- **결정**: 배치 처리
- **이유**: 실시간은 Non-goal, 개발 속도 우선, 성능 안정적
- **대안**: 실시간 스트리밍 (구현 복잡도 3배 증가)
- **결과**: ScanNet 단일 scene의 RGB-D sequence를 먼저 배치 처리하고, 실시간 확장은 keyframe / tracking / async inference 후속 과제로 명시

### ADR-004: Fixed-class map vs query-time open-vocabulary detection

- **결정**: base indoor vocabulary로 object-level semantic map을 먼저 만들고, map에 없는 사용자 질의는 open-vocabulary expansion으로 추가 탐지한다.
- **이유**: fixed vocabulary map은 ScanNet GT로 정량 평가가 가능하고 `SemanticMap` 산출물이 남는다. query-time open-vocabulary detection은 데모 유연성이 높지만 단독으로는 "map 생성"보다 "3D detector"에 가까워진다. 두 방식을 결합하면 평가 가능성과 language-grounded 확장성을 동시에 확보할 수 있다.
- **대안**: 정해진 class만 저장 (평가는 쉽지만 open-vocabulary 가치 약함), 매 질의마다 전체 frame 재탐지 (유연하지만 느리고 평가 기준이 약함), dense semantic map (범위 과도)
- **결과**: main pipeline은 base vocabulary map building이며, query engine은 map lookup을 우선하고 label miss 시 추가 탐지/등록을 수행한다.

---

*계획서 생성: 2026-04-24 | Deep Interview 6라운드 (최종 모호성 17.7%) + Ralplan 합의 워크플로우 완료*
