# 실험 기록

## 프로젝트 설정

- 프로젝트: Language-Grounded 3D Object Mapping
- 데이터셋: ARKitScenes 3DOD
- 선택 scene: `41098076`
- scene 경로: `data/arkitscenes/3dod/Training/41098076`
- 모델 구성: Grounding DINO Swin-T OGC + SAM ViT-B
- 기본 vocabulary:
  - `chair`
  - `table`
  - `sofa`
  - `cabinet`
  - `tv monitor`
  - `refrigerator`
  - `sink`
  - `stove`

## 데이터 요약

- RGB frame: 796장
- Depth frame: 796장
- Intrinsics file: 796개
- Camera trajectory: `lowres_wide.traj`
- GT 객체: 30개

GT label 개수:

```text
cabinet 14
table 4
chair 4
sofa 2
oven 1
refrigerator 1
washer 1
sink 1
tv_monitor 1
stove 1
```

## 현재 구현된 파이프라인

현재 흐름:

```text
ARKitScenes RGB-D frame
→ Grounding DINO text detection
→ SAM bbox-to-mask
→ mask + depth + intrinsics unprojection
→ T_cam_to_world transform
→ world-frame object centroid
→ SemanticMap object association
→ GT centroid evaluation
```

객체 병합 규칙:

```text
canonical label이 같고 centroid distance <= 0.6m이면 같은 객체로 병합
```

평가 매칭 규칙:

```text
label이 같고 prediction-GT centroid distance <= 1.0m이면 match로 처리
```

IoU 대신 centroid 기반 평가를 사용한 이유:

- 이 프로젝트의 목표는 dense 3D instance segmentation이 아니라 language query에 대해 object-level 3D 위치를 반환하는 것이다.
- 따라서 “객체 mask/box가 GT 영역과 얼마나 정확히 겹치는가”보다 “객체 중심 좌표가 실제 객체 위치 근처에 있는가”가 더 직접적인 평가 기준이다.
- SAM mask 품질은 3D centroid 계산에 영향을 주므로 중요하지만, 최종 산출물은 pixel-level mask가 아니라 `label + centroid_m` 형태의 semantic map object다.
- 3D IoU를 계산하려면 object별 누적 point cloud, 3D bounding box fitting, GT box와의 overlap 계산이 추가로 필요하다.
- 따라서 현재 단계에서는 `precision`, `recall`, `3D localization error`, `duplicate_rate`를 centroid-based matching으로 평가하고, IoU 기반 평가는 future work로 둔다.

## 평가 용어 해석

- `GT`: ARKitScenes annotation JSON에 들어있는 정답 객체 수.
- `predictions`: 우리 semantic map이 최종적으로 예측한 객체 수.
- `matches`: prediction 중에서 GT와 label이 같고, 3D centroid 거리가 1m 이내인 객체 수.
- `precision`: 우리가 찾았다고 한 객체 중 실제로 맞은 비율.
- `recall`: 실제 GT 객체 중 우리가 찾아낸 비율.
- `mean L2 error`: match된 객체들의 평균 3D 위치 오차.
- `median L2 error`: match된 객체들의 중앙값 3D 위치 오차.
- `duplicate_rate`: prediction 중 GT에 매칭되지 못한 객체 비율. 중복 객체, 위치 오류, 오탐이 여기에 포함된다.

예를 들어 다음 결과는:

```text
predictions: 16
GT: 30
matches: 14
precision: 0.8750
recall: 0.4667
mean L2 error: 0.2817m
median L2 error: 0.2697m
duplicate_rate: 0.1250
```

다음처럼 해석한다:

- 실제 정답 객체는 30개다.
- 우리 semantic map은 그중 객체 16개를 예측했다.
- 예측한 16개 중 14개는 label과 위치가 모두 맞아서 정답 처리되었다.
- 나머지 2개는 GT와 매칭되지 못했다.
- 따라서 precision은 높지만, recall은 낮다.

즉 이 설정은 찾은 객체는 대부분 맞지만, 실제 scene 안의 객체를 아직 많이 놓치는 conservative한 결과다.

## Threshold 실험

### 20프레임, text_threshold=0.25

- Frame index: `0,40,80,120,160,200,240,280,320,360,400,440,480,520,560,600,640,680,720,760`
- Map: `outputs/maps/41098076_semantic_map_20frames.json`
- `box_threshold`: 0.25
- `text_threshold`: 0.25
- Object count: 40

Label count:

```text
tv_monitor 9
cabinet 9
chair 7
table 6
refrigerator 4
stove 3
sink 2
```

해석:

- Recall을 더 많이 확보하는 방향이지만 noise가 많다.
- 중복 객체와 애매한 phrase가 semantic map에 더 많이 들어온다.

### 20프레임, text_threshold=0.35

- Frame index: 위와 동일
- Map: `outputs/maps/41098076_semantic_map_20frames_text035.json`
- `box_threshold`: 0.25
- `text_threshold`: 0.35
- Object count: 25

Label count:

```text
chair 7
cabinet 6
table 4
tv_monitor 4
refrigerator 2
stove 1
sink 1
```

해석:

- `text_threshold=0.25`보다 map이 더 깔끔하다.
- 약하거나 섞인 text phrase 일부가 제거된다.
- 현재 기본 threshold는 `box_threshold=0.25`, `text_threshold=0.35`로 둔다.

## GT 평가 결과

### 20프레임, text_threshold=0.35, observation_count >= 1

- Metrics: `outputs/metrics_41098076_text035_minobs1.json`

```text
predictions: 25
GT: 30
matches: 13
precision: 0.5200
recall: 0.4333
mean L2 error: 0.3032m
median L2 error: 0.2305m
duplicate_rate: 0.4800
```

### 20프레임, text_threshold=0.35, observation_count >= 2

- Metrics: `outputs/metrics_41098076_text035_minobs2.json`

```text
predictions: 11
GT: 30
matches: 9
precision: 0.8182
recall: 0.3000
mean L2 error: 0.3266m
median L2 error: 0.2305m
duplicate_rate: 0.1818
```

해석:

- 반복 관측 조건을 걸면 precision은 크게 좋아진다.
- 하지만 20프레임만으로는 viewpoint coverage가 부족해서 recall이 낮다.

### 50프레임, text_threshold=0.35, observation_count >= 1

- Map: `outputs/maps/41098076_semantic_map_50frames_text035.json`
- Metrics: `outputs/metrics_41098076_50frames_text035_minobs1.json`

```text
predictions: 41
GT: 30
matches: 20
precision: 0.4878
recall: 0.6667
mean L2 error: 0.2852m
median L2 error: 0.2697m
duplicate_rate: 0.5122
```

### 50프레임, text_threshold=0.35, observation_count >= 2

- Metrics: `outputs/metrics_41098076_50frames_text035_minobs2.json`

```text
predictions: 22
GT: 30
matches: 16
precision: 0.7273
recall: 0.5333
mean L2 error: 0.3059m
median L2 error: 0.2838m
duplicate_rate: 0.2727
```

### 50프레임, text_threshold=0.35, observation_count >= 3

- Metrics: `outputs/metrics_41098076_50frames_text035_minobs3.json`

```text
predictions: 16
GT: 30
matches: 14
precision: 0.8750
recall: 0.4667
mean L2 error: 0.2817m
median L2 error: 0.2697m
duplicate_rate: 0.1250
```

해석:

- 20프레임보다 50프레임에서 recall이 크게 좋아졌다.
- `observation_count >= 2`는 precision과 recall의 균형이 가장 좋다.
- `observation_count >= 3`은 precision은 가장 높지만 recall을 많이 잃는다.
- match된 객체들의 위치 오차는 평균 약 28-31cm 수준이다.

### 100프레임, text_threshold=0.35, observation_count >= 1

- Map: `outputs/maps/41098076_semantic_map_100frames_text035.json`
- Metrics: `outputs/metrics_41098076_100frames_text035_minobs1.json`

```text
predictions: 55
GT: 30
matches: 20
precision: 0.3636
recall: 0.6667
mean L2 error: 0.3224m
median L2 error: 0.3128m
duplicate_rate: 0.6364
```

### 100프레임, text_threshold=0.35, observation_count >= 2

- Metrics: `outputs/metrics_41098076_100frames_text035_minobs2.json`

```text
predictions: 33
GT: 30
matches: 19
precision: 0.5758
recall: 0.6333
mean L2 error: 0.3113m
median L2 error: 0.2982m
duplicate_rate: 0.4242
```

### 100프레임, text_threshold=0.35, observation_count >= 3

- Metrics: `outputs/metrics_41098076_100frames_text035_minobs3.json`

```text
predictions: 27
GT: 30
matches: 19
precision: 0.7037
recall: 0.6333
mean L2 error: 0.3113m
median L2 error: 0.2982m
duplicate_rate: 0.2963
```

해석:

- 100프레임은 50프레임보다 더 많은 viewpoint를 사용하므로 recall이 올라간다.
- 하지만 현재 association rule이 단순한 centroid 거리 기반이라, 프레임을 많이 넣을수록 중복 prediction도 같이 증가한다.
- `observation_count >= 1`은 recall은 높지만 duplicate가 너무 많다.
- `observation_count >= 3`은 100프레임 결과 중 가장 균형이 좋다.
- 50프레임 `observation_count >= 2`와 비교하면 recall은 `0.5333 → 0.6333`으로 좋아졌지만, precision은 `0.7273 → 0.7037`로 약간 낮아졌다.

### Pose keyframe 224프레임, text_threshold=0.35

- Keyframe 기준: camera translation이 직전 선택 keyframe 대비 `0.10m` 이상일 때 선택
- 선택 frame 수: 224장
- Map: `outputs/maps/41098076_semantic_map_keyframes_t010_text035.json`
- `box_threshold`: 0.25
- `text_threshold`: 0.35
- Object count: 69

`observation_count >= 1`:

- Metrics: `outputs/metrics_41098076_keyframes_t010_text035_minobs1.json`

```text
predictions: 69
GT: 30
matches: 22
precision: 0.3188
recall: 0.7333
mean L2 error: 0.3881m
median L2 error: 0.3191m
duplicate_rate: 0.6812
```

`observation_count >= 3`:

- Metrics: `outputs/metrics_41098076_keyframes_t010_text035_minobs3.json`

```text
predictions: 45
GT: 30
matches: 21
precision: 0.4667
recall: 0.7000
mean L2 error: 0.3376m
median L2 error: 0.3060m
duplicate_rate: 0.5333
```

`observation_count >= 6`:

- Metrics: `outputs/metrics_41098076_keyframes_t010_text035_minobs6.json`

```text
predictions: 28
GT: 30
matches: 19
precision: 0.6786
recall: 0.6333
mean L2 error: 0.3144m
median L2 error: 0.3130m
duplicate_rate: 0.3214
```

`observation_count >= 8`:

- Metrics: `outputs/metrics_41098076_keyframes_t010_text035_minobs8.json`

```text
predictions: 24
GT: 30
matches: 18
precision: 0.7500
recall: 0.6000
mean L2 error: 0.3134m
median L2 error: 0.3120m
duplicate_rate: 0.2500
```

`observation_count >= 10`:

- Metrics: `outputs/metrics_41098076_keyframes_t010_text035_minobs10.json`

```text
predictions: 21
GT: 30
matches: 18
precision: 0.8571
recall: 0.6000
mean L2 error: 0.3134m
median L2 error: 0.3120m
duplicate_rate: 0.1429
```

해석:

- 224 keyframe은 raw 기준으로 가장 높은 recall을 만든다. `observation_count >= 1`에서 recall은 `0.7333`이다.
- 하지만 중복 prediction이 크게 늘어 duplicate rate도 `0.6812`까지 증가한다.
- 현재 association rule만으로는 많은 frame을 넣을수록 같은 객체가 여러 object로 쪼개지는 문제가 커진다.
- `observation_count >= 10`을 적용하면 precision `0.8571`, recall `0.6000`으로 매우 깔끔한 map이 된다.
- 다만 100프레임 `observation_count >= 3`과 비교하면 recall은 `0.6333 → 0.6000`으로 약간 낮다.

## 현재 권장 보고 결과

현재까지는 세 가지 결과를 목적에 따라 같이 보고하는 것이 좋다.

### 균형형 결과

50프레임:

```text
box_threshold = 0.25
text_threshold = 0.35
observation_count >= 2

precision: 0.7273
recall: 0.5333
mean L2 error: 0.3059m
median L2 error: 0.2838m
duplicate_rate: 0.2727
```

### Recall 개선 결과

100프레임:

```text
box_threshold = 0.25
text_threshold = 0.35
observation_count >= 3

precision: 0.7037
recall: 0.6333
mean L2 error: 0.3113m
median L2 error: 0.2982m
duplicate_rate: 0.2963
```

현재 프로젝트 관점에서는 100프레임 `observation_count >= 3` 결과가 더 설득력 있다. Object Recall >= 0.6 조건을 넘기면서도 precision이 0.7 이상으로 유지되기 때문이다.

### 가장 깔끔한 keyframe 결과

Pose keyframe 224프레임:

```text
keyframe rule = camera translation >= 0.10m
box_threshold = 0.25
text_threshold = 0.35
observation_count >= 10

precision: 0.8571
recall: 0.6000
mean L2 error: 0.3134m
median L2 error: 0.3120m
duplicate_rate: 0.1429
```

이 결과는 map이 가장 깔끔하다. 다만 recall까지 고려한 대표 결과로는 100프레임 `observation_count >= 3`이 아직 더 균형적이다.

## 현재 한계와 TODO

- Label phrase ambiguity가 아직 남아 있다.
- 높은 `text_threshold`에서 일부 detection은 empty phrase가 되어 skip된다.
- Semantic map duplicate rate가 association threshold에 민감하다.
- 100프레임과 224 keyframe에서 recall은 좋아졌지만 duplicate도 늘었으므로, object association 개선이 필요하다.
- Query engine은 아직 구현되지 않았다.
- 평가는 현재 3D IoU가 아니라 object centroid matching 기준이다.

## 다음 실험

다음으로 개선할 부분:

```text
1. association rule 개선
   - label + centroid distance만 쓰지 말고 observation_count, mask size, viewpoint consistency 등을 반영

2. query engine 구현
   - semantic map에서 label/count/location/nearest 질의 처리

3. visualization 구현
   - predicted centroid와 GT centroid를 같은 3D view에서 확인
```

## 관련 연구 성능 참고

우리 평가는 논문에서 흔히 쓰는 AP/mAP와 직접 비교할 수 없다. 논문 AP는 보통 3D mask 또는 3D bounding box가 GT와 얼마나 겹치는지를 IoU로 평가한다. 반면 현재 우리 평가는 object centroid가 GT centroid와 `1.0m` 이내이면 match로 처리한다.

따라서 우리 `precision`, `recall`은 **centroid-based object localization** 지표이고, 아래 AP 값들은 더 엄격한 3D instance segmentation/detection 지표다.

용어:

- `AP`: Average Precision. confidence threshold를 여러 단계로 바꾸며 precision-recall curve를 만들고, 그 전체 성능을 하나로 요약한 값이다.
- `AP50`: 예측 3D mask/box와 GT의 IoU가 `0.50` 이상일 때만 정답으로 인정하는 AP다.
- `AP25`: 예측 3D mask/box와 GT의 IoU가 `0.25` 이상이면 정답으로 인정하는 AP다.
- 일반적으로 `AP25 > AP50 > AP` 순서로 값이 높게 나온다. `AP25`가 가장 느슨하고, `AP50`은 더 엄격하다.

### SOTA / 강한 최신 방법

```text
Open3DIS, CVPR 2024, ScanNet200 OV-3D instance segmentation
AP:   23.7%
AP50: 29.4%
AP25: 32.8%

Open-YOLO 3D, 2024, ScanNet200 OV-3D instance segmentation
mAP: 24.7%

OpenM3D, ICCV 2025, ScanNet200 OV-3D object detection
mAP@0.25: 26.92%
mAR@0.25: 51.19%

Details Matter / OpenTrack 계열, 2025, ScanNet200 OV-3D instance segmentation
AP:   약 35%
AP50: 약 56%
AP25: 약 70%
```

### 우리와 더 유사한 2D foundation model + RGB-D fusion 계열

```text
OVIR-3D + Grounded-SAM, ScanNet200
AP:   13.0%
AP50: 24.9%
AP25: 32.3%

Open3DIS only 2D / G-SAM, ScanNet200
AP:   18.2%
AP50: 26.1%
AP25: 31.4%

Open3DIS SAM, 100 frames per scene
AP:   18.5%
AP50: 33.5%
AP25: 44.3%
AR50: 63.7%
```

### 우리 대표 결과

```text
100 frames, observation_count >= 3
평가 방식: centroid-based matching within 1.0m

precision: 70.37%
recall:    63.33%
mean L2 localization error: 31.13cm
duplicate_rate: 29.63%
```

해석:

- 논문 AP는 3D mask/box overlap, confidence ranking, duplicate false positive까지 평가하므로 더 어렵고 값이 낮게 나온다.
- 우리 평가는 centroid 위치가 근처인지 보는 방식이라 더 느슨하다.
- 따라서 우리 결과는 SOTA AP와 직접 비교하면 안 된다.
- 대신 “off-the-shelf Grounding DINO + SAM + RGB-D projection + simple centroid association으로 object-level localization을 수행한 prototype” 관점에서는 충분히 괜찮은 결과다.

보고서 표현 예시:

```text
Our method achieves 70.37% precision and 63.33% recall under centroid-based object localization within a 1m matching threshold. This metric is less strict than AP-based 3D instance segmentation metrics, which evaluate 3D mask/box overlap and confidence ranking.
```
