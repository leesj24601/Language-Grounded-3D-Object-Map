# GT-aligned vocabulary 실험 기록

## 프로젝트 설정

- 프로젝트: Language-Grounded 3D Object Mapping
- 데이터셋: ARKitScenes 3DOD
- 선택 scene: `41098076`
- scene 경로: `data/arkitscenes/3dod/Training/41098076`
- 모델 구성: Grounding DINO Swin-T OGC + SAM ViT-B
- 출력 경로: `outputs/gt_aligned_10_label/`

## Vocabulary

이번 실험은 GT label set과 Grounding DINO prompt label set을 맞춘다.

GT label set:

```text
cabinet
chair
oven
refrigerator
sink
sofa
stove
table
tv_monitor
washer
```

Grounding DINO prompt:

```text
cabinet . chair . table . sofa . oven . refrigerator . washer . sink . tv monitor . stove .
```

`tv monitor`는 detection prompt phrase이고, semantic map 내부에서는 `tv_monitor`로 정규화한다.

## 실험 계획

1차 실험은 대표 설정으로 `text_threshold=0.35`를 유지한다.

```text
20 frames, min_observations = 1, 2
50 frames, min_observations = 1, 2, 3
100 frames, min_observations = 1, 2, 3
```

결과가 애매하면 pose keyframe 설정을 추가한다.

```text
pose keyframe 224 frames, min_observations = 3, 6, 8, 10
```

## 결과

### 20프레임, text_threshold=0.35

- Frame index: `0,40,80,120,160,200,240,280,320,360,400,440,480,520,560,600,640,680,720,760`
- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_20frames_text035.json`
- `box_threshold`: 0.25
- `text_threshold`: 0.35
- Object count: 28

Label count:

```text
cabinet 7
chair 7
oven 1
refrigerator 3
sink 1
stove 1
table 4
tv_monitor 3
washer 1
```

`observation_count >= 1`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_20frames_text035_minobs1.json`

```text
predictions: 28
GT: 30
matches: 15
precision: 0.5357
recall: 0.5000
mean L2 error: 0.3251m
median L2 error: 0.2952m
duplicate_rate: 0.4643
```

`observation_count >= 2`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_20frames_text035_minobs2.json`

```text
predictions: 11
GT: 30
matches: 9
precision: 0.8182
recall: 0.3000
mean L2 error: 0.3297m
median L2 error: 0.2305m
duplicate_rate: 0.1818
```

해석:

- GT-aligned prompt는 20프레임에서도 `oven`, `washer`를 map에 포함한다.
- minobs 1은 matches 15, recall 0.5000으로 20프레임 설정 중 가장 넓게 객체를 회수한다.
- minobs 2는 matches 9, recall 0.3000이라 반복 관측 기준에서는 20프레임 coverage가 아직 부족하다.

### 50프레임, text_threshold=0.35

- Frame index: `0,16,32,49,65,81,97,114,130,146,162,178,195,211,227,243,260,276,292,308,324,341,357,373,389,406,422,438,454,471,487,503,519,535,552,568,584,600,617,633,649,665,681,698,714,730,746,763,779,795`
- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_50frames_text035.json`
- `box_threshold`: 0.25
- `text_threshold`: 0.35
- Object count: 45

Label count:

```text
cabinet 11
chair 6
oven 1
refrigerator 4
sink 2
sofa 4
stove 1
table 5
tv_monitor 5
washer 6
```

`observation_count >= 1`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_50frames_text035_minobs1.json`

```text
predictions: 45
GT: 30
matches: 22
precision: 0.4889
recall: 0.7333
mean L2 error: 0.3072m
median L2 error: 0.2840m
duplicate_rate: 0.5111
```

`observation_count >= 2`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_50frames_text035_minobs2.json`

```text
predictions: 25
GT: 30
matches: 18
precision: 0.7200
recall: 0.6000
mean L2 error: 0.3305m
median L2 error: 0.3043m
duplicate_rate: 0.2800
```

`observation_count >= 3`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_50frames_text035_minobs3.json`

```text
predictions: 17
GT: 30
matches: 15
precision: 0.8824
recall: 0.5000
mean L2 error: 0.3088m
median L2 error: 0.2823m
duplicate_rate: 0.1176
```

해석:

- 50프레임 minobs 1은 recall 0.7333으로 높지만 duplicate_rate 0.5111이라 noise가 많다.
- minobs 2는 precision 0.7200, recall 0.6000으로 precision/recall 균형이 좋다.
- minobs 3은 precision 0.8824로 가장 깔끔하지만 recall 0.5000이라 대표 결과로는 보수적이다.

### 100프레임, text_threshold=0.35

- Frame index: `0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,137,145,153,161,169,177,185,193,201,209,217,225,233,241,249,257,265,273,281,289,297,305,313,321,329,337,345,353,361,369,377,385,393,402,410,418,426,434,442,450,458,466,474,482,490,498,506,514,522,530,538,546,554,562,570,578,586,594,602,610,618,626,634,642,650,658,667,675,683,691,699,707,715,723,731,739,747,755,763,771,779,787,795`
- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_100frames_text035.json`
- `box_threshold`: 0.25
- `text_threshold`: 0.35
- Object count: 56

Label count:

```text
cabinet 12
chair 6
oven 1
refrigerator 8
sink 2
sofa 7
stove 1
table 7
tv_monitor 6
washer 6
```

`observation_count >= 1`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_100frames_text035_minobs1.json`

```text
predictions: 56
GT: 30
matches: 21
precision: 0.3750
recall: 0.7000
mean L2 error: 0.3477m
median L2 error: 0.3273m
duplicate_rate: 0.6250
```

`observation_count >= 2`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_100frames_text035_minobs2.json`

```text
predictions: 35
GT: 30
matches: 21
precision: 0.6000
recall: 0.7000
mean L2 error: 0.3111m
median L2 error: 0.3251m
duplicate_rate: 0.4000
```

`observation_count >= 3`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_100frames_text035_minobs3.json`

```text
predictions: 29
GT: 30
matches: 21
precision: 0.7241
recall: 0.7000
mean L2 error: 0.3111m
median L2 error: 0.3251m
duplicate_rate: 0.2759
```

`observation_count >= 4`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_100frames_text035_minobs4.json`

```text
predictions: 25
GT: 30
matches: 21
precision: 0.8400
recall: 0.7000
mean L2 error: 0.3111m
median L2 error: 0.3251m
duplicate_rate: 0.1600
```

`observation_count >= 5`:

- Metrics: `outputs/gt_aligned_10_label/metrics/metrics_41098076_100frames_text035_minobs5.json`

```text
predictions: 22
GT: 30
matches: 19
precision: 0.8636
recall: 0.6333
mean L2 error: 0.3086m
median L2 error: 0.2926m
duplicate_rate: 0.1364
```

해석:

- 100프레임은 50프레임보다 더 많은 viewpoint를 사용하므로 minobs 2/3에서도 recall 0.7000을 유지한다.
- minobs 1/2/3/4 모두 matches 21, recall 0.7000으로 같다.
- minobs 4는 recall을 유지하면서 precision 0.8400, duplicate_rate 0.1600까지 개선하므로 현재 대표 결과로 가장 적절하다.
- minobs 5는 precision 0.8636으로 더 높지만 recall이 0.6333으로 떨어져 대표 결과로는 과하게 보수적이다.

## 현재 권장 대표 결과

대표 결과는 100프레임, GT-aligned 10-label prompt, `observation_count >= 4`로 둔다.

```text
frames: 100 uniformly sampled frames
prompt: cabinet . chair . table . sofa . oven . refrigerator . washer . sink . tv monitor . stove .
box_threshold: 0.25
text_threshold: 0.35
observation_count >= 4
predictions: 25
GT: 30
matches: 21
precision: 0.8400
recall: 0.7000
mean L2 error: 0.3111m
median L2 error: 0.3251m
duplicate_rate: 0.1600
```

따라서 현재 GT-aligned 실험의 대표 설정은 100프레임 minobs 4이다. Recall 0.7000을 유지하면서 duplicate_rate를 0.1600까지 낮추고, precision도 0.8400까지 올라간다.

## 100프레임 text threshold sweep

대표 100프레임 설정에서 `text_threshold=0.35`가 적절한지 확인하기 위해 `0.30`, `0.40`을 추가 평가했다.

공통 설정:

```text
frames: 100 uniformly sampled frames
box_threshold: 0.25
prompt: cabinet . chair . table . sofa . oven . refrigerator . washer . sink . tv monitor . stove .
```

### text_threshold=0.30

- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_100frames_text030.json`
- Object count: 72

Label count:

```text
cabinet 13
chair 7
oven 4
refrigerator 8
sink 2
sofa 8
stove 1
table 11
tv_monitor 7
washer 11
```

Metrics:

```text
minobs 3: predictions 33, matches 21, precision 0.6364, recall 0.7000, duplicate_rate 0.3636
minobs 4: predictions 30, matches 21, precision 0.7000, recall 0.7000, duplicate_rate 0.3000
minobs 5: predictions 27, matches 20, precision 0.7407, recall 0.6667, duplicate_rate 0.2593
```

해석:

- 낮은 text threshold는 object count를 72개까지 늘린다.
- recall 0.7000은 유지되지만 duplicate_rate가 높아 대표 결과로는 덜 깔끔하다.

### text_threshold=0.40

- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_100frames_text040.json`
- Object count: 47

Label count:

```text
cabinet 10
chair 5
oven 1
refrigerator 5
sink 2
sofa 6
stove 1
table 7
tv_monitor 5
washer 5
```

Metrics:

```text
minobs 3: predictions 24, matches 17, precision 0.7083, recall 0.5667, duplicate_rate 0.2917
minobs 4: predictions 20, matches 16, precision 0.8000, recall 0.5333, duplicate_rate 0.2000
minobs 5: predictions 13, matches 11, precision 0.8462, recall 0.3667, duplicate_rate 0.1538
```

해석:

- 높은 text threshold는 object count를 줄이고 precision은 유지하지만 recall 손실이 크다.
- minobs 4 기준 recall이 0.5333까지 내려가므로 대표 결과로는 과하게 보수적이다.

### sweep 결론

`text_threshold=0.35`, `observation_count >= 4`를 대표 설정으로 유지한다.

```text
text_threshold 0.30, minobs 4: precision 0.7000, recall 0.7000, duplicate_rate 0.3000
text_threshold 0.35, minobs 4: precision 0.8400, recall 0.7000, duplicate_rate 0.1600
text_threshold 0.40, minobs 4: precision 0.8000, recall 0.5333, duplicate_rate 0.2000
```

`0.35`는 recall 0.7000을 유지하면서 precision과 duplicate_rate가 가장 좋다.

## 100프레임 association distance sweep

대표 100프레임 설정에서 object association 거리 threshold를 바꾸면 중복 object와 GT match가 어떻게 변하는지 확인했다.

공통 설정:

```text
frames: 100 uniformly sampled frames
box_threshold: 0.25
text_threshold: 0.35
prompt: cabinet . chair . table . sofa . oven . refrigerator . washer . sink . tv monitor . stove .
```

### association_distance_m=0.5

- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_100frames_text035_assoc050.json`
- Object count: 63

Label count:

```text
cabinet 14
chair 7
oven 1
refrigerator 8
sink 2
sofa 9
stove 1
table 8
tv_monitor 7
washer 6
```

Metrics:

```text
minobs 3: predictions 31, matches 21, precision 0.6774, recall 0.7000, duplicate_rate 0.3226
minobs 4: predictions 26, matches 21, precision 0.8077, recall 0.7000, duplicate_rate 0.1923
minobs 5: predictions 20, matches 17, precision 0.8500, recall 0.5667, duplicate_rate 0.1500
```

### association_distance_m=0.7

- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_100frames_text035_assoc070.json`
- Object count: 51

Label count:

```text
cabinet 12
chair 6
oven 1
refrigerator 6
sink 2
sofa 5
stove 1
table 7
tv_monitor 5
washer 6
```

Metrics:

```text
minobs 3: predictions 28, matches 20, precision 0.7143, recall 0.6667, duplicate_rate 0.2857
minobs 4: predictions 26, matches 20, precision 0.7692, recall 0.6667, duplicate_rate 0.2308
minobs 5: predictions 21, matches 18, precision 0.8571, recall 0.6000, duplicate_rate 0.1429
```

### association_distance_m=0.8

- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_100frames_text035_assoc080.json`
- Object count: 49

Label count:

```text
cabinet 11
chair 6
oven 1
refrigerator 6
sink 2
sofa 5
stove 1
table 7
tv_monitor 5
washer 5
```

Metrics:

```text
minobs 3: predictions 28, matches 20, precision 0.7143, recall 0.6667, duplicate_rate 0.2857
minobs 4: predictions 25, matches 19, precision 0.7600, recall 0.6333, duplicate_rate 0.2400
minobs 5: predictions 20, matches 17, precision 0.8500, recall 0.5667, duplicate_rate 0.1500
```

### association sweep 결론

`association_distance_m=0.6`을 유지한다.

```text
association 0.5, minobs 4: precision 0.8077, recall 0.7000, duplicate_rate 0.1923
association 0.6, minobs 4: precision 0.8400, recall 0.7000, duplicate_rate 0.1600
association 0.7, minobs 4: precision 0.7692, recall 0.6667, duplicate_rate 0.2308
association 0.8, minobs 4: precision 0.7600, recall 0.6333, duplicate_rate 0.2400
```

거리 threshold를 0.5로 줄이면 recall은 0.6과 같지만 precision이 낮아지고 duplicate_rate가 증가한다. 반대로 0.7/0.8로 키우면 object count는 줄지만 GT match도 같이 줄어든다. 현재 scene에서는 0.6이 precision, recall, duplicate_rate 균형이 가장 좋다.

## 200프레임 scaling 실험

100프레임보다 더 많은 viewpoint를 사용했을 때 recall이 개선되는지 확인했다.

공통 설정:

```text
frames: 200 uniformly sampled frames
box_threshold: 0.25
text_threshold: 0.35
association_distance_m: 0.6
prompt: cabinet . chair . table . sofa . oven . refrigerator . washer . sink . tv monitor . stove .
```

- Map: `outputs/gt_aligned_10_label/maps/41098076_semantic_map_200frames_text035.json`
- Object count: 75

Label count:

```text
cabinet 14
chair 11
oven 3
refrigerator 11
sink 2
sofa 7
stove 1
table 10
tv_monitor 9
washer 7
```

Metrics:

```text
minobs 4: predictions 35, matches 22, precision 0.6286, recall 0.7333, duplicate_rate 0.3714
minobs 5: predictions 33, matches 22, precision 0.6667, recall 0.7333, duplicate_rate 0.3333
minobs 6: predictions 29, matches 22, precision 0.7586, recall 0.7333, duplicate_rate 0.2414
minobs 8: predictions 23, matches 19, precision 0.8261, recall 0.6333, duplicate_rate 0.1739
```

해석:

- 200프레임은 minobs 4/5/6에서 matches 22, recall 0.7333으로 100프레임 대표 결과보다 match가 1개 늘어난다.
- 하지만 minobs 6 기준 precision 0.7586, duplicate_rate 0.2414라서 100프레임 minobs 4보다 map이 더 noisy하다.
- minobs 8은 duplicate_rate를 낮추지만 recall이 0.6333으로 떨어진다.

대표 설정 판단:

```text
100 frames, minobs 4: predictions 25, matches 21, precision 0.8400, recall 0.7000, duplicate_rate 0.1600
200 frames, minobs 6: predictions 29, matches 22, precision 0.7586, recall 0.7333, duplicate_rate 0.2414
```

Recall 최우선 결과는 200프레임 minobs 6이다. 다만 대표 결과는 precision과 duplicate_rate까지 고려해 100프레임 minobs 4를 유지한다.
