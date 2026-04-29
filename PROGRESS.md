# Language-Grounded 3D Object Mapping Progress

> 기준 계획서: `PROJECT_PLAN.md`  
> 기간: 2026-04-24 ~ 2026-05-29  
> 현재 상태: ARKitScenes selected scene loaded; ARKitScenes adapter implemented  
> Primary dataset: ARKitScenes 3DOD single scene `41098076`  
> Fallback dataset: ScanNet/Replica single scene if ARKitScenes evaluation blocks

---

## 현재 결정 사항

- [x] 프로젝트 목표 확정: RGB-D single-scene 기반 language-grounded 3D semantic object map 생성
- [x] 데이터 전략 확정: ARKitScenes 3DOD `41098076` scene을 primary source로 사용
- [x] fallback 전략 확정: ARKitScenes 처리 실패 시 ScanNet 또는 Replica 한 scene으로 전환
- [x] map 범위 확정: 여러 dataset/scene을 섞지 않고 같은 scene의 RGB-D frame sequence만 누적
- [x] 탐지/세그멘테이션 전략 확정: Grounding DINO + SAM
- [x] 3D map 전략 확정: TSDF가 아닌 centroid 기반 object-level map
- [x] semantic map 전략 확정: base indoor vocabulary로 선제 map 생성 후, map에 없는 질의는 open-vocabulary expansion으로 추가 탐지/등록
- [x] 처리 방식 확정: realtime이 아닌 batch processing
- [x] query MVP 범위 확정: LLM 없이 규칙 기반 natural-language parser
- [x] attribute query 범위 확정: 색상/재질/형상 질의는 MVP 밖, future extension
- [x] 평가 지표 확정: Recall, Precision, 3D Localization Error, Duplicate Rate

---

## 전체 진행률

| 영역 | 상태 | 비고 |
|------|------|------|
| 계획/범위 정의 | Done | `PROJECT_PLAN.md` 최신 기준 |
| 환경 구성 | Done | `cv` Python 3.11.15 + PyTorch CUDA |
| 모델 가중치 | Done | Grounding DINO Swin-T OGC + SAM ViT-B |
| 데이터셋 확보 | Done | ARKitScenes `41098076`, RGB/depth/pose/GT annotation |
| Dataset adapter | Done | `datasets/arkitscenes_adapter.py` |
| Detection/Segmentation | In progress | ARKitScenes frame에서 Grounding DINO + SAM 연결 성공 |
| 3D projection | In progress | SAM mask → world centroid 연결 성공 |
| Semantic map | In progress | 3-frame demo map 생성, centroid association 동작 확인 |
| Query engine | Not started | label + query_type parser |
| Evaluation | In progress | 20-frame map GT centroid 평가 완료 |
| Visualization/demo | Not started | Open3D + notebook/video |

---

## Week 1: 환경 세팅 + ARKitScenes 단일 scene 로딩

**목표**: ARKitScenes 한 scene에서 RGB-D sequence를 공통 `Frame` 형식으로 읽고 단일 3D 투영을 검증한다.

- [ ] `cv` conda 환경 생성
  - [ ] Python 3.11 우선 사용
  - [ ] Python 3.12에서 설치 실패 시 3.11/3.10으로 fallback
- [ ] PyTorch CUDA 환경 확인
- [ ] Grounding DINO 설치 확인
- [ ] SAM 설치 확인
- [ ] Open3D 설치 확인
- [x] ARKitScenes 공식 download script 확보
- [x] Primary scene 1개 선정: `41098076`
- [x] 같은 scene에서 796 RGB-D frame 확보
- [x] ARKitScenes 3DOD files 다운로드
  - [x] lowres RGB frames
  - [x] lowres depth frames
  - [x] lowres wide intrinsics
  - [x] lowres wide trajectory
  - [x] mesh `.ply`
  - [x] 3DOD annotation `.json`
- [x] `datasets/arkitscenes_adapter.py` 구현
- [x] RGB/depth/pose/intrinsics를 공통 `ARKitFrame` 형식으로 로드
- [x] GT object annotation 로딩
  - [x] `segments.obbAligned` centroid/axes/rotation 사용
  - [x] labels: cabinet/table/chair/sofa/oven/refrigerator/washer/sink/tv_monitor/stove
- [x] 데이터 형식 검증
  - [x] RGB shape/dtype
  - [x] depth shape/dtype/scale
  - [x] pose matrix shape
  - [x] intrinsics 값
- [x] `src/projector.py` 기초 구현
- [x] 단일 frame에서 synthetic mask 3D point projection 성공

**Week 1 완료 기준**

- [x] ARKitScenes 1개 scene에서 796 frame의 RGB + depth + pose 로딩 성공
- [x] depth가 meter 단위로 해석되는지 확인
- [x] 단일 frame의 mask 또는 synthetic pixel을 3D point로 변환 성공

---

## Week 2: Grounding DINO + SAM

**목표**: text prompt에서 bbox를 만들고, bbox에서 object mask를 만든다.

- [x] `src/detector.py` 구현
  - [x] Grounding DINO model load
  - [x] base indoor vocabulary text prompt 입력
  - [x] bbox, label, confidence 반환
- [ ] base indoor vocabulary 정의
  - [ ] 평가용 primary category는 ScanNet GT 기준 5개 이하로 제한
  - [ ] demo용 vocabulary는 chair/table/sofa/bed/cabinet/desk/lamp/vase/plant 등으로 확장
- [x] `src/segmentor.py` 구현
  - [x] SAM model load
  - [x] bbox 입력
  - [x] binary mask 반환
- [ ] 단일 frame 탐지 테스트
  - [x] local RGB smoke test: `"chair . table . sofa . couch . vase . flower vase . plant pot ."`
  - [x] ARKitScenes `41098076` frame `2605.411`: base vocabulary detection + SAM + 3D projection
  - [ ] confidence threshold 기본값 `0.35`
- [x] bbox + mask overlay 이미지 저장
- [ ] 실패 prompt 기록 및 prompt 조정

**Week 2 완료 기준**

- [ ] 실내 장면 이미지에서 3개 이상 category에 대해 bbox 생성
- [ ] bbox에서 mask 생성
- [ ] mask area > 100px 조건 통과

---

## Week 3: 3D Projection + Semantic Map

**목표**: 2D mask를 world-frame 3D centroid로 변환하고, multi-frame object map으로 통합한다.

- [ ] `src/projector.py` 완성
  - [ ] `mask_to_pointcloud(mask, depth, K)`
  - [ ] `transform_to_world(points, T_cam2world)`
  - [ ] centroid 계산
- [x] `src/semantic_map.py` 구현
  - [x] `Object3D`
  - [x] `SemanticMap`
  - [x] `observation_count`
  - [x] `seen_frame_ids`
- [x] object association 구현
  - [x] same label
  - [x] centroid distance < 0.6m
  - [x] match 시 centroid 누적 평균 update
  - [x] match 실패 시 새 object 생성
- [x] 3 frame mini integration 테스트
- [x] `outputs/maps/41098076_semantic_map_demo.json` 저장
- [ ] TODO: label normalization 개선
  - [ ] `"tv monitor"` ↔ `tv_monitor`, `"couch"` ↔ `sofa` alias 안정화
  - [ ] `"cabinet refrigerator"`처럼 여러 class가 섞인 Grounding DINO phrase 처리 정책 결정
  - [ ] base vocabulary와 query-time expansion이 같은 canonical label 체계를 쓰도록 정리
- [x] 20 frame multi-frame integration 테스트
- [ ] `outputs/maps/scene_map.json` 저장

**Week 3 완료 기준**

- [ ] 50 frame 처리 후 SemanticMap에 5개 이상 object 등록
- [ ] 같은 object가 frame마다 중복 등록되지 않음
- [ ] `observation_count`가 반복 관측을 반영

---

## Week 4: Query + Evaluation + Visualization

**목표**: semantic map에 질의하고, GT와 비교해 정량 지표를 산출한다.

- [ ] `src/query_engine.py` 구현
  - [ ] location query: `"chair 위치"`
  - [ ] nearest query: `"가장 가까운 chair"`
  - [ ] count query: `"chair 몇 개"`
  - [ ] 규칙 기반 parser로 `label`, `query_type` 추출
- [ ] query-time open-vocabulary expansion 구현
  - [ ] map에 label이 있으면 기존 SemanticMap 조회
  - [ ] map에 label이 없으면 해당 label prompt로 추가 탐지
  - [ ] 추가 detection/mask를 3D centroid로 변환해 SemanticMap에 등록
  - [ ] expansion 객체는 base map 객체와 source metadata로 구분
- [ ] `gt_objects.json` 최종 생성
  - [ ] ScanNet primary scene annotation 기반
  - [ ] instance mesh vertices 평균 centroid
- [x] `src/evaluator.py` 구현
  - [x] Hungarian matching
  - [x] Recall
  - [x] Precision
  - [x] 3D Localization Error, mean L2
  - [x] Duplicate Rate
- [ ] `src/visualizer.py` 구현
  - [ ] camera trajectory 표시
  - [ ] predicted centroid 표시
  - [ ] GT centroid 표시
  - [ ] category별 색상 표시
- [ ] `outputs/metrics.json` 생성

**Week 4 완료 기준**

- [ ] Object Recall >= 0.6
- [ ] Object Precision >= 0.5
- [ ] 3D Localization Error <= 1.0m
- [ ] query 3종이 `query.py`에서 정상 동작

---

## Week 5: 통합 + 데모 + 보고서

**목표**: end-to-end pipeline을 정리하고 발표/보고서용 산출물을 만든다.

- [ ] `main.py` 통합
  - [ ] `python main.py --scene data/raw/ --prompts "chair . table . sofa . TV"`
- [ ] `query.py` 대화형 인터페이스 완성
- [ ] `evaluate.py` 실행 스크립트 완성
- [ ] `notebooks/demo.ipynb` 작성
  - [ ] bbox visualization
  - [ ] mask visualization
  - [ ] 3D points/centroid visualization
  - [ ] semantic map visualization
  - [ ] query demo
- [ ] 시연 영상 제작
  - [ ] ScanNet frame overlay
  - [ ] Open3D 3D map
  - [ ] text query response
- [ ] 발표 자료 작성
- [ ] 최종 보고서 작성

**Week 5 완료 기준**

- [ ] end-to-end run 성공
- [ ] `outputs/maps/scene_map.json` 생성
- [ ] `outputs/metrics.json` 생성
- [ ] 시연 영상 완성
- [ ] 보고서/발표자료 완성

---

## 테스트 및 검증 체크리스트

| 단계 | 대상 | 상태 | 완료 기준 |
|------|------|------|----------|
| Unit | `src/projector.py` | Not started | known point 오차 < 1e-5 |
| Unit | `src/semantic_map.py` | Not started | merge/split case 통과 |
| Unit | `src/evaluator.py` | Not started | hand-labeled expected 값과 일치 |
| Integration | dataset adapter | Not started | 5 frame 연속 로딩 성공 |
| Smoke | `main.py` | Not started | `outputs/maps/scene_map.json` 생성 |
| Full run | `main.py` + `evaluate.py` | Not started | `outputs/metrics.json` 생성 |

---

## Acceptance Criteria

- [ ] AC1: Grounding DINO 텍스트 탐지, confidence >= 0.35
- [x] AC2: SAM bbox-to-mask, mask area > 100px
- [x] AC3: 단일 frame 3D projection, Z > 0
- [ ] AC4: 50 frame multi-frame map integration
- [x] AC5: 중복 제거 동작 확인
- [ ] AC6: query 3종 정상 응답
- [ ] AC7: Object Recall >= 0.6
- [ ] AC8: Object Precision >= 0.5
- [ ] AC9: 3D Localization Error <= 1.0m
- [ ] AC10: Open3D visualization screenshot/video
- [ ] AC11: end-to-end demo video
- [ ] AC12: map에 없는 open-vocabulary label 질의 시 추가 탐지 후 SemanticMap 등록 또는 not-found 반환

---

## 리스크 로그

| 리스크 | 상태 | 대응 |
|--------|------|------|
| ARKitScenes 좌표/단위 불일치 | Open | 공식 loader 방식으로 pose 해석, projection 단계에서 GT scale 재확인 |
| GT centroid 추출 난이도 | Mitigated | ARKitScenes `segments.obbAligned.centroid` 사용 |
| Grounding DINO 설치 실패 | Open | Python 3.11/3.10 fallback |
| Grounding DINO 탐지율 저조 | Open | prompt/threshold 조정 |
| Grounding DINO label phrase ambiguity | Open | label normalization 개선, mixed phrase skip/분리 정책 추가 |
| base vocabulary 누락으로 질의 객체 미등록 | Open | map miss 시 open-vocabulary expansion으로 추가 탐지 |
| depth scale mismatch | Open | Week 1에서 meter 단위 검증 |
| 같은 객체 merge/split 오류 | Open | association threshold 0.5m로 시작, 고정 실험 설정 기록 |
| 10-20 frame 확장 시 duplicate 증가 | Open | label normalization 후 threshold/observation_count 기준 조정 |
| 일정 초과 | Open | Week 4 이후 query 2종으로 축소 가능 |

---

## 산출물 체크리스트

- [x] ARKitScenes `41098076` raw frame subset
- [ ] `data/gt_objects.json`
- [x] `datasets/arkitscenes_adapter.py`
- [x] `src/detector.py`
- [x] `src/segmentor.py`
- [ ] `src/projector.py`
- [x] `src/semantic_map.py`
- [ ] `src/query_engine.py`
- [x] `src/evaluator.py`
- [ ] `src/visualizer.py`
- [ ] `main.py`
- [ ] `query.py`
- [ ] `evaluate.py`
- [ ] `outputs/maps/scene_map.json`
- [ ] `outputs/metrics.json`
- [ ] `outputs/figures/`
- [ ] `notebooks/demo.ipynb`
- [ ] 시연 영상
- [ ] 발표 자료
- [ ] 최종 보고서

---

## 진행 로그

### 2026-04-24

- [x] `PROJECT_PLAN.md` 기준 계획 정리 완료
- [x] ScanNet single-scene primary 전략 확정
- [x] Replica fallback 전략 확정
- [x] Isaac Sim을 future extension으로 격하
- [x] object association, query MVP, 평가 지표, 테스트 계획을 문서화
- [ ] 구현 시작 전 다음 액션: `cv` 환경 구성 및 ScanNet 접근 승인 확인

### 2026-04-28

- [x] `cv` conda 환경을 Python 3.11.15 기준으로 정리
- [x] PyTorch CUDA, Grounding DINO, SAM, Open3D, SciPy 설치 및 import 검증
- [x] Grounding DINO Swin-T OGC config/checkpoint 다운로드
- [x] SAM ViT-B checkpoint 다운로드
- [x] 단일 RGB 이미지 smoke test에서 `chair`, `table`, `vase/plant pot` bbox + mask 생성 확인
- [x] 프로젝트 방향 보강: base indoor vocabulary map building + query-time open-vocabulary expansion
- [x] ARKitScenes annotation-only preview로 `41098076` scene 선정
- [x] ARKitScenes `41098076` full 3DOD scene 다운로드 및 기존 후보 scene 정리
- [x] `datasets/arkitscenes_adapter.py` 구현
- [x] `scripts/inspect_arkitscenes_scene.py`로 RGB/depth/intrinsics/pose/GT 로딩 검증
- [x] `src/projector.py` 구현 및 `scripts/verify_projector.py`로 synthetic mask 단일 frame 3D projection 검증
- [x] ARKitScenes frame에서 base vocabulary Grounding DINO detection + SAM + 3D projection 실행
- [x] 여러 frame 처리용 semantic map/object association 구현
- [ ] TODO: label normalization 개선 후 10-20 frame map build 실행
- [x] 20 frame 균등 샘플링 map 생성: `outputs/maps/41098076_semantic_map_20frames.json`
  - object_count: 40
  - label_counts: tv_monitor 9, cabinet 9, chair 7, table 6, refrigerator 4, stove 3, sink 2
  - 반복 관측 상위: cabinet_003 10회, cabinet_016 9회, cabinet_010 8회, table_032 6회
  - 관찰: semantic map은 생성되지만 `tv_monitor` 과검출과 mixed phrase 이슈가 있어 label normalization/필터링 필요
- [x] text_threshold 실험: `0.25 → 0.35`
  - 결과 파일: `outputs/maps/41098076_semantic_map_20frames_text035.json`
  - object_count: 40 → 25
  - label_counts: chair 7, cabinet 6, table 4, tv_monitor 4, refrigerator 2, stove 1, sink 1
  - 관찰: mixed phrase 상당수가 빈 label로 제외되어 map이 더 깔끔해짐
  - 다음 기준 후보: `box_threshold=0.25`, `text_threshold=0.35`
- [x] GT centroid 기반 평가 실행
  - 전체 예측 기준: `outputs/metrics_41098076_text035_minobs1.json`
    - predictions 25, GT 30, matches 13
    - precision 0.52, recall 0.4333, mean L2 0.3032m, median L2 0.2305m, duplicate_rate 0.48
  - 반복 관측 객체 기준(`observation_count >= 2`): `outputs/metrics_41098076_text035_minobs2.json`
    - predictions 11, GT 30, matches 9
    - precision 0.8182, recall 0.3, mean L2 0.3266m, median L2 0.2305m, duplicate_rate 0.1818
  - 해석: 반복 관측 필터를 쓰면 precision은 크게 좋아지지만, 20 frame만 사용해서 recall은 낮음
