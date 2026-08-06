# Notion Mermaid 파이프라인 설계

## 목표

기존의 일렬 화살표 목록을 실제 입력 의존성을 보여주는 Mermaid 흐름도로 교체한다.

## 구조

- RGB Frame은 Grounding DINO Detection과 SAM Mask 생성을 순서대로 거친다.
- SAM Mask, Depth Map, Camera Pose는 3D Projection에서 합류한다.
- Projection 결과는 Frame-level Observation과 Association & Merge를 거쳐 Semantic Object Map이 된다.
- Semantic Object Map에서 Evaluation과 Query Demo로 분기한다.

## 표현

- 왼쪽에서 오른쪽으로 읽는 `flowchart LR`을 사용한다.
- 입력, 처리, 결과 노드를 서로 다른 저채도 색상과 테두리로 구분한다.
- 노드 안에는 한국어 설명과 기존 영문 컴포넌트명을 함께 사용한다.
- Mermaid 바로 아래의 `데이터 흐름` 설명은 상세 입력 정의로서 유지한다.

## 변경 범위와 검증

- `## 3. 전체 파이프라인`의 기존 8단계 화살표 목록만 교체한다.
- 다른 섹션, 이미지, 실험 결과, 라이브 데모는 변경하지 않는다.
- 적용 후 Mermaid 코드 블록, Projection 합류 간선 3개, Map 이후 분기 2개, 기존 데이터 흐름 설명의 존재를 재조회한다.
