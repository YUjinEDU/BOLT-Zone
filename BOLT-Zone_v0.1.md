
# BOLT-Zone 공식 문서 (v0.1)



## 1. 한 줄 소개



**BOLT-Zone**은 야구공의 **모션 블러(blur)**를 “노이즈”가 아니라 “정보(방향/속도 힌트)”로 다루는 **blur-aware 라벨 철학**을 계승하되, 이를 **YOLO26 + OBB(회전 박스)**와 **가변연산 게이팅(gating)**으로 재구성해 **노트북 CPU 실시간 스트라이크 판별**을 목표로 하는 실전형 시스템이다.

(Detect는 항상 가볍게, Refine은 필요할 때만 켠다)



---



## 2. 설계 목표



### 2.1 기능 목표



* 입력: 영상 스트림(심판 시점 / 포수 POV / 직접 촬영)

* 출력:



&nbsp; 1. 공 트랙(프레임별 공 중심 좌표, 시간)

&nbsp; 2. 블러 벡터(각도, 길이 등) — Refine 단계

&nbsp; 3. 최종 판정: Strike / Ball (+ 교차 위치, 교차 시각)



### 2.2 시스템 목표



* **CPU only** 노트북에서도 “실시간에 가까운” 동작 (추론/지연 안정성)

* GPU가 있으면 더 높은 FPS 및 더 높은 정확도

* 재현 가능(Export: ONNX / OpenVINO로 배포 파이프라인 포함)

&nbsp; Ultralytics는 Export 모드에서 다양한 포맷을 지원하고, ONNX/OpenVINO 변환 가이드를 제공한다. ([docs.ultralytics.com][1])



---



## 3. 핵심 아이디어(개념) — 왜 OBB + Gating인가?



### 3.1 Blur-aware 관측 모델



야구공이 빠르면 프레임에서 공은 점이 아니라 **선분(스트릭)**처럼 관측된다.

우리는 블러 스트릭을 다음의 “관측”으로 모델링한다:



* 공 중심(블러 중앙): (mathbf{c} = (x_c, y_c))

* 블러 방향(각도): (theta)

* 블러 길이(픽셀): (l)



블러 스트릭의 양 끝점은

[

mathbf{p}_1 = mathbf{c} + frac{l}{2}begin{bmatrix}costheta sinthetaend{bmatrix},quad

mathbf{p}_2 = mathbf{c} - frac{l}{2}begin{bmatrix}costheta sinthetaend{bmatrix}

]

처럼 정의할 수 있다.

이 철학이 “BlurBall식” blur-aware 라벨링의 요지(중앙점/방향/길이를 정보로 사용)이고, BOLT-Zone은 이를 **OBB(회전 박스)**로 구현한다.



### 3.2 OBB로 블러를 표현하는 이유



Ultralytics OBB는 객체를 “회전된 박스”로 감싸므로, 블러 스트릭을 **길쭉한 회전 박스**로 라벨링하면:



* 박스 중심 = (mathbf{c})

* 긴 변 방향 = (theta)

* 긴 변 길이 = (l) (혹은 (l)에 비례하는 값)



으로 자연스럽게 연결된다.

Ultralytics YOLO는 OBB 태스크를 별도로 제공하며 `-obb` 접미사의 모델(`yolo26n-obb.pt` 등)로 학습/예측/Export를 지원한다. ([docs.ultralytics.com][2])



### 3.3 Gating(가변연산)이 필요한 이유



야구 영상은 “공이 날아오는 시간”이 전체의 일부다.

그래서 **비용 큰 정밀화(OBB refine)**를 매 프레임 돌리면 CPU에서 낭비가 크다.



BOLT-Zone은 “상태(state)”를 둔다:



* **Idle**: 공 트랙 없음 (Refine 절대 금지)

* **Acquire**: 공 등장 후보 확보

* **Track**: 공 트랙 유지

* **End**: 포구/타격/이탈로 트랙 종료



그리고 Refine은 오직 **Track 상태**에서만 “불확실성 신호”가 감지될 때 켠다.



---



## 4. 시스템 아키텍처



### 4.1 모듈 구성



* **BOLT-Detect**: YOLO26n(일반 bbox)로 공을 빠르게 탐지

* **BOLT-Track**: ByteTrack 또는 BoT-SORT로 트랙 ID 유지

&nbsp; Ultralytics는 Track 모드에서 BoT-SORT/ByteTrack을 지원하고 YAML로 설정 가능. ([docs.ultralytics.com][3])

* **BOLT-Gate**: “지금 Refine 켤까?”를 판단하는 게이팅 로직

* **BOLT-Refine**: YOLO26n-OBB(회전 박스)로 블러 스트릭 정밀 추정

&nbsp; OBB 모델/태스크 안내 및 사용은 Ultralytics 문서에 정리되어 있다. ([docs.ultralytics.com][2])

* **BOLT-Zone**: 궤적/교차 계산 및 Strike/Ball 판정기



### 4.2 데이터 흐름(프레임 단위)



1. 프레임 (I_t) 입력

2. (가볍게) **BOLT-Detect** → 후보 bbox(중심 (hat{mathbf{c}}_t), conf)

3. **BOLT-Track** → track state 추정(트랙 존재 여부, 예측 위치)

4. **BOLT-Gate** → Refine ON/OFF

5. Refine ON이면 ROI에서 **BOLT-Refine(OBB)** 실행

&nbsp;  → (hat{mathbf{c}}_t, hat{theta}_t, hat{l}_t) 갱신

6. **BOLT-Zone** → 궤적 업데이트 → 스트라이크존 평면 교차 → 판정



---



## 5. 데이터셋 설계



### 5.1 입력 시점(도메인) 3종



1. 심판 시점(MLB/Skilled Catcher 류)

2. 포수 POV(POV BASEBALL 류)

3. 직접 촬영(권리/배포/재현성 측면에서 가장 안전하고 논문에도 유리)



※ 실험 설계에서 **시점별 분리 테스트**를 반드시 권장:



* Train/Val은 일부 시점으로 구성하되

* Test는 “완전 다른 시점”을 통째로 두어 도메인 쉬프트 평가



### 5.2 라벨 종류 2종(핵심)



* **Detect 라벨 (bbox)**: 공의 축정렬 박스

* **Refine 라벨 (OBB)**: 블러 스트릭 회전 박스



Ultralytics OBB 데이터는 “네 꼭짓점” 기반의 YOLO OBB 포맷을 사용한다. ([docs.ultralytics.com][4])

(실제 라벨 파일은 공백 구분을 흔히 사용하며, class + 8개 좌표 형태가 일반적이라는 커뮤니티 확인도 있다. ([GitHub][5]))



### 5.3 OBB 라벨 포맷(정규화 좌표)



각 객체 인스턴스는 한 줄:



[

texttt{class};;x_1;y_1;x_2;y_2;x_3;y_3;x_4;y_4

]



* ((x_i, y_i))는 이미지 너비/높이로 나눈 **0~1 정규화 좌표**

* 점 순서는 일관되게(시계/반시계) 유지



Ultralytics 문서가 OBB를 네 꼭짓점으로 표현한다고 명시한다. ([docs.ultralytics.com][4])



### 5.4 라벨링 도구 추천



* **CVAT**: Ultralytics YOLO 포맷(Detection/OBB 등)을 지원한다고 안내되어 있음. ([docs.cvat.ai][6])

&nbsp; → “Detect는 bbox”, “Refine는 OBB(회전 박스)”를 CVAT에서 분리 태스크로 만들거나 동일 태스크에 2종 라벨로 관리 가능.



---



## 6. BOLT-Gate 설계 (가변연산 핵심)



### 6.1 왜 conf 하나로 부족한가?



야구 영상에는 흰색 잡음(조명 반사, 라인, 글러브, 공인 듯한 물체)이 많아 conf만으로 Refine를 켜면 오탐에 취약하다.



그래서 Gate는 “트랙 기반 상태” + “불확실성 신호”의 AND 조건으로 설계한다.



### 6.2 상태 머신(필수)



* **Idle**: 활성 트랙 없음 → Refine 금지

* **Acquire**: 후보가 연속 (k)프레임 관측되면 Track으로 전이

* **Track**: 트랙 유지 중 → Refine 후보 가능

* **End**: 트랙 끊김/포구 이벤트 → Idle 복귀



Ultralytics Track 모드는 탐지 결과에 ID를 붙여 트랙을 유지하는 형태이며, BoT-SORT/ByteTrack 같은 트래커를 사용한다. ([docs.ultralytics.com][3])



### 6.3 불확실성 신호(추천 3종)



Track 상태에서만 아래를 평가:



1. **Detection 품질 저하**
```
[
text{low_conf} = (text{conf}_t < tau_c)
]
```
2. **트랙 예측 대비 잔차(residual) 증가**
```
[
r_t = |hat{mathbf{c}}^{text{det}}_t - hat{mathbf{c}}^{text{pred}}_t|
quad,quad
text{high_resid}=(r_t>tau_r)
]
```
3. **지터(jitter) 증가**

최근 (W)프레임 중심 분산:
```
[
J_t = mathrm{Var}(hat{x}*{t-W:t}) + mathrm{Var}(hat{y}*{t-W:t})
quad,quad
text{high_jitter}=(J_t>tau_j)
]
```


**Gate 규칙 예시**

```
[

text{RefineOn} =

text{TrackAlive} land

(text{low_conf} lor text{high_resid} lor text{high_jitter})

]



※ 더 공격적으로 하려면 “블러 징후(길쭉한 모양)” 같은 신호도 추가할 수 있다.



---



## 7. 트래킹(Tracking) 개념 요약



### 7.1 ByteTrack / BoT-SORT를 왜 쓰나?



* “탐지”는 프레임별 결과이지만, 판정은 시간 축에서 궤적이 중요하다.

* Multi-Object Tracking은 객체에 ID를 부여해 프레임 간 일관성을 준다.

* Ultralytics는 Track 모드에서 BoT-SORT/ByteTrack을 제공하고 YAML로 바꿀 수 있다. ([docs.ultralytics.com][3])



(참고로 ByteTrack은 낮은 score의 검출도 연관에 활용하는 아이디어로 알려져 있고, 탐지기 선택에 유연하다는 설명 자료가 있다. ([datature.com][7]))



### 7.2 구현 포인트



* BOLT-Detect의 출력 박스를 Track 모드에 넣어 트랙 생성

* 트랙이 존재할 때만 Gate 평가

* Gate가 Refine를 켜면, Refine 결과(더 좋은 중심/각도/길이)를 트랙 업데이트에 반영(“측정 업데이트”)



---



## 8. 스트라이크 판정(Zone) 수학/개념



여기부터는 네 프로젝트(AR Strike Zone)와 직접 연결되는 핵심이야. 구현은 다양한 방식이 가능하지만 “논문/문서” 관점에서 가장 깔끔한 기본형을 적을게.



### 8.1 좌표계



* 이미지 좌표: ((u,v)) 픽셀

* 카메라 좌표: ((X,Y,Z))

* 스트라이크존은 3D 공간의 직육면체(또는 평면 폴리곤 집합)로 정의



### 8.2 2D 트랙 → 3D 추정(선택지)



1. **단일 카메라 + 보정(캘리브레이션) + 공 크기 기반 거리 추정(근사)**

2. **표식(ArUco 등) 기반으로 스트라이크존 평면/좌표계 복원**

3. **스테레오(두 카메라)로 깊이 계산**



BOLT-Zone의 논문 새로움은 “블러-인지 + CPU 실시간”이 중심이므로, 3D 복원은 “기존 AR 스트라이크존 파이프라인”과 결합한다고 기술하면 충분하다. (세부는 너희가 이미 가진 자산)



### 8.3 판정 핵심



* 공 중심 궤적 (mathbf{c}_t) 또는 3D 궤적 (mathbf{x}_t)

* 스트라이크존 평면(또는 전면)과의 교차 시각 (t^*)를 추정

* 교차 위치가 존 내부면 Strike



시간 보간(선형):

[

t^* approx t_0 + alpha (t_1-t_0),quad

alpha = frac{d_0}{d_0-d_1}

]

여기서 (d_0,d_1)는 각 시각에서 평면까지의 부호 거리(signed distance).

교차점 (mathbf{x}^*) 계산 후, 존 내부 포함 테스트(폴리곤/박스 포함) 수행.



---



## 9. 학습/추론/배포(End-to-End 워크플로우)



### 9.1 프로젝트 레포 구조(권장)



```

BOLT-Zone/

 docs/
   BOLT-Zone.md              # 이 문서
   dataset_spec.md
   labeling_guide.md
   evaluation_protocol.md
 data/
   raw/                      # 원본 영상
   clips/                    # 공 등장 구간 클립
   yolo_detect/              # bbox 라벨
   yolo_obb/                 # OBB 라벨
 bolt/
   detect/
   refine/
   gate/
   zone/
   track/
   utils/
 scripts/
   extract_clips.py
   convert_labels.py
   train_detect.sh
   train_obb.sh
   export_models.sh
   benchmark_cpu.py
 runs/

```



### 9.2 YOLO26 Detect 학습 (BOLT-Detect)



* 목표: 공을 놓치지 않게(Recall↑), CPU에서 빠르게



Ultralytics는 모델/학습/예측을 통합 API/CLI로 제공한다. ([GitHub][8])



(예시, 개념)



* `yolo detect train model=yolo26n.pt data=... imgsz=...`



### 9.3 YOLO26 OBB 학습 (BOLT-Refine)



Ultralytics OBB 태스크 문서:



* `yolo26n-obb.pt` 같은 OBB 모델 사용

* 커스텀 OBB 데이터로 train/val/predict/export 가능 ([docs.ultralytics.com][2])



### 9.4 Tracking 실행 (BOLT-Track)



Ultralytics Track 모드:



* BoT-SORT/ByteTrack 지원

* YAML로 tracker 설정 가능 ([docs.ultralytics.com][3])



### 9.5 Export & CPU 배포 (필수)



* Export 모드 개요: 다양한 포맷으로 내보내기 ([docs.ultralytics.com][1])

* ONNX Export: 변환 및 ONNX Runtime 활용 안내 ([docs.ultralytics.com][9])

* OpenVINO Export: CPU 속도 향상(최대 3x) 및 Intel GPU/NPU 가속 언급 ([docs.ultralytics.com][10])



권장 배포 조합:



* 노트북 CPU: **OpenVINO** 또는 **ONNX Runtime**

* GPU 있으면: TensorRT도 후보(추후)



---



## 10. 평가 프로토콜(논문용으로 강하게)



### 10.1 Detect 평가



* 공 검출 Recall @ 관심 구간(스트라이크존 근처)

* FP(오탐)율: 공이 없을 때 공이 있다고 잡는 비율



### 10.2 Refine 평가



* 중심점 오차(픽셀): (|hat{mathbf{c}}_t - mathbf{c}^{gt}_t|)

* 각도 오차: (|hat{theta}-theta^{gt}|)

* 길이 오차: (|hat{l}-l^{gt}|)



### 10.3 End-to-End(최종 KPI)



* **Strike/Ball 정확도**

* 교차 시각 오차 (|t^*-hat{t}^*|)

* 교차 위치 오차 (|mathbf{x}^*-hat{mathbf{x}}^*|)



### 10.4 실시간성(중요: 평균보다 분포)



* 평균 FPS

* 지연의 95퍼센타일(p95) / 최악 지연(max)

* CPU 사용률, 전력(가능하면)



---



## 11. 코딩 에이전트에게 줄 “작업 분해(Task DAG)”



코딩 에이전트가 바로 착수하도록 “모듈 단위”로 나누면:



1. **데이터 파이프라인**



&nbsp;  * 영상→클립 추출(공 등장 구간)

&nbsp;  * 라벨 포맷 검증(Detect/OBB)

2. **학습 스크립트**



&nbsp;  * Detect 학습

&nbsp;  * OBB 학습

3. **추론 런타임**



&nbsp;  * 프레임 입력 / 전처리 / ROI 생성

&nbsp;  * Detect→Track→Gate→(Refine)→Zone 흐름 구현

4. **Export/배포**



&nbsp;  * ONNX/OpenVINO 변환 스크립트

&nbsp;  * CPU 벤치마크(지연 분포 측정)

5. **평가**



&nbsp;  * 트랙 품질/판정 정확도/지연 분포 리포트 생성



---



# 참고문헌 / 레퍼런스 정리 (문서용)



## A. Ultralytics YOLO26 / OBB / Tracking / Export (공식)



* Ultralytics OBB Datasets Overview — YOLO OBB 포맷(네 꼭짓점) 설명 ([docs.ultralytics.com][4])

* Ultralytics OBB Task Guide — `-obb` 모델, 학습/예측/Export 개요 ([docs.ultralytics.com][2])

* Ultralytics Track Mode — BoT-SORT/ByteTrack 트래킹, 설정 방법 ([docs.ultralytics.com][3])

* Ultralytics Tracking Datasets Overview — 지원 트래커 목록 및 현재 구조(탐지 모델 재사용) ([docs.ultralytics.com][11])

* Ultralytics Export Mode — 모델 Export 개요 ([docs.ultralytics.com][1])

* Ultralytics ONNX Integration — ONNX Export 및 배포 개요 ([docs.ultralytics.com][9])

* Ultralytics OpenVINO Integration — OpenVINO Export 및 CPU speedup 언급 ([docs.ultralytics.com][10])



## B. 라벨링 툴/포맷



* CVAT: Ultralytics YOLO 포맷(Detection/OBB 등) 안내 ([docs.cvat.ai][6])

* Ultralytics OBB 포맷 관련 Discussion(구분자 등 실무 참고) ([GitHub][5])



## C. 트래킹 개념 참고(보조 자료)



* BYTETrack 개념 소개(탐지기 선택 유연성 등) ([datature.com][7])



---



## 다음으로 내가 같이 해줄 것(바로 실무로 이어지는 항목)



이 문서를 기반으로, 다음 단계로는 보통 “실제 파일/코드”가 필요해.

유진이 지금 가진 폴더(Strikezone, blurball clone 등)와 맞춰서:



* **라벨 규칙(OBB 폭/길이 정의)**를 야구공 데이터에 맞게 확정

* **Ultralytics 학습 커맨드/하이퍼파라미터 템플릿** 제공

* **BOLT-Gate 임계값((tau_c,tau_r,tau_j)) 자동 튜닝**(validation 기반)

* **CPU(OpenVINO/ONNX) 벤치마크 스크립트**까지 만들어서 “p95 지연” 그래프 뽑기



원하면 바로 이어서 “`dataset_spec.md` / `labeling_guide.md` / `evaluation_protocol.md`”를 각각 더 길게, 논문 수준으로 쪼개서 만들어줄 수도 있어.





[1]: https://docs.ultralytics.com/modes/export/?utm_source=chatgpt.com "Model Export with Ultralytics YOLO"

[2]: https://docs.ultralytics.com/tasks/obb/?utm_source=chatgpt.com "Oriented Bounding Boxes Object Detection"

[3]: https://docs.ultralytics.com/modes/track/?utm_source=chatgpt.com "Multi-Object Tracking with Ultralytics YOLO"

[4]: https://docs.ultralytics.com/datasets/obb/?utm_source=chatgpt.com "Oriented Bounding Box (OBB) Datasets Overview"

[5]: https://github.com/orgs/ultralytics/discussions/8462?utm_source=chatgpt.com "datasets/obb/ · ultralytics · Discussion #8462"

[6]: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/?utm_source=chatgpt.com "Ultralytics YOLO"

[7]: https://datature.com/blog/introduction-to-bytetrack-multi-object-tracking-by-associating-every-detection-box?utm_source=chatgpt.com "An Introduction to BYTETrack: Multi-Object Tracking ..."

[8]: https://github.com/ultralytics/ultralytics?utm_source=chatgpt.com "Ultralytics YOLO"

[9]: https://docs.ultralytics.com/integrations/onnx/?utm_source=chatgpt.com "ONNX Export for YOLO26 Models"

[10]: https://docs.ultralytics.com/integrations/openvino/?utm_source=chatgpt.com "Intel OpenVINO Export"

[11]: https://docs.ultralytics.com/datasets/track/?utm_source=chatgpt.com "Multi-object Tracking Datasets Overview"



