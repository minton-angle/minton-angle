# 🏸 민턴 각? (Minton-Angle)

> **배드민턴 입문자를 위한 실시간 AI 코칭 앱**
> "'오늘 배드민턴 각?' 이라는 친근한 제안과, 입문자에게 꼭 필요한 '정석의 각도'를 알려주겠다는 두 가지 의미를 담았습니다."

🔗 **서비스 링크:** [minton-angle.vercel.app](https://minton-angle.vercel.app)

---

## 🎯 1. 프로젝트 개요 (Overview)
배드민턴은 남녀노소 즐기는 압도적인 인기 스포츠이지만, 입문자들이 겪는 진입 장벽이 존재합니다. 
레슨을 받기에는 시간과 장소가 한정적이고 비용이 부담되며, 기존 서비스들은 "스윙을 열심히 연습해보세요"와 같은 추상적인 피드백을 제공하여 자기주도 학습에 한계가 있었습니다.

**민턴 각?**은 이러한 문제를 해결하기 위해 컴퓨터 비전(CV)과 LLM을 결합하여, **스마트폰 하나로 언제 어디서나 전문가 수준의 1:1 맞춤형 AI 코칭**을 받을 수 있는 서비스를 제공합니다.

---

## ✨ 2. 주요 기능 (Key Features)

### 🏸 그립 교정 (Grip Correction)
* 부상 방지와 실력 향상의 기초인 '올바른 포핸드 그립' 여부를 실시간으로 판별합니다.
* 오답 클래스(테니스 그립, 검지 펴짐, 엄지 펴짐 등)를 직관적으로 분석하여 피드백을 제공합니다.



### 🧍‍♂️ 기본 스윙 자세 교정 (Swing Posture Correction)
* 스윙을 **3단계(준비, 백스윙, 임팩트/팔로우)** 로 구분하여 세분화된 자세 분석을 제공합니다.
* 전문가(국가대표 선출 코치)의 정석 자세(Ground Truth)와 내 자세를 1:1로 비교 시각화하여 문제점을 정확히 짚어줍니다.




### 📊 LLM 종합 AI 피드백 리포트 (Comprehensive Report)
* 과거 기록을 바탕으로 주/월 단위 성장 추이와 자세 변동성을 분석합니다.
* 점수가 낮고 변동성이 큰 동작을 찾아내어, **RAG 기반 맞춤형 코칭 피드백**과 연관된 **유튜브 추천 영상**을 제공합니다.



---

## 🛠 3. 기술 스택 (Tech Stack)

### 💻 Frontend & Backend
* **Frontend:** HTML5, CSS3, JavaScript (Vercel)
* **Backend:** FastAPI, Python (AWS Cloud, Docker)
* **Database:** PostgreSQL

### 🧠 AI & Computer Vision
* **Vision Models:** YOLO11n, MediaPipe Pose, TrackNetV3
* **LLM:** Llama-3.1-8b-instant (RAG Pipeline 적용)
* **Embeddings:** `intfloat/multilingual-e5-base`

---

## 🏗 4. 시스템 아키텍처 및 API 설계

### 🗄 Database (ERD 요약)
* `USER`: 사용자 계정 및 인증 정보 관리
* `POST`: 분석 세션 관리 (실시간/동영상) 및 종합 점수
* `FILE`: 키프레임 이미지 및 영상 파일 경로 저장
* `ANALYSIS`: CV 분석 결과 및 단계별 오차/점수 저장
* `LLM_REPORT`: 생성된 맞춤형 LLM 피드백 보관

### 📡 주요 API (FastAPI)
* `/api/auth/*`: 회원 가입, 로그인, 정보 조회/수정
* `/api/realtime/*`: 웹캠을 통한 실시간 스윙 3회 분석 및 즉각 피드백
* `/api/upload/*`: 동영상 업로드, 상태 조회 및 분석 결과 반환
* `/api/report/*`: 스윙 자세 리포트, LLM 피드백 리포트 생성 및 조회
* `/api/grip/*`: 라켓 그립 이미지 전송 및 AI 진단 결과 반환
* `/api/calendar/*`: 날짜별 분석 히스토리 및 월별 요약 제공

---

## 🧠 5. 핵심 AI 알고리즘 (Core AI Algorithms)

### 5.1. 그립 분류 알고리즘 (YOLO11n)
* **접근 방식의 전환:** 초기 MediaPipe 기반 3D 각도 측정 방식을 시도했으나, 손가락 겹침 및 촬영 구도 제약으로 인식 한계가 발생했습니다. 이를 해결하기 위해 단순 각도 계산 방식을 버리고 **YOLO 기반의 객체 분류(Classification) 모델로 재정의**했습니다.
* **클래스 정의 (6종):** [정답] 올바른 그립 / [오답] 테니스 그립, 검지 펴짐, 검지-엄지 순서 불일치, 엄지 펴짐 / 기타
* **모델 최적화:** 정확도(mAP50 0.987)와 추론 속도(`inference_time_ms`), 모델 크기를 종합적으로 평가하여 **YOLO11n**을 최종 채택했습니다.

### 5.2. 스윙 자세 교정 알고리즘 (MediaPipe & FastDTW)
국가대표 출신 코치의 완벽한 스윙 동작을 Ground Truth(GT)로 삼고, 100점 만점 룰 기반 평가를 진행합니다.
* **전처리 및 정규화:** 다양한 신체 크기에 대응하기 위해 어깨 폭과 골반을 기준으로 스켈레톤 데이터를 정규화하고 강력한 보간(Interpolation) 처리를 적용했습니다.
* **FastDTW:** 사람마다 다른 스윙 속도를 맞추기 위해 전문가와 사용자의 프레임을 동기화했습니다.
* **10개 세부 평가 지표:**
  * **Ready (준비):** 팔꿈치 각도, 손목 높이, 어깨-발 너비, 팔꿈치 높이
  * **Backswing (백스윙):** 골반/어깨 회전율, 손목 깊이, 팔꿈치 들림 비, L자 팔 각도, 골반 회전, 어깨 회전
  * **Impact & Follow:** 팔 뻗음 각도, 손목 높이 비율, 팔로우스루 시 손목/팔꿈치 x좌표 교차 검증

### 5.3. 맞춤형 LLM 코칭 (RAG Architecture)
실제 배드민턴 코칭 매뉴얼 자료를 벡터화하여 DB에 구축했습니다. 분석된 약점 데이터(점수 및 변동성)를 프롬프트와 함께 Llama 3 모델에 전달하여, "왜 틀렸는지"와 "어떻게 고쳐야 하는지"를 구체적으로 알려주는 전문적인 텍스트 리포트를 생성합니다.

---

## 🚀 6. R&D 및 트러블슈팅 (Troubleshooting)

### 🔴 셔틀콕 궤적 기반 스윙 분석의 구조적 한계 (R&D)
* **시도:** TrackNetV3와 YOLO를 이용해 셔틀콕의 타점과 낙하점을 포착하고 이상적인 스윙 궤적을 시각화하고자 했습니다.
* **결론:** 일반 체육관의 하얀 벽으로 인한 보호색 현상(객체 소실), 조명 노이즈, 정적 객체 오탐지 등 현존하는 비전 AI가 겪는 통제되지 않은 환경에서의 명백한 기술적 한계를 확인했습니다. 무리한 도입 대신, 좀 더 보완 후 기능 추가를 할 예정입니다.

### 🟢 브라우저 캐시 오염에 따른 동기화 문제 해결
* **문제:** 프론트엔드/백엔드 코드 연동 과정에서 예상치 못한 브라우저 캐시 및 `localStorage` 오염으로 인해 새로운 분석 세션(`post_id`)을 정상적으로 받아오지 못하는 이슈가 발생했습니다. 
* **해결:** 에러 발생 시 및 세션 만료 시 `localStorage.clear()` 로직을 명시적으로 추가하여 캐시를 안전하게 정리함으로써 문제를 완벽하게 해결했습니다.

### 🟢 MediaPipe 기반 그립 분석의 한계
* **문제:** 촬영 구도에 따라 MediaPipe의 성능 편차가 심해 일관된 진단을 제공하기 어려움
* **해결:** → 문제를 그립 분류로 재정의: 단순 각도 계산 방식 X, Yolo11n 모델 파인튜닝 후 객체 분류로 해결했습니다.

---

## 👨‍💻 7. 팀원 소개 및 역할 (Team)

| 이름 | 역할 분담 |
| :--- | :--- |
| **노은서** | 팀장/PM |
| **김민지** | 프론트엔드, 스윙 알고리즘 |
| **이원호** | 백엔드, LLM 종합 리포트 |
| **권주은** | 백엔드, 자세교정 알고리즘 |
| **한태호** | 셔틀콕 궤적 알고리즘, 그립 알고리즘 |





## 📌 8. 폴더 구조











## ✨ 9. demo 영상



### 9.1 그립 교정

<p align="center">
  <video src=https://github.com/user-attachments/assets/f5c024e4-2fc8-47ab-afb7-af6814f3000f autoplay loop muted playsinline width="80%"></video>
</p>


### 9.2 실시간 레슨 모드

<p align="center">
  <video src=https://github.com/user-attachments/assets/9aae412e-16f1-484b-815e-41b4d04210fe autoplay loop muted playsinline width="80%"></video>
</p>


### 9.3 초보자 영상 (65점) vs  전문가 영상 (90점)

<p align="center">
  <video src=https://github.com/user-attachments/assets/f73b143d-3ac7-4096-91c4-07d7e8f035d6 autoplay loop muted playsinline width="80%"></video>
</p



### 9.4 리포트 캘린더,성장 리포트

<p align="center">
  <video src=https://github.com/user-attachments/assets/f78e6aa3-2811-4aa9-ae94-401de94775ac autoplay loop muted playsinline width="80%"></video>
</p


### 9.5 마이페이지-탈퇴

<p align="center">
  <video src=https://github.com/user-attachments/assets/74bed9c2-a241-4f32-b0a7-c5dc92c738f6 autoplay loop muted playsinline width="80%"></video>
</p










