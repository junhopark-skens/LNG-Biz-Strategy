# GenCast Weather Forecasting Demo

Google DeepMind의 **GenCast** AI 기상 예측 모델 데모 리포지토리입니다.

## 📖 GenCast란?

GenCast는 Google DeepMind가 개발한 최첨단 AI 기상 예측 모델로, 기존의 ECMWF ENS 시스템 대비 97.2%의 예측 시나리오에서 더 높은 정확도를 보입니다.

- **고해상도 예측**: 0.25° 및 1.0° 해상도 지원
- **앙상블 예측**: 50개 이상의 확률론적 예측 생성
- **장기 예보**: 최대 15일 전 일기예보
- **빠른 처리**: TPU v5 사용 시 15일 예측을 8분 내 완료

## 📂 리포지토리 구성

이 리포지토리는 두 가지 GenCast 데모 노트북을 제공합니다:

### 1. `gencast_mini_demo.ipynb` ⭐ 추천
- **무료 Google Colab**에서 실행 가능
- TPUv2-8 무료 사용
- GenCast Mini 모델 (1.0° 해상도)
- 비용 없이 바로 체험 가능

### 2. `gencast_demo_cloud_vm.ipynb`
- Google Cloud TPU VM 필요 (유료)
- 전체 GenCast 모델 지원 (0.25° 및 1.0°)
- 프로덕션 수준의 성능

## 🚀 빠른 시작 (GenCast Mini)

### 1단계: Colab에서 노트북 열기

가장 빠르고 쉬운 방법은 공식 Colab 링크를 사용하는 것입니다:

👉 **[GenCast Mini Demo - Colab에서 열기](https://colab.research.google.com/github/deepmind/graphcast/blob/master/gencast_mini_demo.ipynb)**

또는 이 리포지토리의 노트북을 사용:
```bash
# 노트북 다운로드
wget https://raw.githubusercontent.com/junhopark-skens/LNG-Biz-Strategy/main/gencast_mini_demo.ipynb
```

### 2단계: TPU 런타임 설정

Colab에서:
1. **런타임 (Runtime)** → **런타임 유형 변경 (Change runtime type)**
2. **하드웨어 가속기 (Hardware accelerator)** → **TPU** 선택
3. **저장 (Save)** 클릭

### 3단계: 노트북 실행

노트북의 셀을 순서대로 실행하면 됩니다. 주요 단계:
1. 패키지 설치 및 import
2. 모델 및 데이터 로드
3. 날씨 예측 실행 (앙상블 생성)
4. 결과 시각화 및 분석

## 📊 결과 해석

GenCast는 다음과 같은 결과를 생성합니다:

- **Predictions**: AI 모델의 날씨 예측 (앙상블 멤버별)
- **Targets**: 실제 관측 데이터 (정답)
- **Diff**: 예측과 실제의 차이
- **Ensemble Mean**: 앙상블 평균 (가장 신뢰도 높은 예측)
- **CRPS**: 예측 정확도 지표 (낮을수록 좋음)

**상세한 결과 해석 방법은 [결과해석가이드.md](결과해석가이드.md)를 참고하세요.**

## 🌡️ 주요 날씨 변수

| 변수명 | 설명 | 단위 |
|--------|------|------|
| `2m_temperature` | 지상 2m 높이 기온 | K (켈빈) |
| `geopotential` | 지오포텐셜 고도 (500hPa) | m²/s² |
| `mean_sea_level_pressure` | 해수면 기압 | Pa |
| `10m_u/v_component_of_wind` | 지상 10m 바람 | m/s |
| `total_precipitation_12hr` | 12시간 누적 강수량 | m |

## 💻 Google Cloud TPU VM 설정 (전체 모델용)

전체 GenCast 모델을 실행하려면 Google Cloud TPU VM이 필요합니다:

```bash
# GCP 프로젝트 설정
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]

# TPU VM 생성
gcloud compute tpus tpu-vm create gencast-tpu \
  --zone=us-central1-a \
  --accelerator-type=v5litepod-8 \
  --version=tpu-ubuntu2204-base

# TPU VM 접속
gcloud compute tpus tpu-vm ssh gencast-tpu --zone=us-central1-a
```

자세한 설정 방법은 [DeepMind GraphCast 공식 문서](https://github.com/google-deepmind/graphcast/blob/main/docs/cloud_vm_setup.md)를 참고하세요.

## 📋 시스템 요구사항

### GenCast Mini
- Google Colab 무료 티어
- TPUv2-8 (Colab 무료 제공)
- 별도의 설치 불필요

### 전체 GenCast 모델

| 모델 | 시스템 메모리 | vRAM (GPU) | TPU |
|------|--------------|------------|-----|
| GenCast 0.25° | ~300GB | ~60GB | TPU v5 권장 |
| GenCast 1.0° | ~24GB | ~16GB | TPU v2/v5 |

## 🔗 참고 자료

- [GenCast 논문 (Nature)](https://www.nature.com/articles/s41586-024-08252-9)
- [Google DeepMind 블로그](https://deepmind.google/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/)
- [GraphCast GitHub (공식)](https://github.com/google-deepmind/graphcast)
- [ECMWF ai-models-gencast](https://github.com/ecmwf-lab/ai-models-gencast)

## ⚠️ 문제 해결

### "Only interpret mode is supported on CPU backend" 오류
→ Colab 런타임을 **TPU**로 변경하세요.

### 메모리 부족 오류
→ GenCast Mini 모델을 사용하거나, Google Cloud TPU VM을 사용하세요.

### 데이터 다운로드 실패
→ 노트북의 데이터 경로가 올바른지 확인하세요.

## 📝 라이선스

이 리포지토리의 노트북은 DeepMind Technologies Limited의 Apache License 2.0 하에 제공됩니다.

## 🙋 기여 및 문의

- Issues: GitHub Issues 탭 사용
- Pull Requests: 환영합니다!

---

**Last Updated**: 2025-12-01
