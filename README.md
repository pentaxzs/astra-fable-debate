# Astra × Fable Debate

GPT-6 Astra와 Claude Fable 5를 3라운드로 토론시키는 로컬 Streamlit 웹앱입니다.

## 기능

- 토론 주제 직접 입력
- Astra/Fable 역할 자유 지정
- ROUND 1: 독립 주장
- ROUND 2: 상대 주장 비판
- ROUND 3: 최종 반론
- 같은 라운드의 두 모델을 병렬 호출해 대기시간 단축
- 토론 결과를 Markdown으로 저장
- API Key는 앱 파일에 저장하지 않음

## 1. Python 확인

Python 3.10 이상을 권장합니다.

```bash
python3 --version
```

## 2. 설치

터미널에서 이 폴더로 이동한 뒤:

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열립니다. 보통 `http://localhost:8501`입니다.

## API Key 입력

가장 간단한 방법은 앱 왼쪽 사이드바에 OpenAI / Anthropic API Key를 직접 입력하는 것입니다.

환경변수를 쓰려면:

### macOS / Linux

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

## 기본 모델

- OpenAI: `gpt-6-astra`
- Anthropic: `claude-fable-5`

모델 접근 권한이 계정에 아직 열리지 않았다면 API에서 권한/모델 오류가 발생할 수 있습니다. 이 경우 사이드바의 모델 ID를 본인 계정에서 사용 가능한 모델로 바꿀 수 있습니다.

## 비용 주의

한 번의 토론에서 총 6회의 모델 API 호출이 발생합니다. 긴 응답일수록 API 비용이 증가하므로 처음에는 Fable `max_tokens`를 1,500~3,000 정도로 두는 것을 권장합니다.
