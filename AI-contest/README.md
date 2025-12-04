# 연세대 요람도우미 요람조람

연세대학교 요람(PDF) 및 홈페이지 기반으로 교과정보와 학사정보를 실시간으로 답변해주는 챗봇입니다.

## 주요 기능

- 📘 연세대 요람 PDF 및 홈페이지 기반 답변
- 💡 전공, 교양수업, 교과이수 등에 관한 질문 답변
- ⏱️ 약 10초만에 실시간 답변
- 🔍 표와 인포그래픽 정보도 추출하여 활용

## 환경 설정

### 1. 저장소 클론

```bash
git clone <repository-url>
cd AI-contest
```

### 2. 가상환경 설정 (선택사항)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. API 키 설정

#### 방법 1: 환경변수 파일 (.env) 사용 (로컬 개발)

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 내용을 입력하세요:

```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**API 키 발급 방법:**
- **Google Generative AI**: [Google AI Studio](https://makersuite.google.com/app/apikey)에서 발급
- **Tavily Search**: [Tavily](https://tavily.com/)에서 발급

#### 방법 2: 환경변수 직접 설정

Windows (PowerShell):
```powershell
$env:GOOGLE_API_KEY="your_key_here"
$env:TAVILY_API_KEY="your_key_here"
```

Windows (CMD):
```cmd
set GOOGLE_API_KEY=your_key_here
set TAVILY_API_KEY=your_key_here
```

Mac/Linux:
```bash
export GOOGLE_API_KEY="your_key_here"
export TAVILY_API_KEY="your_key_here"
```

#### 방법 3: Streamlit Cloud 배포 시

Streamlit Cloud의 Secrets 기능을 사용하세요:

1. Streamlit Cloud 대시보드에서 앱 선택
2. **Settings** → **Secrets**
3. 다음 형식으로 추가:

```toml
GOOGLE_API_KEY = "your_actual_key"
TAVILY_API_KEY = "your_actual_key"
```

### 5. 실행

```bash
streamlit run catchat.py
```

브라우저에서 자동으로 열리며, 첫 실행 시 PDF와 웹페이지 인덱싱에 시간이 걸릴 수 있습니다.

## 프로젝트 구조

```
AI-contest/
├── catchat.py                 # 메인 애플리케이션
├── requirements.txt           # 패키지 의존성
├── pyproject.toml            # 프로젝트 설정
├── .env                      # 환경변수 파일 (로컬 전용, Git에 업로드 안 됨)
├── .env.example              # 환경변수 템플릿
├── .gitignore                # Git 제외 파일 목록
├── YonseiUniversityCatalog.pdf  # 요람 PDF 파일
├── vectorstore_cache_pdf/    # PDF 벡터스토어 캐시
└── vectorstore_cache_web/    # 웹페이지 벡터스토어 캐시
```

## 주요 기술 스택

- **Streamlit**: 웹 인터페이스
- **LangChain**: RAG 파이프라인 구성
- **Google Generative AI (Gemini)**: LLM
- **Tavily**: 웹 검색
- **FAISS**: 벡터 검색
- **BeautifulSoup4**: HTML 파싱
- **Pandas**: 표 데이터 처리

## 사용 방법

1. 애플리케이션 실행 후 질문을 입력하세요
2. 예시 질문:
   - "컴퓨터과학과 전공 수업 목록을 알려줘"
   - "졸업 요건이 뭐야?"
   - "교양 수업 이수 기준은?"

## 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요 (이미 `.gitignore`에 추가됨)
- API 키는 외부에 노출하지 마세요
- 첫 실행 시 PDF와 웹페이지 로딩에 시간이 걸릴 수 있습니다
- 학사일정, 과목별 개설여부 등 학기별로 변동되는 정보는 답변이 어려울 수 있습니다

## 라이선스

이 프로젝트는 교육 목적으로 만들어졌습니다.

