import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd

# 환경변수 로드
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(
    page_title="연세대 요람도우미 요람조람",
    page_icon="https://www.yonsei.ac.kr/sites/sc/images/sub/img-symbol6.png",
    layout="centered",
)


# 고정 PDF 경로
PDF_PATH = "YonseiUniversityCatalog.pdf"

# 지정된 웹페이지 링크 목록 (사용자가 직접 추가)
SPECIFIED_URLS = [
    "https://www.yonsei.ac.kr/sc/275/subview.do",
    "https://www.yonsei.ac.kr/sc/276/subview.do",
    "https://www.yonsei.ac.kr/sc/277/subview.do",
    "https://www.yonsei.ac.kr/sc/278/subview.do",
    "https://www.yonsei.ac.kr/sc/279/subview.do",
    "https://www.yonsei.ac.kr/sc/386/subview.do",
    "https://www.yonsei.ac.kr/sc/387/subview.do",
    "https://www.yonsei.ac.kr/sc/281/subview.do",
    "https://www.yonsei.ac.kr/sc/376/subview.do",
    "https://www.yonsei.ac.kr/sc/377/subview.do",
    "https://www.yonsei.ac.kr/sc/378/subview.do",
    "https://www.yonsei.ac.kr/sc/379/subview.do",
    "https://www.yonsei.ac.kr/sc/383/subview.do",
    "https://www.yonsei.ac.kr/sc/384/subview.do",
    "https://www.yonsei.ac.kr/sc/385/subview.do",
    "https://www.yonsei.ac.kr/sc/301/subview.do",
    "https://www.yonsei.ac.kr/sc/254/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGc2MlMkY1OCUyRjk0MjA3OCUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2ZmluZFR5cGUlM0QlMjZmaW5kV29yZCUzRCUyNmZpbmRDbFNlcSUzRCUyNmZpbmRPcG53cmQlM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2cGFzc3dvcmQlM0QlMjY%3D",
    "https://www.yonsei.ac.kr/sc/254/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGc2MlMkY1OCUyRjk0MjA2NCUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2ZmluZFR5cGUlM0QlMjZmaW5kV29yZCUzRCUyNmZpbmRDbFNlcSUzRCUyNmZpbmRPcG53cmQlM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2cGFzc3dvcmQlM0QlMjY%3D",
    "https://www.yonsei.ac.kr/sc/254/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGc2MlMkY1OCUyRjk0MjA2MiUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2ZmluZFR5cGUlM0QlMjZmaW5kV29yZCUzRCUyNmZpbmRDbFNlcSUzRCUyNmZpbmRPcG53cmQlM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2cGFzc3dvcmQlM0QlMjY%3D",
    "https://www.yonsei.ac.kr/sc/254/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGc2MlMkY1OCUyRjk0MjA2MSUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2ZmluZFR5cGUlM0QlMjZmaW5kV29yZCUzRCUyNmZpbmRDbFNlcSUzRCUyNmZpbmRPcG53cmQlM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2cGFzc3dvcmQlM0QlMjY%3D",
    "https://www.yonsei.ac.kr/sc/254/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGc2MlMkY1OCUyRjkyMzI1OCUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2ZmluZFR5cGUlM0RzaiUyNmZpbmRXb3JkJTNEJUVBJUI1JUIwJUVBJUIwJTk1JUVDJUEyJThDJTI2ZmluZENsU2VxJTNEJTI2ZmluZE9wbndyZCUzRCUyNnJnc0JnbmRlU3RyJTNEJTI2cmdzRW5kZGVTdHIlM0QlMjZwYXNzd29yZCUzRCUyNg%3D%3D",
    "https://www.yonsei.ac.kr/sc/254/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGc2MlMkY1OCUyRjk0MTIwMCUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2ZmluZFR5cGUlM0RzaiUyNmZpbmRXb3JkJTNEJUVBJUI1JUIwJUVBJUIwJTk1JUVDJUEyJThDJTI2ZmluZENsU2VxJTNEJTI2ZmluZE9wbndyZCUzRCUyNnJnc0JnbmRlU3RyJTNEJTI2cmdzRW5kZGVTdHIlM0QlMjZwYXNzd29yZCUzRCUyNg%3D%3D",
    "https://libart.yonsei.ac.kr/libart/degree/requirements_10.do",
    "https://computing.yonsei.ac.kr/sub3_1.php",
    "https://swedu.yonsei.ac.kr/yonseisw/swedu02.do",
    "https://swedu.yonsei.ac.kr/yonseisw/swedu01.do",
    "https://swedu.yonsei.ac.kr/yonseisw/swedu03.do",
    "https://universitycollege.yonsei.ac.kr/fresh/refinement/goal.do",
    "https://yicrc.yonsei.ac.kr/main/rc.asp?mid=m01_06",
    "https://yicrc.yonsei.ac.kr/main/rc.asp?mid=m01_04",
    "https://yicrc.yonsei.ac.kr/main/rc.asp?mid=m01_01",
    "https://www.yonsei.ac.kr/sc/285/subview.do",
    "https://www.yonsei.ac.kr/sc/286/subview.do",
    "https://www.yonsei.ac.kr/sc/388/subview.do",
    "https://ihei.yonsei.ac.kr/ihei/Program/program_whole.do",
    "https://ihei.yonsei.ac.kr/ihei/innovation/innovation_program.do",
    "https://oia.yonsei.ac.kr/partner/chStu.asp",
    "https://oia.yonsei.ac.kr/partner/chStu2.asp",
    "https://oia.yonsei.ac.kr/partner/chStu3.asp",
    "https://oia.yonsei.ac.kr/partner/chStu4.asp",
    "https://oia.yonsei.ac.kr/partner/chStu5.asp",
    "https://oia.yonsei.ac.kr/partner/chStu6.asp",
    "https://oia.yonsei.ac.kr/partner/chStu7.asp",
    "https://oia.yonsei.ac.kr/partner/chStu10.asp",
    "https://oia.yonsei.ac.kr/partner/chStu11.asp",
    "https://oia.yonsei.ac.kr/partner/grade.asp"
]
    


def load_pdf_docs(pdf_path: str):
    """pypdf로 PDF 문서를 읽어 LangChain Document 리스트로 변환"""
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": i + 1,
                    "source": pdf_path,
                    "source_type": "pdf",
                },
            )
        )
    return docs


def split_docs(docs):
    """재귀적 문자 텍스트 분할"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def extract_tables_from_html(html_content: str, base_url: str) -> str:
    """HTML에서 표를 추출하여 구조화된 텍스트로 변환"""
    soup = BeautifulSoup(html_content, 'html.parser')
    tables_text = []
    
    for i, table in enumerate(soup.find_all('table')):
        try:
            # pandas를 사용하여 표를 읽고 텍스트로 변환
            dfs = pd.read_html(str(table))
            if dfs:
                table_text = f"\n[표 {i+1}]\n"
                table_text += dfs[0].to_string(index=False)
                tables_text.append(table_text)
        except Exception as e:
            # pandas로 읽기 실패 시 BeautifulSoup으로 직접 추출
            table_text = f"\n[표 {i+1}]\n"
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                    table_text += row_text + '\n'
            if table_text.strip():
                tables_text.append(table_text)
    
    return '\n'.join(tables_text)


def extract_images_from_html(html_content: str, base_url: str) -> str:
    """HTML에서 이미지 및 인포그래픽 정보를 추출"""
    soup = BeautifulSoup(html_content, 'html.parser')
    images_info = []
    
    for img in soup.find_all('img'):
        img_src = img.get('src', '')
        img_alt = img.get('alt', '')
        img_title = img.get('title', '')
        
        # 상대 경로를 절대 경로로 변환
        if img_src:
            img_url = urljoin(base_url, img_src)
            
            img_info = f"[이미지: {img_alt or img_title or '이미지'}]"
            if img_alt:
                img_info += f" 설명: {img_alt}"
            if img_title and img_title != img_alt:
                img_info += f" 제목: {img_title}"
            images_info.append(img_info)
    
    return '\n'.join(images_info)


def load_web_docs(urls: list[str]):
    """웹페이지들을 로드하여 LangChain Document 리스트로 변환 (표와 이미지 포함)"""
    if not urls:
        return []
    
    docs = []
    failed_urls = []
    success_count = 0
    
    def load_single_url(url):
        try:
            # User-Agent 헤더 추가
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # HTML 원본 가져오기 (표와 이미지 추출용)
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            html_content = response.text
            
            # WebBaseLoader로 기본 텍스트 로드
            loader = WebBaseLoader(url)
            loaded_docs = loader.load()
            
            # 표 추출
            tables_text = extract_tables_from_html(html_content, url)
            
            # 이미지/인포그래픽 정보 추출
            images_text = extract_images_from_html(html_content, url)
            
            # 각 문서에 표와 이미지 정보 추가
            valid_docs = []
            for doc in loaded_docs:
                enhanced_content = doc.page_content
                
                # 빈 콘텐츠 필터링
                if not enhanced_content or len(enhanced_content.strip()) < 50:
                    continue
                
                # 표가 있으면 추가
                if tables_text:
                    enhanced_content += f"\n\n=== 표 정보 ===\n{tables_text}"
                
                # 이미지 정보가 있으면 추가
                if images_text:
                    enhanced_content += f"\n\n=== 이미지 정보 ===\n{images_text}"
                
                doc.page_content = enhanced_content
                doc.metadata.update({
                    "source": url,
                    "source_type": "web",
                    "content_length": len(doc.page_content),
                })
                valid_docs.append(doc)
            
            if not valid_docs:
                failed_urls.append((url, "Empty or too short content"))
            
            return valid_docs
        except Exception as e:
            error_msg = str(e)
            failed_urls.append((url, error_msg))
            print(f"Error loading {url}: {error_msg}")
            return []
    
    # 병렬로 웹페이지 로드 (속도 향상)
    with ThreadPoolExecutor(max_workers=3) as executor:  # 워커 수 조정 (서버 부하 방지)
        future_to_url = {executor.submit(load_single_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            try:
                loaded_docs = future.result(timeout=30)  # 타임아웃 추가
                if loaded_docs:
                    docs.extend(loaded_docs)
                    success_count += 1
            except Exception as e:
                url = future_to_url[future]
                failed_urls.append((url, f"Timeout or error: {str(e)}"))
                print(f"Error processing {url}: {e}")
    
    # 상세한 로그 출력
    print(f"\n{'='*70}")
    print(f"Web Loading Summary:")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {len(failed_urls)}")
    print(f"  Total Documents: {len(docs)}")
    
    if failed_urls:
        print(f"\nFailed URLs ({len(failed_urls)}):")
        for url, error in failed_urls[:10]:  # 처음 10개만 표시
            print(f"  ✗ {url}")
            print(f"    → {error[:80]}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more")
    print(f"{'='*70}\n")
    
    return docs


def format_docs(docs):
    """retriever 결과를 사람이 읽기 좋은 문자열로 포맷"""
    lines = []
    for d in docs:
        src_type = d.metadata.get("source_type", "unknown")
        page = d.metadata.get("page")
        prefix = f"[{src_type.upper()}]"
        if page:
            prefix += f"[p{page}]"
        lines.append(f"{prefix} {d.page_content}")
    return "\n\n".join(lines)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """임베딩 모델을 별도로 캐싱"""
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


@st.cache_resource(show_spinner=False)
def init_rag_pipeline(google_api_key: str, tavily_api_key: str):
    """PDF(1순위), 지정된 웹페이지(2순위), Tavily 웹검색(3순위)을 사용하는 RAG 파이프라인 초기화."""
    # 1) LLM & 임베딩
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=google_api_key,
    )
    # 임베딩 모델 (별도 캐싱)
    embeddings = get_embeddings()

    # 2) Tavily 검색 도구
    search = TavilySearch(
        max_results=3,
        include_answer=True,
        # 일반적으로 TAVILY_API_KEY 환경변수를 사용하므로
        # 여기서는 별도 api_key 인자를 넘기지 않음
    )

    # 3) PDF 벡터스토어 (디스크 캐싱)
    pdf_cache_dir = "vectorstore_cache_pdf"

    try:
        if os.path.exists(pdf_cache_dir):
            # 캐시가 있으면 PDF 로딩/청킹 스킵하고 바로 로드
            pdf_vs = FAISS.load_local(
                pdf_cache_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # 캐시가 없을 때만 PDF 로딩/청킹/벡터화 수행
            pdf_docs = load_pdf_docs(PDF_PATH)
            pdf_chunks = split_docs(pdf_docs)
            pdf_vs = FAISS.from_documents(pdf_chunks, embeddings)
            os.makedirs(pdf_cache_dir, exist_ok=True)
            pdf_vs.save_local(pdf_cache_dir)
    except Exception:
        # 캐시가 깨졌거나 버전이 달라서 로드 실패 시, 다시 생성
        pdf_docs = load_pdf_docs(PDF_PATH)
        pdf_chunks = split_docs(pdf_docs)
        pdf_vs = FAISS.from_documents(pdf_chunks, embeddings)
        os.makedirs(pdf_cache_dir, exist_ok=True)
        pdf_vs.save_local(pdf_cache_dir)

    pdf_retriever = pdf_vs.as_retriever(search_kwargs={"k": 6})

    # 4) 지정된 웹페이지 벡터스토어 (디스크 캐싱)
    web_cache_dir = "vectorstore_cache_web"
    
    try:
        if os.path.exists(web_cache_dir) and SPECIFIED_URLS:
            web_vs = FAISS.load_local(
                web_cache_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            # 웹페이지 로드 및 벡터화
            if SPECIFIED_URLS:
                web_docs = load_web_docs(SPECIFIED_URLS)
                if web_docs:
                    web_chunks = split_docs(web_docs)
                    web_vs = FAISS.from_documents(web_chunks, embeddings)
                    os.makedirs(web_cache_dir, exist_ok=True)
                    web_vs.save_local(web_cache_dir)
                else:
                    # 웹페이지 로드 실패 시 빈 벡터스토어 생성
                    web_vs = FAISS.from_texts([""], embeddings)
            else:
                # URL이 없으면 빈 벡터스토어 생성
                web_vs = FAISS.from_texts([""], embeddings)
    except Exception as e:
        print(f"Error loading web cache: {e}")
        # 재시도
        try:
            if SPECIFIED_URLS:
                web_docs = load_web_docs(SPECIFIED_URLS)
                if web_docs:
                    web_chunks = split_docs(web_docs)
                    web_vs = FAISS.from_documents(web_chunks, embeddings)
                    os.makedirs(web_cache_dir, exist_ok=True)
                    web_vs.save_local(web_cache_dir)
                else:
                    web_vs = FAISS.from_texts([""], embeddings)
            else:
                web_vs = FAISS.from_texts([""], embeddings)
        except Exception:
            web_vs = FAISS.from_texts([""], embeddings)

    web_retriever = web_vs.as_retriever(search_kwargs={"k": 6})

    # 5) Tavily 검색 → 문자열 컨텍스트로 변환
    def tavily_retrieve(question: str) -> str:
        try:
            result = search.invoke(question)
            return str(result)
        except Exception:
            return ""

    # 6) 프롬프트 (우선순위 규칙 명시)
    prompt = ChatPromptTemplate.from_template(
        """
당신은 연세대학교 요람 (대학 교과과정 안내서)에 대한 질문을 처리하는 어시스턴트로서,
연세대학교 요람(PDF), 지정된 웹페이지, 일반 웹검색 결과를 사용합니다.

정보 사용 우선순위는 다음과 같습니다.
1. PDF 요람 문서(YonseiUniversityCatalog.pdf)의 내용
2. 지정된 웹페이지의 내용
3. Tavily 웹검색 결과

규칙:
- 먼저 PDF 컨텍스트를 가장 신뢰하고, 모순되는 정보가 있을 경우 PDF 내용을 우선합니다.
- PDF와 지정된 웹페이지 모두에 관련 내용이 없을 때만 Tavily 웹검색 컨텍스트를 보조로 사용합니다.
- 세 컨텍스트 어디에도 관련 정보가 없지만, 요람이나 교과과정, 학사정보 등에 대해 물어보는 것은 맞다고 판단되면,
  "아래 내용은 내장 정보에서 찾지 못한 내용이므로 신뢰하기 어렵습니다. 단순 참고용으로만 활용하세요."라고 말한 뒤,
  연세대학교에 한해 물어봤다는 전제 하에 답변을 생성합니다.
- 질문이 요람이나 교과과정, 학사정보 등에 대해 물어보는 것이 아니라고 판단되면, "교과정보 및 학사정보에 관한 질문인지 다시 한 번 확인해주세요."라고 답변합니다.
- 모든 답변은 한국어로 작성하세요.

[PDF CONTEXT]
{pdf_context}

[WEB CONTEXT]
{web_context}

[SEARCH CONTEXT]
{search_context}

[질문]
{question}
"""
    )

    # 7) 최종 질의 → 답변 함수
    # 체인 방식 사용 (더 안정적)
    chain = prompt | llm | StrOutputParser()
    
    def answer(question: str) -> str:
        try:
            # 질문 재구성 프롬프트
            rewrite_prompt = ChatPromptTemplate.from_template(
                """
                당신은 연세대학교 요람(대학 교과과정 안내서)에 대한 질문을 처리하는 어시스턴트입니다.
                
                사용자의 질문이 요람 문서에서 답변하기에 명확하지 않거나 모호한 경우, 
                요람 문서의 맥락에 맞게 질문을 더 구체적이고 명확하게 재구성해주세요.
                
                규칙:
                - 원래 질문의 의도를 최대한 보존하세요
                - 요람 문서에서 찾을 수 있는 정보 유형(학과, 과목, 학점, 졸업요건 등)에 맞게 용어를 재구성하세요
                - 연세대학교 관련 맥락을 명확히 하세요
                - 질문이 모호하다면 연세대학교 요람 문서의 구조(학과별, 과목별 등)에 맞게 구체화하세요
                - 질문이 이미 명확하면 그대로 반환하세요
                
                원래 질문: {question}
                
                재구성된 질문:
                """
            )
            
            rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            refined_question = rewrite_chain.invoke({"question": question})
            
            # 1순위: PDF (보강된 질문으로 검색)
            pdf_docs_rel = pdf_retriever.invoke(refined_question)
            pdf_context = format_docs(pdf_docs_rel)
            
            # 2순위: 지정된 웹페이지 (보강된 질문으로 검색)
            web_docs_rel = web_retriever.invoke(refined_question)
            web_context = format_docs(web_docs_rel)
            
            # 3순위: Tavily 웹검색
            search_context = tavily_retrieve(question)
            
            # 체인을 사용하여 답변 생성 (더 안정적)
            result = chain.invoke({
                "pdf_context": pdf_context,
                "web_context": web_context,
                "search_context": search_context,
                "question": question,
            })
            return result
            
        except Exception as e:
            # 상세한 오류 정보 반환 (디버깅용)
            import traceback
            error_msg = f"답변 생성 중 오류 발생: {str(e)}\n\n{traceback.format_exc()}"
            return error_msg

    return answer


# 환경 변수에서 키 가져오기 (환경변수 또는 Streamlit Secrets 지원)
try:
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    tavily_api_key = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY", None)
except Exception:
    # Streamlit Secrets가 없는 경우 (로컬 환경)
    api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    st.error("Error: GOOGLE_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
elif not tavily_api_key:
    st.error("Error: TAVILY_API_KEY not found. Please set it in environment variables or Streamlit secrets.")
else:
    try:
        # RAG 파이프라인이 아직 준비 안 되었으면, 먼저 로딩 전용 화면을 보여줍니다.
        if "rag_qa" not in st.session_state:
            # 화면 전체를 덮는 로딩 오버레이 표시
            st.markdown(
                """<style>
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: #f5f7fb;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    z-index: 9999;
}
.loading-overlay img {
    width: 220px;
    max-width: 60vw;
}
.loading-overlay .loading-title {
    font-size: 1.9rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
    color: #0f1c3f;
}
.loading-overlay .loading-subtitle {
    font-size: 1rem;
    color: #4a4a4a;
    margin-bottom: 1.5rem;
    max-width: 420px;
}
.loading-overlay .loading-indicator {
    font-size: 1rem;
    color: #0f62fe;
    letter-spacing: 0.1rem;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0% { opacity: 0.3; }
    50% { opacity: 1; }
    100% { opacity: 0.3; }
}
</style>
<div class="loading-overlay">
    <img src="https://www.yonsei.ac.kr/sites/sc/images/sub/img-sig8.png" alt="Yonsei Symbol" />
    <div class="loading-title">연세대학교 요람도우미 요람조람</div>
    <div class="loading-subtitle">요람 PDF를 읽고 검색 인덱스를 준비하는 중입니다. <br>잠시만 기다려 주세요.</div>
    <div class="loading-indicator">LOADING...</div>
</div>""",
                unsafe_allow_html=True,
            )

            # 리소스를 초기화하고 세션에 저장
            rag_qa = init_rag_pipeline(api_key, tavily_api_key)
            st.session_state["rag_qa"] = rag_qa

            # 초기화가 끝났으니 전체 화면을 다시 그립니다.
            st.rerun()

        # 여기부터는 rag_qa가 이미 준비된 상태
        rag_qa = st.session_state["rag_qa"]

        # 상단 헤더
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(
                "https://www.yonsei.ac.kr/sites/sc/images/sub/img-symbol6.png", width=75
            )
        with col2:
            st.title("연세대 요람도우미 :blue[요람조람]")
            st.markdown("#### 우리대학 요람을 요목조목! \n요람 및 교과정보, 전공 및 학사정보에 실시간 답변해주는 연세대 전용 챗봇입니다.")

        # 사이드바 고정 너비
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                width: 365px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # 사이드바
        with st.sidebar:
            st.header("💬 대화 관리\n")
            st.header("")

            if st.button("🔄 대화 초기화", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

            st.markdown("---")
            st.markdown(
                '''
            **🔍 이 챗봇은:**
            - 연세대 요람PDF 및 홈페이지 기반 답변 📘
            - 필요한 전공, 과목, 교과이수, 각종 제도 등에 관한 모든 질문 답변 가능 💡
            - 상담센터의 답변을 기다릴 필요없이 약 10초만에 실시간 답변 ⏱️

            **✍️ 질문 가이드:**
            - 학사일정, 과목별 개설여부 및 일정이나 교수님 정보와 같이
              학기별로 변동되는 정보는 답변이 어렵습니다.
            - 과목 목록을 알고 싶을 때에는 표를 요청하면 좋습니다.
              예시) "xx과 전공수업 목록을 표로 정리해줘."
            - 질문은 명료하고 구체적일수록 좋습니다.
            '''
            )

        # Session State 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 지금까지 대화 출력
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                avatar = "👤"
            else:
                avatar = "https://www.yonsei.ac.kr/sites/sc/images/sub/img-symbol6.png"

            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

        # 사용자 입력
        user_input = st.chat_input("메시지를 입력하세요")

        if user_input:
            # 사용자 메시지 저장/표시
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user", avatar="👤"):
                st.write(user_input)

            # RAG 기반 답변 생성
            with st.chat_message(
                "assistant",
                avatar="https://www.yonsei.ac.kr/sites/sc/images/sub/img-symbol6.png",
            ):
                with st.spinner("생각 중..."):
                    try:
                        assistant_message = rag_qa(user_input)
                    except Exception as e:
                        assistant_message = f"답변 생성 중 오류가 발생했습니다: {e}"
                    st.write(assistant_message)

            # assistant 메시지 저장
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_message}
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")