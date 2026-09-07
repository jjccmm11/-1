import os
import re
import pandas as pd
from datetime import datetime, timedelta
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Groq API 키 설정
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY 환경변수를 설정한 뒤 실행하세요.")

# 2. 한국어 지원 초고해상도 임베딩 모델 지정
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# =========================================================================
# [대혁신] 듀얼 데이터베이스 아키텍처 구축 (TXT 공간과 XLSX 공간의 격리 및 전문화)
# =========================================================================

# (A) 학교 안내 가이드북 전용 데이터베이스 빌드
txt_chunks = []
if os.path.exists("school_info.txt"):
    txt_loader = TextLoader("school_info.txt", encoding="utf-8")
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60, separators=["\n\n", "\n", "."])
    txt_chunks = txt_splitter.split_documents(txt_loader.load())

# 안내서 정보가 비어있지 않다면 전용 DB 빌드
if txt_chunks:
    info_vectorstore = Chroma.from_documents(documents=txt_chunks, embedding=embeddings, collection_name="school_info")
    info_retriever = info_vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    info_retriever = None

# (B) 급식 식단표 전용 데이터베이스 빌드
menu_documents = []
if os.path.exists("school_menu.xlsx"):
    print("📊 [인프라] school_menu.xlsx 급식 데이터를 정밀 정형화하는 중...")
    df = pd.read_excel("school_menu.xlsx", dtype=str)
    
    for idx, row in df.iterrows():
        raw_date = str(row['날짜']).strip()
        try:
            date_obj = datetime.strptime(raw_date, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%Y년 %m월 %d일")
            search_date_key = date_obj.strftime("%Y-%m-%d")
        except:
            formatted_date = raw_date
            search_date_key = raw_date
            
        lunch_menu = str(row['점심']).strip() if pd.notna(row['점심']) else "급식없음"
        dinner_menu = str(row['석식']).strip() if pd.notna(row['석식']) else "급식없음"
        note = str(row['비고']).strip() if pd.notna(row['비고']) else "없음"
        
        # 날짜와 매칭 키워드를 중복 배치하여 유사도 스코어 부스팅을 유도합니다.
        lunch_sentence = f"일자코드: {search_date_key} 날짜정보: {formatted_date} 요일: {row['요일']} 시간구분: 점심 중식 급식 메뉴 식단 정보는 [{lunch_menu}] 입니다. 특이사항: {note}"
        dinner_sentence = f"일자코드: {search_date_key} 날짜정보: {formatted_date} 요일: {row['요일']} 시간구분: 저녁 석식 급식 메뉴 식단 정보는 [{dinner_menu}] 입니다. 특이사항: {note}"
        
        menu_documents.append(Document(page_content=lunch_sentence, metadata={"date": search_date_key, "type": "lunch"}))
        menu_documents.append(Document(page_content=dinner_sentence, metadata={"date": search_date_key, "type": "dinner"}))

if menu_documents:
    menu_vectorstore = Chroma.from_documents(documents=menu_documents, embedding=embeddings, collection_name="school_menu")
    menu_retriever = menu_vectorstore.as_retriever(search_kwargs={"k": 3})
else:
    menu_retriever = None

# =========================================================================
# [핵심 보정] 전천후 시계열 질의 추적 및 스마트 컨텍스트 라우팅 엔진 함수
# =========================================================================
def intelligent_router(inputs):
    user_query = inputs["question"]
    current_date_str = inputs["current_date"]
    
    # 1. 시계열 질문일 경우 쿼리 가중치 분석 및 확장 정밀 보정
    enhanced_menu_query = user_query
    
    try:
        match = re.search(r"(\d{4}년 \d{2}월 \d{2}일)", current_date_str)
        if match:
            date_clean_str = match.group(1)
            today_obj = datetime.strptime(date_clean_str, "%Y년 %m월 %d일")
            
            # (계산 A) '오늘' 컨텍스트 매핑
            if "오늘" in user_query:
                enhanced_menu_query += f" {today_obj.strftime('%Y-%m-%d')} {today_obj.strftime('%Y년 %m월 %d일')}"
            
            # (계산 B) '내일' 컨텍스트 매핑 (+1일)
            if "내일" in user_query:
                tomorrow_obj = today_obj + timedelta(days=1)
                enhanced_menu_query += f" {tomorrow_obj.strftime('%Y-%m-%d')} {tomorrow_obj.strftime('%Y년 %m월 %d일')}"
                
            # (계산 C) [신규 보정] '어제' 컨텍스트 매핑 (-1일 연산 수식 가동)
            if "어제" in user_query:
                yesterday_obj = today_obj - timedelta(days=1)
                enhanced_menu_query += f" {yesterday_obj.strftime('%Y-%m-%d')} {yesterday_obj.strftime('%Y년 %m월 %d일')}"
            
            # (계산 D) [신규 보정] "9월 11일" 또는 "9/11" 같은 특정 날짜 형태 입력 시 숫자 추출 매핑
            # 유저가 년도를 생략하더라도 현재 구동중인 시스템의 연도(YYYY)를 조합해 강제 싱크를 맞춥니다.
            date_digits = re.findall(r"(\d+)월\s*(\d+)일", user_query)
            if date_digits:
                m, d = int(date_digits[0][0]), int(date_digits[0][1])
                target_obj = datetime(year=today_obj.year, month=m, day=d)
                enhanced_menu_query += f" {target_obj.strftime('%Y-%m-%d')} {target_obj.strftime('%Y년 %m월 %d일')}"
    except Exception as e:
        pass
        
    if "점심" in user_query or "중식" in user_query:
        enhanced_menu_query += " 점심 중식 급식 메뉴 식단"
    if "저녁" in user_query or "석식" in user_query:
        enhanced_menu_query += " 저녁 석식 급식 메뉴 식단"

    # 2. 질문의 목적지에 따라 지식을 추출할 최적의 전문 방(DB 시트)을 결정하는 라우팅 메커니즘
    contexts = []
    
    # 질문에 급식, 점심, 저녁, 밥, 석식 메뉴 명사가 포함되어 있다면 급식 전용 DB 개설
    if any(keyword in user_query for keyword in ["급식", "점심", "중식", "저녁", "석식", "밥", "메뉴", "식단"]) and menu_retriever:
        print(f"🎯 [라우터 가동 - 급식 전용 매핑]: '{user_query}' -> '{enhanced_menu_query}'")
        menu_docs = menu_retriever.invoke(enhanced_menu_query)
        contexts.append("\n".join(doc.page_content for doc in menu_docs))
        
    # 그 외의 질문이거나 축제, 교칙, 동아리 등 가이드북 명사일 경우 가이드북 전용 DB 다이렉트 개설
    if not contexts or any(keyword in user_query for keyword in ["축제", "교칙", "동아리", "행사", "안내", "학교", "번호", "위치"]):
        if info_retriever:
            print(f"📚 [라우터 가동 - 가이드북 전용 매핑]: '{user_query}'")
            info_docs = info_retriever.invoke(user_query)
            contexts.append("\n".join(doc.page_content for doc in info_docs))
            
    return "\n\n".join(contexts)

# 3. 최신 공식 권장 글로벌 플래그십 대체 모델 엔드포인트 세팅 (400, 404 에러 원천 해결)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

system_prompt_template = (
    "너는 우리 학교 전문 안내 AI 비서야. 너의 이름은 살레봇이야. "
    "반드시 제공된 [학교 지식 데이터베이스 추출 컨텍스트] 내용에만 엄격히 기반해서 '사실 그대로' 한국어로 답변해줘.\n\n"
    " [현재 시스템 기준 시계열 정보]\n"
    "{current_date}\n\n"
    " [학교 지식 데이터베이스 추출 컨텍스트]\n"
    "{context}\n\n"
    "지침:\n"
    "1. 학교 기본 안내 정보(교칙, 축제, 동아리 등)에 대해 물어보면 지식베이스 팩트 텍스트 내용을 근거로 정확한 날짜만 대답하고 지어내지마.\n"
    "2. 급식 메뉴를 답변할 때는 사용자가 질문한 날짜 정보(오늘, 내일, 어제 또는 특정 월일)와 컨텍스트의 일자 정보가 완벽히 일치하는지 재차 확인하고 대답해줘.\n"
    "3. 지식 데이터베이스 추출 컨텍스트 내에 질문과 매칭되는 정확한 식단이나 텍스트 정보가 확인되지 않는다면, 절대 가짜 정보를 꾸며내지 말고 '제공된 학교 문서에 관련 정보가 확인되지 않습니다'라고 정중하게 답변해."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("human", "{question}"),
])

# 4. 데이터 파이프라인 결합
rag_chain = (
    {
        "context": intelligent_router, 
        "question": lambda x: x["question"],
        "current_date": lambda x: x["current_date"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 인터랙티브 대화형 살레봇 엔진 구동
print("\n" + "="*60)
print("🤖 [시스템] 듀얼 인프라 라우터 통합 완료 - 살레봇 최종 완전판 시동 성공")
print("종료하려면 '종료' 또는 'q'를 입력하세요.")
print("="*60 + "\n")

while True:
    user_question = input("🙋 질문을 입력하세요: ").strip()
    
    if user_question.lower() in ['종료', 'exit', 'quit', 'q']:
        print("🤖 살레봇 가동을 해제합니다. 수고하셨습니다!")
        break
        
    if not user_question:
        continue
        
    now = datetime.now()
    weekday_list = ['월', '화', '수', '목', '금', '토', '일']
    current_date_str = f"현재 일시: {now.strftime('%Y년 %m월 %d일')} ({weekday_list[now.weekday()]}요일)"
    
    print(f"🕒 [클럭 틱 패킷 생성]: {current_date_str}")
    
    try:
        response = rag_chain.invoke({
            "question": user_question,
            "current_date": current_date_str
        })
        print(f"\n🤖 살레봇 답변:\n{response}")
        print("\n" + "-"*60 + "\n")
        
    except Exception as e:
        print(f"❌ 런타임 예외 에러 발생: {e}")
        print("질문을 다시 입력해 주세요.\n")
