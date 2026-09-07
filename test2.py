import os
import re
import calendar
from datetime import datetime, timedelta
from langchain_core.documents import Document  # 원문 보존을 위한 객체
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

# 2. 한국어 특화 고밀도 인덱싱 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# 3. 데이터 로드 및 하루 단위 문장 원형 보존 파이프라인
final_chunks = []

if os.path.exists("school_info.txt"):
    print("📝 [시스템] school_info.txt 원문 유실 방지 독립 인덱싱 중...")
    with open("school_info.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # "ID-01번지식: 2026년 09월 01일 ..." 처럼 날짜 앞에 접두사가 붙는 경우까지
    # 대응하기 위해 줄 맨 앞 고정(re.match)이 아니라 줄 전체에서 날짜 패턴을 검색(re.search)합니다.
    current_section = ""
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue

        # 날짜(YYYY년 MM월 DD일)가 줄 어디에든 포함된 급식 문장은
        # 분할기를 거치지 않고 통째로 1개의 문서 객체로 만듭니다.
        if re.search(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일", line_strip):
            final_chunks.append(Document(page_content=line_strip, metadata={"source": "school_info.txt", "type": "menu"}))
        else:
            current_section += line_strip + "\n"
            if len(current_section) > 300:
                final_chunks.append(Document(page_content=current_section.strip(), metadata={"source": "school_info.txt", "type": "info"}))
                current_section = ""

    if current_section.strip():
        final_chunks.append(Document(page_content=current_section.strip(), metadata={"source": "school_info.txt", "type": "info"}))

# =========================================================================
# [핵심 수정] 날짜(급식) 문서에 대한 "정확 매칭용" 사전(dict) 구축
# 벡터 유사도는 "9월 11일"과 "9월 12일"을 거의 구분하지 못하기 때문에,
# 날짜가 명확한 질문은 이 dict로 100% 정확히 찾아냅니다.
# =========================================================================
menu_lookup = {}
for doc in final_chunks:
    if doc.metadata.get("type") == "menu":
        # "ID-01번지식: 2026년 09월 01일 ..." 처럼 날짜 앞에 접두사가 붙어도
        # 찾아낼 수 있도록 줄 전체에서 날짜 패턴을 검색합니다.
        m = re.search(r"(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)", doc.page_content)
        if m:
            # 표기 편차(09월 vs 9월, 공백 등)를 없애기 위해 key를 정규화
            key = re.sub(r"\s+", "", m.group(1))
            key = re.sub(r"(\d{4})년(\d{1,2})월(\d{1,2})일",
                          lambda x: f"{x.group(1)}년{int(x.group(2)):02d}월{int(x.group(3)):02d}일",
                          key)
            menu_lookup.setdefault(key, []).append(doc.page_content)

print(f"📅 [시스템] 급식 문서 {len(menu_lookup)}일치 정확매칭 인덱스 구축 완료")
if menu_lookup:
    sorted_keys = sorted(menu_lookup.keys())
    print(f"   ↳ 최소 날짜: {sorted_keys[0]} / 최대 날짜: {sorted_keys[-1]}")
else:
    print("   ⚠️ 급식(menu) 타입 문서가 하나도 인덱싱되지 않았습니다. school_info.txt의 날짜 줄 포맷을 확인하세요.")


def format_key(dt: datetime) -> str:
    return dt.strftime("%Y년%m월%d일")

# 4. 가상 인메모리 단일 Vector DB 빌드 (일반 안내정보 / 폴백용)
vectorstore = Chroma.from_documents(documents=final_chunks, embedding=embeddings, collection_name="salebot_perfect_mono")
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =========================================================================
# [시계열 정밀 추적] 자릿수 패딩 및 다중 날짜 포맷 교정 + 정확매칭 우선 조회
# =========================================================================
def search_context(inputs):
    user_query = inputs["question"]
    current_date_str = inputs["current_date"]

    enhanced_query = user_query
    target_dt_objs = []  # 이번 질문에서 파싱된 정확한 날짜(datetime)들. dict lookup용
    is_range_query = False

    try:
        match_today = re.search(r"(\d{4}년 \d{2}월 \d{2}일)", current_date_str)
        if match_today:
            date_clean_str = match_today.group(1)
            today_obj = datetime.strptime(date_clean_str, "%Y년 %m월 %d일")

            # ---------------------------------------------------------
            # (-1) [최최우선] "O월 전체/모두/전부", "이번달", "다음달",
            # "지난달" 같은 "한 달 통째" 질문 감지
            # ---------------------------------------------------------
            month_all_target = None  # (year, month) 튜플

            month_all_match = re.search(r"(\d{1,2})\s*월\s*(전체|모두|모든|전부|다)", user_query)
            if month_all_match:
                mm = int(month_all_match.group(1))
                # 지금 물어본 달이 올해 안 지난 경우 vs 이미 지난 경우 등은 단순화하여 올해 기준으로 처리
                month_all_target = (today_obj.year, mm)
            elif "이번달" in user_query or "이번 달" in user_query:
                month_all_target = (today_obj.year, today_obj.month)
            elif "다음달" in user_query or "다음 달" in user_query:
                ny, nm = (today_obj.year + 1, 1) if today_obj.month == 12 else (today_obj.year, today_obj.month + 1)
                month_all_target = (ny, nm)
            elif "지난달" in user_query or "지난 달" in user_query or "저번달" in user_query or "저번 달" in user_query:
                py, pm = (today_obj.year - 1, 12) if today_obj.month == 1 else (today_obj.year, today_obj.month - 1)
                month_all_target = (py, pm)

            if month_all_target:
                ty, tm = month_all_target
                last_day = calendar.monthrange(ty, tm)[1]
                start_obj = datetime(year=ty, month=tm, day=1)
                end_obj = datetime(year=ty, month=tm, day=last_day)
                cur = start_obj
                while cur <= end_obj:
                    target_dt_objs.append(cur)
                    cur += timedelta(days=1)
                is_range_query = True

            # ---------------------------------------------------------
            # (0) 날짜 "범위" 질문 감지: "9월 1일부터 5일까지",
            # "9월 1일부터 9월 5일까지", "9월1일~9월5일" 등
            # (위에서 월 전체 질문으로 이미 처리됐으면 건너뜀)
            # ---------------------------------------------------------
            range_pattern = r"(\d{1,2})\s*월\s*(\d{1,2})\s*일\s*(?:부터|에서|~|-)\s*(?:(\d{1,2})\s*월\s*)?(\d{1,2})\s*일"
            range_match = None if is_range_query else re.search(range_pattern, user_query)
            if range_match:
                sm, sd, em, ed = range_match.groups()
                sm = int(sm)
                sd = int(sd)
                em = int(em) if em else sm
                ed = int(ed)
                start_obj = datetime(year=today_obj.year, month=sm, day=sd)
                end_obj = datetime(year=today_obj.year, month=em, day=ed)
                if end_obj < start_obj:
                    # "12월 30일부터 1월 3일까지"처럼 해가 넘어가는 경우 보정
                    end_obj = datetime(year=today_obj.year + 1, month=em, day=ed)
                cur = start_obj
                while cur <= end_obj:
                    target_dt_objs.append(cur)
                    cur += timedelta(days=1)
                is_range_query = True

            # ---------------------------------------------------------
            # (1) 범위가 아닐 때만 개별 날짜 표현 파싱
            # ---------------------------------------------------------
            if not is_range_query:
                if "오늘" in user_query:
                    target_dt_objs.append(today_obj)
                if "내일" in user_query:
                    target_dt_objs.append(today_obj + timedelta(days=1))
                if "어제" in user_query:
                    target_dt_objs.append(today_obj - timedelta(days=1))

                m_after = re.search(r"(\d+)\s*일\s*후", user_query)
                if m_after:
                    target_dt_objs.append(today_obj + timedelta(days=int(m_after.group(1))))
                m_before = re.search(r"(\d+)\s*일\s*전", user_query)
                if m_before:
                    target_dt_objs.append(today_obj - timedelta(days=int(m_before.group(1))))

                if not target_dt_objs:
                    date_patterns = [
                        r"(\d+)\s*월\s*(\d+)\s*일",
                        r"(\d+)\s*/\s*(\d+)",
                        r"(\d+)\s*-\s*(\d+)",
                    ]
                    for pattern in date_patterns:
                        search_match = re.search(pattern, user_query)
                        if search_match:
                            mm = int(search_match.group(1))
                            dd = int(search_match.group(2))
                            target_dt_objs.append(datetime(year=today_obj.year, month=mm, day=dd))
                            break

            enhanced_query += " " + " ".join(o.strftime("%Y년 %m월 %d일") for o in target_dt_objs)

            for day in ['월', '화', '수', '목', '금', '토', '일']:
                if f"{day}요일" in user_query:
                    enhanced_query += f" {day}요일"
    except Exception as e:
        print(f"⚠️ [날짜 파싱 예외]: {e}")

    if any(k in user_query for k in ["급식", "점심", "중식", "저녁", "석식", "밥", "메뉴", "식단"]):
        enhanced_query += " 점심 중식 저녁 석식 급식 메뉴 식단 정보"

    print(f"🎯 [최종 인프라 쿼리 맵]: '{user_query}' -> '{enhanced_query}'")
    if target_dt_objs:
        print(f"🔑 [정확매칭 대상 날짜]: {[format_key(o) for o in target_dt_objs]} (범위질문: {is_range_query})")

    # ---------------------------------------------------------------
    # 1순위: 날짜가 명확히 파싱됐으면 dict에서 "정확히" 찾는다 (벡터 검색 X)
    # 범위 질문일 경우, 데이터가 있는 날짜/없는 날짜를 모두 명시해서
    # LLM이 "일부는 있고 일부는 없다"를 정확히 구분해 답하도록 만든다.
    # ---------------------------------------------------------------
    if target_dt_objs:
        found_parts = []
        missing_dates = []
        for o in target_dt_objs:
            key = format_key(o)
            if key in menu_lookup:
                found_parts.extend(menu_lookup[key])
            else:
                missing_dates.append(o.strftime("%Y년 %m월 %d일"))

        if found_parts or (is_range_query and missing_dates):
            context = "\n\n".join(found_parts)
            if missing_dates:
                context += "\n\n[참고: 아래 날짜는 학교 문서에 급식 정보가 존재하지 않음 - 주말/공휴일/방학일 수 있음]\n"
                context += ", ".join(missing_dates)
            return context

    # ---------------------------------------------------------------
    # 2순위: 날짜를 못 찾았거나(오타 등) 날짜 질문이 아니면 벡터 검색으로 폴백
    # ---------------------------------------------------------------
    docs = base_retriever.invoke(enhanced_query)
    return "\n\n".join(doc.page_content for doc in docs)

# 5. LLM 세팅
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

system_prompt_template = (
    "너는 우리 학교 전문 안내 AI 비서야. 너의 이름은 살레봇이야. "
    "반드시 제공된 [학교 지식 데이터베이스 추출 컨텍스트] 내용에만 엄격히 기반해서 '사실 그대로' 한국어로 답변해줘.\n\n"
    " [현재 시스템 기준 시계열 정보]\n"
    "{current_date}\n\n"
    " [학교 지식 데이터베이스 추출 컨텍스트]\n"
    "{context}\n\n"
    "지침:\n"
    "1. 학교 기본 안내 정보(교칙, 축제 등)에 대해 물어보면 지식베이스 컨텍스트 내용을 토대로 정확한 팩트와 날짜만 정갈하게 답변해.\n"
    "2. 어제, 오늘, 내일 등 급식을 물어보면 현재 시스템 기준 시계열 정보를 바탕으로 상대적 날짜를 정확히 계산하여, 컨텍스트에 기술된 해당 날짜 문장의 메뉴를 그대로 매칭해서 답변해줘.\n"
    "3. 'O월 O일부터 O월 O일까지'처럼 여러 날짜 급식을 한 번에 물어보면, 컨텍스트에 있는 날짜들은 날짜별로 각각 메뉴를 나열하고, 컨텍스트 하단에 '[참고: ... 정보가 존재하지 않음]'으로 명시된 날짜는 '해당 날짜는 급식 정보가 없습니다(주말/공휴일/방학 등)'라고 그 날짜만 콕 집어 답해줘. 일부 날짜에 정보가 없다고 해서 전체 질문을 거부하지 마.\n"
    "4. 지식 데이터베이스 추출 컨텍스트 내에 질문과 매칭되는 정확한 날짜나 텍스트 데이터가 전혀 존재하지 않는다면, 절대 상상해서 거짓말하지 말고 '제공된 학교 문서에 관련 정보가 확인되지 않습니다'라고 정중하게 답변해."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("human", "{question}"),
])

rag_chain = (
    {
        "context": search_context,
        "question": lambda x: x["question"],
        "current_date": lambda x: x["current_date"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*60)
print("🤖 [시스템] 날짜 정확매칭 강화판 - 살레봇 무결성 최종 완결판 시동")
print("종료하려면 '종료' 또는 'q'를 입력하세요.")
print("="*60 + "\n")

while True:
    user_question = input("🙋 질문을 입력하세요: ").strip()

    if user_question.lower() in ['종료', 'exit', 'quit', 'q']:
        print("🤖 살레봇 엔진 가동을 안전하게 종료합니다.")
        break

    if not user_question:
        continue

    now = datetime.now()
    weekday_list = ['월', '화', '수', '목', '금', '토', '일']
    current_date_str = f"현재 일시: {now.strftime('%Y년 %m월 %d일')} ({weekday_list[now.weekday()]}요일)"

    print(f"🕒 [클럭 타임스탬프]: {current_date_str}")

    try:
        response = rag_chain.invoke({
            "question": user_question,
            "current_date": current_date_str
        })
        print(f"\n🤖 살레봇 답변:\n{response}")
        print("\n" + "-"*60 + "\n")

    except Exception as e:
        print(f"❌ 런타임 예외 에러 발생: {e}")
        print("질문을 명확히 다시 입력해 주세요.\n")