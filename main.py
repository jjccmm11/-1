import os
import re
import time
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# 2. 학교 정보 파일을 "검색" 없이 통째로 읽어서 메모리에 올려둔다.
#    (=이게 지금 단계의 "학습"에 해당하는 부분. 파일 내용을 그대로 변수에 담아두는 것)
with open("school_info.txt", "r", encoding="utf-8") as f:
    school_info = f.read()

with open("school_menu.txt", "r", encoding="utf-8") as f:
    school_menu = f.read()

print(f"📚 [시스템] school_info.txt 로드 완료 (총 {len(school_info)}자)")
print(f"📚 [시스템] school_menu.txt 로드 완료 (총 {len(school_menu)}자)")


# 2-1. [추가] 날짜별 빠른 조회용 사전(dict) 구축
#    데이터가 커지면 매번 전체를 다 보내는 게 느려지므로, "YYYY년 MM월 DD일"이
#    포함된 줄들을 파일별로 각각 날짜별로 미리 모아둔다.
#    + 날짜가 없는 "일반 안내" 줄은 따로 모아둔다 (교칙 등, 달과 무관하게 항상 필요할 수 있는 내용).
#    이렇게 분리해두면, 못 찾은 날짜를 물어봐도 "일반 안내"만 보내면 되니까
#    전체 파일(다른 달 데이터까지 전부)을 다시 보낼 필요가 없어진다.
def build_date_lookup(text):
    lookup = {}
    general_lines = []
    for line in text.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        m = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", line_strip)
        if m:
            key = f"{m.group(1)}년{int(m.group(2)):02d}월{int(m.group(3)):02d}일"
            lookup.setdefault(key, []).append(line_strip)
        else:
            general_lines.append(line_strip)
    return lookup, general_lines


info_date_lookup, info_general_lines = build_date_lookup(school_info)
menu_date_lookup, menu_general_lines = build_date_lookup(school_menu)

print(f"📅 [시스템] 학교 정보 날짜 사전 {len(info_date_lookup)}일치 + 일반 안내 {len(info_general_lines)}줄 구축 완료")
print(f"📅 [시스템] 급식 날짜 사전 {len(menu_date_lookup)}일치 구축 완료")

# 2-2. 데이터 총량은 참고용으로만 출력 (이제 크기와 무관하게 항상
#    날짜 매칭 또는 키워드 검색으로 필요한 부분만 골라 보내므로, 전체를
#    통째로 보내는 모드는 더 이상 사용하지 않는다 -> TPM 한도 걱정 없음)
total_len = len(school_info) + len(school_menu)
print(f"🧠 [시스템] 데이터 총량 {total_len}자 (항상 필요한 부분만 골라서 전송하므로 크기와 무관하게 안전)")

# 3. LLM 설정
# [핵심 수정] openai/gpt-oss-120b는 "추론(reasoning) 모델"이라 질문마다 내부적으로
# 생각하는 양이 들쭉날쭉해서 응답 시간이 크게 요동칠 수 있다(같은 크기 컨텍스트여도
# 2초 vs 35초씩 차이 날 수 있음). reasoning_effort를 낮게 고정해서 속도를 일정하게 만든다.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, reasoning_effort="low")

# 4. 프롬프트: 매 질문마다 school_info / school_menu를 각각 컨텍스트로 넣어준다.
system_prompt_template = (
    "너는 우리 학교 전문 안내 AI 비서야. 이름은 살레봇이야. "
    "아래 [학교 정보]에 있는 내용을 물어보면 학교 정보에 근거해서 한국어로 정확하게 답변해줘."
    "아래 [학교 메뉴]에 있는 내용을 물어보면 학교 메뉴에 근거해서 한국어로 정확하게 답변해줘."
    "학사일정을 물어봤는데 학사일정이 없으면 정보가 없다고 말하지말고, '오늘은 별도로 일정이 없습니다' 정도의 느낌으로 말해줘"
    "급식이 없는날엔 특이사항에 뭐라고 적혀있다면 '[특이사항] 때문에 별도로 급식이 존재하지 않습니다'의 느낌으로 말해줘'"
    "정보에 없는 내용을 물어보면 상상하지 말고 모른다고 솔직하게 말해.\n\n"
    "[현재 날짜]\n{current_date}\n\n"
    "[학교 정보]\n{context}\n"
    "[학교 메뉴]\n{menu_context}\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    ("human", "{question}"),
])

chain = prompt | llm  # [진단용 변경] 토큰 사용량 확인을 위해 StrOutputParser를 빼고 원본 응답(AIMessage)을 그대로 받는다.


# [추가] 질문에서 날짜를 찾아, 있으면 두 사전에서 각각 그 줄만 빠르게 꺼내고
# (전체 문맥 모드거나 날짜를 못 찾으면) 기존처럼 전체를 그대로 넣는다.
def extract_target_dates(user_question, today_obj, last_target_date=None):
    target_dates = []

    if "오늘" in user_question:
        target_dates.append(today_obj)
    if "내일" in user_question:
        target_dates.append(today_obj + timedelta(days=1))
    if "어제" in user_question:
        target_dates.append(today_obj - timedelta(days=1))
    if "일주일" in user_question:
        target_dates.append(today_obj + timedelta(days=7))
    if "일주일 후" in user_question:
        target_dates.append(today_obj - timedelta(days=7))
# [핵심 수정] "그 3일전", "그날부터 3일후"처럼 "그"가 들어간 상대 날짜 표현은
# "오늘"이 아니라 "직전에 확인했던 날짜"를 기준으로 계산해야 한다.
# ("그리고", "그래서" 같은 무관한 단어에 오작동하지 않도록 패턴을 좁혀서 검사)
    refers_to_last_date = bool(re.search(r"그날|그때|그로부터|그\s", user_question))
    base_date = last_target_date if (refers_to_last_date and last_target_date is not None) else today_obj

    # [추가] "N일 전", "N일 뒤/후" 같은 상대 날짜 표현 (기준일: base_date)
    m_before = re.search(r"(\d+)\s*일\s*전", user_question)
    if m_before:
        target_dates.append(base_date - timedelta(days=int(m_before.group(1))))

    m_after = re.search(r"(\d+)\s*일\s*(?:뒤|후)", user_question)
    if m_after:
        target_dates.append(base_date + timedelta(days=int(m_after.group(1))))

    # [추가] "일주일 전", "이주일 전", "3주 전", "3주일 후" 같은 "주" 단위 상대 날짜 (기준일: base_date)
    korean_num_map = {"한": 1, "일": 1, "이": 2, "두": 2, "삼": 3, "세": 3, "사": 4, "네": 4, "오": 5, "다섯": 5}

    m_week_before = re.search(r"(\d+|[가-힣])\s*주(?:일)?\s*전", user_question)
    if m_week_before:
        raw = m_week_before.group(1)
        weeks = int(raw) if raw.isdigit() else korean_num_map.get(raw, 1)
        target_dates.append(base_date - timedelta(weeks=weeks))

    m_week_after = re.search(r"(\d+|[가-힣])\s*주(?:일)?\s*(?:뒤|후)", user_question)
    if m_week_after:
        raw = m_week_after.group(1)
        weeks = int(raw) if raw.isdigit() else korean_num_map.get(raw, 1)
        target_dates.append(base_date + timedelta(weeks=weeks))

    # [추가] "다음주"/"이번주" (+ 특정 요일) 표현
    # 요일까지 지정하면 그 날짜 하나만, 요일 없이 "다음주 급식 알려줘"처럼 물으면
    # 그 주 7일 전체를 대상 날짜로 넣는다 (있는 날짜만 나중에 자동으로 골라짐).
    weekday_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    this_week_monday = today_obj - timedelta(days=today_obj.weekday())

    # 주의: "다다음주"는 "다음주"를 포함하고 있으므로 반드시 더 긴 표현부터 검사한다.
    week_offset = None
    if re.search(r"다\s*다\s*음\s*주", user_question):
        week_offset = 2
    elif re.search(r"다음\s*주", user_question):
        week_offset = 1
    elif re.search(r"이번\s*주", user_question):
        week_offset = 0
    elif re.search(r"지지난\s*주", user_question):
        week_offset = -2
    elif re.search(r"(?:지난|저번)\s*주", user_question):
        week_offset = -1

    week_start = None
    if week_offset is not None:
        week_start = this_week_monday + timedelta(weeks=week_offset)

    if week_start is not None:
        matched_weekday = None
        for name, idx in weekday_map.items():
            if f"{name}요일" in user_question:
                matched_weekday = idx
                break
        if matched_weekday is not None:
            target_dates.append(week_start + timedelta(days=matched_weekday))
        else:
            target_dates.extend(week_start + timedelta(days=i) for i in range(7))

    # [추가] "그날", "그 다음날", "그 전날" 처럼 오늘이 아니라 "직전에 물어본 날짜"를
    # 기준으로 삼는 표현. 예: "9월14일 급식" 물어본 다음 "그 다음날 급식"이라고 하면
    # last_target_date(=9월14일) + 1일로 계산한다.
    if not target_dates and last_target_date is not None:
        if re.search(r"다음\s*날|다음날", user_question):
            target_dates.append(last_target_date + timedelta(days=1))
        elif re.search(r"전\s*날|전날", user_question):
            target_dates.append(last_target_date - timedelta(days=1))
        elif "그날" in user_question or "그 날" in user_question:
            target_dates.append(last_target_date)

    if not target_dates:
        m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", user_question)
        if m:
            mm, dd = int(m.group(1)), int(m.group(2))
            target_dates.append(datetime(year=today_obj.year, month=mm, day=dd))

    return target_dates


def build_context(user_question, today_obj):
    target_dates = extract_target_dates(user_question, today_obj)

    if target_dates:
        date_keys = [d.strftime("%Y년%m월%d일") for d in target_dates]
        date_labels = [d.strftime("%Y년 %m월 %d일") for d in target_dates]

        info_lines = []
        menu_lines = []
        for key in date_keys:
            info_lines.extend(info_date_lookup.get(key, []))
            menu_lines.extend(menu_date_lookup.get(key, []))

        # 찾은 날짜 줄 + 날짜 없는 "일반 안내"만 합쳐서 보낸다.
        # -> 데이터가 몇 달치로 늘어나도 항상 가벼운 상태 유지.
        context_parts = info_lines + info_general_lines
        context = "\n".join(context_parts) if context_parts else "\n".join(info_general_lines)

        # 급식은 하루 단위로 딱 떨어지는 포맷이라, 못 찾으면 "정보 없음"이라고 명시.
        if menu_lines:
            menu_context = "\n".join(menu_lines)
        else:
            menu_context = f"[참고: {', '.join(date_labels)}에 해당하는 급식 정보가 school_menu.txt에 없음]"

        return context, menu_context

    # [핵심 변경] 날짜가 없는 일반 질문이라고 해서 전체를 다 보내지 않는다.
    # 학교 정보/급식 데이터 "전체" 중에서 질문과 관련도 높은 줄만 키워드 검색으로 골라 보낸다.
    # (모든 줄이 검색 대상이라 정보 손실은 없고, 실제로 보내는 양만 항상 작게 유지된다)
    all_info_lines = info_general_lines + [line for lines in info_date_lookup.values() for line in lines]
    all_menu_lines = [line for lines in menu_date_lookup.values() for line in lines]

    context = simple_keyword_search(user_question, all_info_lines) or "\n".join(info_general_lines)
    menu_context = simple_keyword_search(user_question, all_menu_lines) or "질문과 관련된 급식 정보를 찾지 못했습니다."

    return context, menu_context


# [교체] 눈 감고 앞부분만 자르는 clip_text 대신, 질문과 겹치는 단어가 많은 줄부터
# 골라 담는 가벼운 키워드 검색. 임베딩/벡터DB 없이도 "관련 있는 부분만 선택"할 수 있다.
# -> 데이터를 버리는 게 아니라, 전체(모든 줄) 중에서 이번 질문에 필요한 것만 뽑아 쓰는 것.
KEYWORD_SEARCH_MAX_CHARS = 2500  # 한 번에 보낼 수 있는 최대 글자 수 (TPM 8000 기준 여유 있게)


def get_char_ngrams(text, n=2):
    # 띄어쓰기가 있든 없든 매칭되도록, 공백을 지우고 글자 단위로 n개씩 잘라서 비교한다.
    # (예: "수련회언제야" 도 "수련","련회","회언"... 식으로 쪼개지므로
    #  데이터 속 "수련회"라는 글자가 그대로 겹치게 됨)
    cleaned = re.sub(r"\s+", "", text)
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i:i + n] for i in range(len(cleaned) - n + 1)}


def simple_keyword_search(user_question, lines, max_chars=KEYWORD_SEARCH_MAX_CHARS):
    if not lines:
        return ""
    question_grams = get_char_ngrams(user_question)
    scored = []
    for line in lines:
        line_grams = get_char_ngrams(line)
        overlap = len(question_grams & line_grams)
        if overlap > 0:
            scored.append((overlap, line))
    scored.sort(key=lambda x: -x[0])

    selected = []
    total_len = 0
    for _, line in scored:
        if total_len + len(line) > max_chars:
            break
        selected.append(line)
        total_len += len(line)
    return "\n".join(selected)


# 5. 대화 루프
print("\n" + "=" * 50)
print("🤖 살레봇 (Full-Context + 날짜 사전 하이브리드) 시작")
print("종료하려면 '종료' 를 입력하세요.")
print("=" * 50 + "\n")

weekday_list = ['월', '화', '수', '목', '금', '토', '일']

while True:
    user_question = input("🙋 질문: ").strip()

    if user_question.lower() in ["종료", "exit", "quit", "q"]:
        print("🤖 살레봇을 종료합니다.")
        break

    if not user_question:
        continue
    
    now = datetime.now()
    current_date_str = f"{now.strftime('%Y년 %m월 %d일')} ({weekday_list[now.weekday()]}요일)"

    # [추가] 어디서 시간이 드는지 직접 측정: 딕셔너리 조회 vs LLM 응답
    t0 = time.perf_counter()
    context, menu_context = build_context(user_question, now)
    t1 = time.perf_counter()

    print(f"🔍 [진단] 컨텍스트 검색 소요 시간: {t1 - t0:.4f}초 "
          f"(학교정보 {len(context)}자 + 급식 {len(menu_context)}자 전송 예정)")

    response = chain.invoke({
        "question": user_question,
        "context": context,
        "menu_context": menu_context,
        "current_date": current_date_str,
    })
    t2 = time.perf_counter()

    print(f"⏱️ [진단] LLM 응답 소요 시간: {t2 - t1:.4f}초")

    # [추가] 실제 사용된 토큰 수를 출력해서, 느려진 게 "토큰을 많이 만들어서"인지
    # 아니면 "토큰 수와 무관한 API/네트워크 지연"인지 구분한다.
    usage = getattr(response, "usage_metadata", None) or {}
    raw_usage = (getattr(response, "response_metadata", {}) or {}).get("token_usage", {})
    print(f"🔢 [진단] 토큰 사용량: {usage or raw_usage}")

    print(f"\n🤖 살레봇: {response.content}\n")