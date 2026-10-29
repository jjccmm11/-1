import os
import re
import time
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from groq import Groq
import os
import json
import urllib.request
import urllib.error
from datetime import datetime
from auto_sync import sync_if_due, get_context_texts, start_daily_scheduler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def send_report_to_discord(question, answer):
    """오답 신고를 디스코드 웹훅으로 보낸다. 성공하면 True, 실패하면 False."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print("⚠️ DISCORD_WEBHOOK_URL이 설정되지 않아 신고를 보내지 못했습니다.")
        return False

    # 디스코드 메시지는 2000자 제한 -> 넉넉히 잘라서 보낸다
    content = (
        f"🚩 **살레봇 오답 신고** ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n"
        f"**질문:** {question[:300]}\n"
        f"**답변:**\n{answer[:1400]}"
    )[:1900]

    data = json.dumps({
        "content": content,
        "allowed_mentions": {"parse": []},  # 질문에 @everyone 같은 게 있어도 알림이 울리지 않게
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "SalebotReport/1.0"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)  # 성공하면 디스코드가 204를 돌려줌
        return True
    except urllib.error.HTTPError as e:
        print(f"⚠️ 디스코드 전송 실패 (HTTP {e.code})")
    except Exception as e:
        print(f"⚠️ 디스코드 전송 실패: {e}")
    return False
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# 2. 기존 수동 작성 파일을 읽고, 자동 수집 데이터는 SQLite DB에서 불러온다.
# NEIS_API_KEY가 설정되어 있으면 마지막 수집 후 24시간이 지났을 때 자동 갱신한다.
try:
    sync_result = sync_if_due()
    print(f"🌐 [자동수집] {sync_result}")
except Exception as exc:
    print(f"⚠️ [자동수집] 시작 시 수집 실패. 저장된 데이터/수동 파일로 계속 실행합니다: {exc}")

for filename in ("school_info.txt", "school_menu.txt"):
    if not os.path.exists(filename):
        open(filename, "a", encoding="utf-8").close()

with open("school_info.txt", "r", encoding="utf-8") as f:
    manual_school_info = f.read()
with open("school_menu.txt", "r", encoding="utf-8") as f:
    manual_school_menu = f.read()

db_info_text, db_menu_text = get_context_texts()
school_info = manual_school_info + ("\\n" + db_info_text if db_info_text else "")
school_menu = manual_school_menu + ("\\n" + db_menu_text if db_menu_text else "")

print(f"📚 [시스템] school_info.txt + DB 로드 완료 (총 {len(school_info)}자)")
print(f"📚 [시스템] school_menu.txt + DB 로드 완료 (총 {len(school_menu)}자)")


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

def refresh_school_context_from_db():
    """백그라운드 수집 후 현재 실행 중인 챗봇의 검색 사전도 새로 만든다."""
    global school_info, school_menu
    global info_date_lookup, info_general_lines, menu_date_lookup, menu_general_lines
    db_info, db_menu = get_context_texts()
    school_info = manual_school_info + ("\\n" + db_info if db_info else "")
    school_menu = manual_school_menu + ("\\n" + db_menu if db_menu else "")
    info_date_lookup, info_general_lines = build_date_lookup(school_info)
    menu_date_lookup, menu_general_lines = build_date_lookup(school_menu)
    print("🔄 [자동수집] DB 데이터로 검색 사전을 갱신했습니다.")

# 실행 중인 프로세스에서도 하루마다 갱신한다.
start_daily_scheduler(refresh_school_context_from_db)


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
    "정보에 없는 내용을 물어보면 상상하지 말고 모른다고 솔직하게 말해. "
    "중요: 질문에 '10월 30일'처럼 구체적인 날짜가 있으면 [현재 날짜]보다 질문에 적힌 날짜를 우선해. "
    "'오늘'이라는 단어가 함께 있어도 구체적인 날짜를 임의로 현재 날짜로 바꾸지 마. "
    "[조회 대상 날짜]에 적힌 날짜의 학교 자료만 근거로 답하고, 다른 날짜 자료를 섞지 마. "
    "조회 대상 날짜의 자료가 제공되어 있으면 그 자료를 확인해서 답해.\n\n"
    "[현재 날짜]\n{current_date}\n"
    "[조회 대상 날짜]\n{target_date_text}\n\n"
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
    """질문에서 대상 날짜를 추출한다.

    우선순위:
    1) 질문에 명시된 날짜(예: 오늘 10월 30일 일정) 우선
    2) 오늘/내일/어제/모레는 실제 오늘 날짜 기준
    3) 그 다음날/그 전날은 직전에 조회한 날짜 기준
    4) N일 뒤/일주일 뒤 등은 직전 조회 날짜가 있으면 그 날짜 기준
    5) 이번 주/다음 주 등은 실제 오늘 날짜 기준
    """
    question = re.sub(r"\s+", " ", user_question).strip()
    target_dates = []

    # 1. 명시적인 날짜가 있으면 '오늘' 같은 표현보다 먼저 처리한다.
    # 예: "오늘 10월 30일 일정", "2026년 10월 30일 학사일정"
    explicit = re.search(
        r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일",
        question
    )
    if explicit:
        year = int(explicit.group(1)) if explicit.group(1) else today_obj.year
        month = int(explicit.group(2))
        day = int(explicit.group(3))
        try:
            return [datetime(year, month, day)]
        except ValueError:
            # 잘못된 날짜라면 임의로 다른 날짜를 만들지 않는다.
            return []

    # 2. 절대적인 상대 날짜 표현은 항상 실제 오늘 날짜 기준이다.
    # 이전 대화에서 다른 날짜를 조회했더라도 "내일"은 오늘의 다음 날이다.
    if re.search(r"모레", question):
        return [today_obj + timedelta(days=2)]
    if re.search(r"내일", question):
        return [today_obj + timedelta(days=1)]
    if re.search(r"어제", question):
        return [today_obj - timedelta(days=1)]
    if re.search(r"오늘", question):
        return [today_obj]

    # 3. 직전 조회 날짜를 가리키는 표현
    if last_target_date is not None:
        if re.search(r"그\s*다음\s*날|그다음날|다음\s*날|다음날", question):
            return [last_target_date + timedelta(days=1)]
        if re.search(r"그\s*전\s*날|그전날|전\s*날|전날", question):
            return [last_target_date - timedelta(days=1)]
        if re.search(r"그\s*날|그날", question):
            return [last_target_date]

    # 4. 일/주 단위 상대 날짜. 문맥상 직전 날짜가 있으면 그것을 기준으로 한다.
    relative_base = last_target_date or today_obj
    m_before = re.search(r"(\d+)\s*일\s*전", question)
    if m_before:
        return [relative_base - timedelta(days=int(m_before.group(1)))]

    m_after = re.search(r"(\d+)\s*일\s*(?:뒤|후|후에)", question)
    if m_after:
        return [relative_base + timedelta(days=int(m_after.group(1)))]

    korean_num_map = {
        "한": 1, "일": 1, "이": 2, "두": 2, "삼": 3, "세": 3,
        "사": 4, "네": 4, "오": 5, "다섯": 5
    }
    m_week_before = re.search(r"(\d+|다섯|한|일|이|두|삼|세|사|네|오)\s*주(?:일)?\s*전", question)
    if m_week_before:
        raw = m_week_before.group(1)
        weeks = int(raw) if raw.isdigit() else korean_num_map.get(raw, 1)
        return [relative_base - timedelta(weeks=weeks)]

    m_week_after = re.search(r"(\d+|다섯|한|일|이|두|삼|세|사|네|오)\s*주(?:일)?\s*(?:뒤|후|후에)", question)
    if m_week_after:
        raw = m_week_after.group(1)
        weeks = int(raw) if raw.isdigit() else korean_num_map.get(raw, 1)
        return [relative_base + timedelta(weeks=weeks)]

    # 5. 이번 주/다음 주/지난 주 + 요일. 이 표현은 오늘이 속한 주를 기준으로 한다.
    weekday_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    this_week_monday = today_obj - timedelta(days=today_obj.weekday())
    week_offset = None
    if re.search(r"다\s*다\s*음\s*주", question):
        week_offset = 2
    elif re.search(r"다음\s*주", question):
        week_offset = 1
    elif re.search(r"이번\s*주", question):
        week_offset = 0
    elif re.search(r"지지난\s*주", question):
        week_offset = -2
    elif re.search(r"(?:지난|저번)\s*주", question):
        week_offset = -1

    if week_offset is not None:
        week_start = this_week_monday + timedelta(weeks=week_offset)
        matched_weekday = None
        for name, idx in weekday_map.items():
            if f"{name}요일" in question:
                matched_weekday = idx
                break
        if matched_weekday is not None:
            return [week_start + timedelta(days=matched_weekday)]
        return [week_start + timedelta(days=i) for i in range(7)]

    # 6. 날짜를 찾지 못했으면 빈 목록을 반환해 일반 키워드 검색으로 넘긴다.
    return []


def build_context(user_question, today_obj, last_target_date=None):
    target_dates = extract_target_dates(user_question, today_obj, last_target_date)

    if target_dates:
        date_keys = [d.strftime("%Y년%m월%d일") for d in target_dates]
        date_labels = [d.strftime("%Y년 %m월 %d일") for d in target_dates]

        info_lines = []
        menu_lines = []
        for key in date_keys:
            info_lines.extend(info_date_lookup.get(key, []))
            menu_lines.extend(menu_date_lookup.get(key, []))

        # 날짜별 자료가 없더라도 일반 안내는 함께 전달한다.
        context_parts = info_lines + info_general_lines
        context = "\n".join(context_parts) if context_parts else "\n".join(info_general_lines)

        if menu_lines:
            menu_context = "\n".join(menu_lines)
        else:
            menu_context = f"[참고: {', '.join(date_labels)}에 해당하는 급식 정보가 school_menu.txt에 없음]"

        # 대상 날짜도 반환해 다음 질문의 '그 다음날' 등에 활용한다.
        return context, menu_context, target_dates

    # 날짜가 없는 질문은 전체 자료에서 관련도가 높은 줄을 검색한다.
    all_info_lines = info_general_lines + [line for lines in info_date_lookup.values() for line in lines]
    all_menu_lines = [line for lines in menu_date_lookup.values() for line in lines]

    context = simple_keyword_search(user_question, all_info_lines) or "\n".join(info_general_lines)
    menu_context = simple_keyword_search(user_question, all_menu_lines) or "질문과 관련된 급식 정보를 찾지 못했습니다."

    return context, menu_context, []


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
print("답변이 틀렸다면 다음 입력창에 '/신고' 를 입력해주세요. (salebot_report.txt에 저장됩니다)")
print("=" * 50 + "\n")

weekday_list = ['월', '화', '수', '목', '금', '토', '일']

def clean_markdown(text):
    """마크다운 표/제목/굵게 표시를 일반 텍스트로 바꿔준다."""
    out, header, last_first = [], None, ""
    for raw in text.splitlines():
        line = raw.strip().replace("**", "").replace("__", "")
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # 표 구분선(|---|---|) 제거
            if header is None:
                header = cells
                continue
            if cells[0]:
                last_first = cells[0]
                out.append(f"\n📅 {last_first}")
            rest = cells[1:]
            line_txt = f"  • {rest[0]}: {rest[1]}" if len(rest) >= 2 else f"  • {' / '.join(rest)}"
            for h, v in zip(header[3:], rest[2:]):
                if v and v != "없음":
                    line_txt += f"  [{h}: {v}]"
            out.append(line_txt)
            continue
        header = None
        if re.fullmatch(r"[-*_]{3,}", line):
            continue  # --- 구분선 제거
        line = re.sub(r"^#{1,6}\s*", "", line)  # ### 제목 기호 제거
        line = re.sub(r"^>\s?", "", line)       # > 인용 기호 제거
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()

# =========================================================================
# [추가] 월/연 단위 "목록" 질문 처리 (이번달, 다음달, 9월 전체, 올해 ...)
# 한 달치 급식은 토큰 한도(8,000)를 넘기 때문에 LLM에 보내지 않고,
# 코드가 데이터에서 직접 꺼내 정리해서 출력한다. (잘림/오류/지연 없음)
# =========================================================================
MENU_WORDS = ["급식", "점심", "중식", "저녁", "석식", "밥", "메뉴", "식단"]
INFO_WORDS = ["일정", "학사", "행사", "시험", "방학", "공지", "뭐 있", "뭐있"]
# 목록이 아니라 "골라서 답해야 하는" 질문은 기존 방식(LLM)으로 처리
FILTER_WORDS = ["언제", "나오는", "나와", "중에", "제일", "가장", "몇", "며칠", "어떤 날"]
YEAR_PATTERN = (r"올해|금년|올\s*한\s*해|한\s*해|이번\s*해|이번\s*(?:년|연)\s*도?|연도|(?<!학)년도"
                r"|연간|작년|내년|(?<!\d)1\s*년|일\s*년|\d{4}\s*년")
# 월/일/요일/상대날짜가 같이 들어 있으면 "연 단위"가 아니라 특정 날짜 질문이므로 막지 않는다
SPECIFIC_DATE_PATTERN = (r"\d{1,2}\s*월|오늘|내일|어제|모레|그날|전날|다음\s*날"
                         r"|(?:이번|다음|지난|저번|다다음|지지난)\s*주"
                         r"|\d+\s*(?:일|주)\s*(?:전|후|뒤)|[일이한두삼세]\s*주일?\s*(?:전|후|뒤)")


def detect_month_scope(question, today_obj):
    if re.search(r"이번\s*달", question):
        return today_obj.year, today_obj.month
    if re.search(r"다음\s*달", question):
        return (today_obj.year + 1, 1) if today_obj.month == 12 else (today_obj.year, today_obj.month + 1)
    if re.search(r"(?:지난|저번)\s*달", question):
        return (today_obj.year - 1, 12) if today_obj.month == 1 else (today_obj.year, today_obj.month - 1)
    # "9월", "9월달", "2026년 9월" (뒤에 일자 숫자가 오면 특정 날짜 질문이므로 제외)
    m = re.search(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월(?:\s*달)?(?!\s*\d)", question)
    if m and 1 <= int(m.group(2)) <= 12:
        return (int(m.group(1)) if m.group(1) else today_obj.year), int(m.group(2))
    return None


def format_menu_line(line):
    d = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일", line)
    lunch = re.search(r"식단\s*:\s*(.*?)\.?\s*석식", line, re.S)
    dinner = re.search(r"석식\s*저녁\s*:\s*(.*?)\.?\s*특이사항", line, re.S)
    note = re.search(r"특이사항\s*:\s*(.*?)\.?\s*$", line, re.S)
    if not (d and lunch):
        return "  • " + line  # 형식이 다르면 원문 그대로
    out = [f"  • 점심: {lunch.group(1).strip()}"]
    if dinner:
        out.append(f"  • 석식: {dinner.group(1).strip()}")
    if note and note.group(1).strip() not in ("", "없음"):
        out.append(f"  • 특이사항: {note.group(1).strip()}")
    return "\n".join(out)


def date_header(key):
    y, m, d = int(key[:4]), int(key[5:7]), int(key[8:10])
    return f"\n📅 {m}월 {d}일 ({weekday_list[datetime(y, m, d).weekday()]})"


def build_month_listing(year, month, want_menu, want_info):
    prefix = f"{year}년{month:02d}월"
    sections = []
    if want_menu:
        keys = sorted(k for k in menu_date_lookup if k.startswith(prefix))
        body = [f"🍱 {year}년 {month}월 급식"]
        for k in keys:
            body.append(date_header(k))
            body.extend(format_menu_line(l) for l in menu_date_lookup[k])
        sections.append("\n".join(body) if keys else f"🍱 {year}년 {month}월 급식 자료가 없습니다.")
    if want_info:
        keys = sorted(k for k in info_date_lookup if k.startswith(prefix))
        body = [f"📚 {year}년 {month}월 학사일정"]
        for k in keys:
            body.append(date_header(k))
            for l in info_date_lookup[k]:
                text = re.sub(r"^.*?\d{4}년\s*\d{1,2}월\s*\d{1,2}일\s*(?:[월화수목금토일]요일)?\s*:?\s*", "", l) or l
                body.append("  • " + text)
        sections.append("\n".join(body) if keys else f"📚 {year}년 {month}월 학사일정 자료가 없습니다.")
    return "\n\n".join(sections)


def handle_bulk_query(question, today_obj):
    """월/연 단위 목록 질문이면 답변 문자열을, 아니면 None을 반환."""
    want_menu = any(w in question for w in MENU_WORDS)
    want_info = any(w in question for w in INFO_WORDS)
    if not (want_menu or want_info):
        return None

    scope = detect_month_scope(question, today_obj)

    # 연 단위(올해/이번연도/2026년/1년 전체...)는 정보량이 너무 커서 무조건 막는다.
    if (not scope and re.search(YEAR_PATTERN, question)
            and not re.search(SPECIFIC_DATE_PATTERN, question)):
        return ("연 단위(1년 전체) 조회는 정보량이 너무 많아서 보내드릴 수 없어요 🙏\n"
                "월 단위로 물어봐 주세요!\n"
                "  예) '9월 급식', '이번달 학사일정', '다음달 급식이랑 일정'")

    if any(w in question for w in FILTER_WORDS):
        return None
    if scope:
        return build_month_listing(scope[0], scope[1], want_menu, want_info)
    return None


# =========================================================================
# [추가] 오답 신고 기능: "/신고" 입력 시 직전 질문/답변을 salebot_report.txt에 저장
# =========================================================================
REPORT_PATH = "salebot_report.txt"
last_question = None
last_answer = None
last_target_date = None


def save_report(question, answer):
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"질문: {question}\n")
        f.write(f"답변: {answer}\n")
        f.write("-" * 40 + "\n")


while True:
    user_question = input("🙋 질문: ").strip()

    if user_question.lower() in ["종료", "exit", "quit", "q"]:
        print("🤖 살레봇을 종료합니다.")
        break

    if not user_question:
        continue
    
    # 관리자용 수동 동기화 명령. 배포 환경에서 필요할 때 데이터 갱신을 시험할 수 있다.
    if user_question == "/동기화":
        try:
            from auto_sync import sync_all
            print(f"🌐 [자동수집] 수동 갱신 결과: {sync_all()}")
            refresh_school_context_from_db()
            answer_text = "자동 수집을 실행했습니다. 콘솔의 출처별 결과를 확인해 주세요."
        except Exception as exc:
            answer_text = f"자동 수집 중 오류가 발생했습니다: {exc}"
            print(f"⚠️ {answer_text}")
        print(f"\n🤖 살레봇: {answer_text}\n")
        last_question, last_answer = user_question, answer_text
        continue

    # [추가] 오답 신고: 직전 질문/답변을 salebot_report.txt에 저장 (LLM 호출 없음)
    if user_question == "/신고":
        if last_question is None:
            print("🤖 아직 신고할 이전 답변이 없습니다.\n")
        else:
            if send_report_to_discord(last_question, last_answer):
                print("🚩 신고가 접수되었습니다. 감사합니다!\n")
            else:
                print("🚩 신고를 보내지 못했습니다. 잠시 후 다시 시도해주세요.\n")
        continue

    # [추가] 월/연 단위 목록 질문(이번달, 9월 전체, 올해 ...)은 LLM 없이 코드가 직접 답한다.
    bulk_answer = handle_bulk_query(user_question, datetime.now())
    if bulk_answer is not None:
        print(f"\n🤖 살레봇:\n{bulk_answer}\n")
        last_question, last_answer = user_question, bulk_answer
        continue

    now = datetime.now()
    current_date_str = f"{now.strftime('%Y년 %m월 %d일')} ({weekday_list[now.weekday()]}요일)"

    # [추가] 어디서 시간이 드는지 직접 측정: 딕셔너리 조회 vs LLM 응답
    t0 = time.perf_counter()
    context, menu_context, queried_dates = build_context(user_question, now, last_target_date)
    if queried_dates:
        last_target_date = queried_dates[-1]
    t1 = time.perf_counter()

    print(f"🔍 [진단] 컨텍스트 검색 소요 시간: {t1 - t0:.4f}초 "
          f"(학교정보 {len(context)}자 + 급식 {len(menu_context)}자 전송 예정)")
    if queried_dates:
        target_date_text = ", ".join(d.strftime("%Y년 %m월 %d일") for d in queried_dates)
        print(f"🗓️ [진단] 실제 조회 대상 날짜: {target_date_text}")
        print(f"🗓️ [진단] 일정 자료 일치 줄 수: {sum(len(info_date_lookup.get(d.strftime('%Y년%m월%d일'), [])) for d in queried_dates)}")
        print(f"🗓️ [진단] 급식 자료 일치 줄 수: {sum(len(menu_date_lookup.get(d.strftime('%Y년%m월%d일'), [])) for d in queried_dates)}")
    else:
        target_date_text = "질문에서 특정 날짜를 지정하지 않음"

    response = chain.invoke({
        "question": user_question,
        "context": context,
        "menu_context": menu_context,
        "current_date": current_date_str,
        "target_date_text": target_date_text,
    })
    t2 = time.perf_counter()

    print(f"⏱️ [진단] LLM 응답 소요 시간: {t2 - t1:.4f}초")

    # [수정] 날짜가 정해진 학사일정 질문은 LLM의 추측/오해를 거치지 않고
    # 해당 날짜의 school_info.txt 원문을 직접 반환한다.
    # 예: "오늘 학사일정", "오늘 일정", "오늘 10월 29일 학사일정"
    is_schedule_query = any(
        word in user_question for word in ["학사일정", "일정", "행사", "시험", "방학"]
    )
    if queried_dates and is_schedule_query:
        direct_lines = []
        missing_dates = []

        for target_date in queried_dates:
            key = target_date.strftime("%Y년%m월%d일")
            date_lines = info_date_lookup.get(key, [])
            if date_lines:
                direct_lines.extend(date_lines)
            else:
                missing_dates.append(target_date.strftime("%Y년 %m월 %d일"))

        if direct_lines:
            answer_text = "\n".join(f"• {line}" for line in direct_lines)
        else:
            missing_text = ", ".join(missing_dates)
            answer_text = (
                f"{missing_text} 학사일정 자료를 school_info.txt에서 찾지 못했습니다. "
                "일정이 없는 날인지 자료가 누락된 것인지 확인이 필요합니다."
            )

        print(f"\n🤖 살레봇: {answer_text}\n")
        last_question, last_answer = user_question, answer_text
        continue

    # [추가] 실제 사용된 토큰 수를 출력해서, 느려진 게 "토큰을 많이 만들어서"인지
    # 아니면 "토큰 수와 무관한 API/네트워크 지연"인지 구분한다.
    usage = getattr(response, "usage_metadata", None) or {}
    raw_usage = (getattr(response, "response_metadata", {}) or {}).get("token_usage", {})
    print(f"🔢 [진단] 토큰 사용량: {usage or raw_usage}")

    answer_text = clean_markdown(response.content)
    print(f"\n🤖 살레봇: {answer_text}\n")
    last_question, last_answer = user_question, answer_text  # /신고용으로 직전 대화 기억