"""
살레봇 자동 정보 수집 모듈
- NEIS Open API: 학교 기본정보, 급식, 학사일정, 고등학교 시간표(학년/반 설정 필요)
- 공식 학교 홈페이지: 공지사항/가정통신문 링크를 찾아 본문을 수집(사이트 구조가 바뀌면 조정 필요)
- SQLite 저장, 실패 시 기존 DB 자료 유지
"""
import os
import re
import json
import time
import sqlite3
import logging
import threading
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

DB_PATH = os.getenv("SALEBOT_DB_PATH", "school_data.db")
SCHOOL_NAME = os.getenv("SCHOOL_NAME", "살레시오고등학교")
ATPT_CODE = os.getenv("NEIS_ATPT_CODE", "F10")  # 광주광역시교육청 코드
SCHOOL_HOME = os.getenv("SCHOOL_HOME_URL", "https://salesio.jge.hs.kr/main/main.php")
NEIS_KEY = os.getenv("NEIS_API_KEY", "").strip()
SYNC_INTERVAL_SECONDS = int(os.getenv("SYNC_INTERVAL_SECONDS", "86400"))
REQUEST_TIMEOUT = int(os.getenv("DATA_REQUEST_TIMEOUT", "15"))
LOG = logging.getLogger("salebot_sync")

NEIS_BASE = "https://open.neis.go.kr/hub"


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS school_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            record_date TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            UNIQUE(category, record_date, title, url)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_status (
            source TEXT PRIMARY KEY,
            last_success TEXT NOT NULL,
            last_message TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.commit()
    return conn


def _upsert_records(records):
    if not records:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    conn = db_connect()
    try:
        count = 0
        for r in records:
            conn.execute("""
                INSERT INTO school_records
                    (category, record_date, title, content, url, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, record_date, title, url)
                DO UPDATE SET content=excluded.content, source=excluded.source,
                              updated_at=excluded.updated_at
            """, (
                str(r.get("category", "일반")),
                str(r.get("record_date", "")),
                str(r.get("title", ""))[:500],
                str(r.get("content", ""))[:12000],
                str(r.get("url", ""))[:2000],
                str(r.get("source", ""))[:200],
                now,
            ))
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def _mark_success(source, message=""):
    conn = db_connect()
    try:
        conn.execute("""
            INSERT INTO sync_status(source, last_success, last_message)
            VALUES (?, ?, ?)
            ON CONFLICT(source) DO UPDATE SET
                last_success=excluded.last_success, last_message=excluded.last_message
        """, (source, datetime.now().isoformat(timespec="seconds"), message[:1000]))
        conn.commit()
    finally:
        conn.close()


def _fetch_json(endpoint, params):
    if not NEIS_KEY:
        raise RuntimeError("NEIS_API_KEY 환경변수가 설정되지 않았습니다.")
    from urllib.parse import urlencode
    query = dict(params)
    query.update({"KEY": NEIS_KEY, "Type": "json", "pIndex": 1, "pSize": 1000})
    url = f"{NEIS_BASE}/{endpoint}?{urlencode(query)}"
    req = Request(url, headers={"User-Agent": "SalebotSchoolAssistant/1.0"})
    with urlopen(req, timeout=REQUEST_TIMEOUT) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8-sig"))
    # NEIS는 정상 결과도 root 배열의 첫 번째 요소에 RESULT를 넣어 반환할 수 있음.
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                result = value[0].get("head", [{}])
                if result and isinstance(result, list):
                    result_info = result[0].get("RESULT", {}) if isinstance(result[0], dict) else {}
                    code = result_info.get("CODE")
                    if code and code not in ("INFO-000",):
                        raise RuntimeError(f"NEIS {endpoint} 오류 {code}: {result_info.get('MESSAGE', '')}")
                rows = []
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("row"), list):
                        rows.extend(item["row"])
                if rows:
                    return rows
        # 오류 payload에서 코드 추출
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "head" in item:
                        for h in item["head"]:
                            result = h.get("RESULT", {}) if isinstance(h, dict) else {}
                            if result.get("CODE") and result.get("CODE") != "INFO-000":
                                raise RuntimeError(f"NEIS 오류 {result.get('CODE')}: {result.get('MESSAGE', '')}")
        return []
    return []


def _find_school():
    rows = _fetch_json("schoolInfo", {
        "SCHUL_NM": SCHOOL_NAME,
    })
    # 학교명 완전 일치 우선
    exact = [r for r in rows if re.sub(r"\s+", "", r.get("SCHUL_NM", "")) ==
             re.sub(r"\s+", "", SCHOOL_NAME)]
    candidates = exact or rows
    if not candidates:
        raise RuntimeError(f"NEIS에서 '{SCHOOL_NAME}' 학교를 찾지 못했습니다.")
    row = candidates[0]
    school_code = row.get("SD_SCHUL_CODE", "")
    if not school_code:
        raise RuntimeError("학교 기본정보 응답에 SD_SCHUL_CODE가 없습니다.")
    return school_code, row


def _neis_school_record(row):
    fields = [
        ("학교명", "SCHUL_NM"), ("학교종류", "SCHUL_KND_SC_NM"),
        ("주소", "ORG_RDNMA"), ("대표전화", "ORG_TELNO"),
        ("홈페이지", "HMPG_ADRES"), ("남녀공학", "COEDU_SC_NM"),
        ("주야구분", "DGHT_SC_NM"), ("개교기념일", "FOND_YMD"),
    ]
    parts = []
    for label, key in fields:
        value = row.get(key, "")
        if value:
            parts.append(f"{label}: {value}")
    return {
        "category": "학교기본정보", "record_date": "",
        "title": "살레시오고등학교 기본정보",
        "content": " / ".join(parts) or json.dumps(row, ensure_ascii=False),
        "url": SCHOOL_HOME, "source": "NEIS SchoolInfo",
    }


def _sync_school_info():
    code, row = _find_school()
    _upsert_records([_neis_school_record(row)])
    _mark_success("neis_schoolinfo", f"학교코드={code}")
    return code


def _sync_meals(school_code, start=None, end=None):
    start = start or datetime.now().date()
    end = end or (start + timedelta(days=60))
    rows = _fetch_json("mealServiceDietInfo", {
        "ATPT_OFCDC_SC_CODE": ATPT_CODE,
        "SD_SCHUL_CODE": school_code,
        "MLSV_FROM_YMD": start.strftime("%Y%m%d"),
        "MLSV_TO_YMD": end.strftime("%Y%m%d"),
    })
    records = []
    for row in rows:
        ymd = row.get("MLSV_YMD", "")
        if not re.fullmatch(r"\d{8}", ymd):
            continue
        date_label = f"{ymd[:4]}년 {int(ymd[4:6])}월 {int(ymd[6:])}일"
        meal_type = row.get("MMEAL_SC_NM", "")
        menu = re.sub(r"<br\s*/?>", ", ", row.get("DDISH_NM", ""), flags=re.I)
        menu = re.sub(r"\s+", " ", menu).strip()
        content = f"{date_label} {meal_type} 급식 메뉴: {menu}"
        if row.get("CAL_INFO"):
            content += f" / 칼로리: {row['CAL_INFO']}"
        if row.get("ORPLC_INFO"):
            content += f" / 원산지: {row['ORPLC_INFO']}"
        records.append({
            "category": "급식", "record_date": ymd, "title": f"{date_label} {meal_type}",
            "content": content, "url": "https://open.neis.go.kr/", "source": "NEIS mealServiceDietInfo",
        })
    if records:
        _upsert_records(records)
        _mark_success("neis_meals", f"{len(records)}건 저장")
    else:
        # 빈 응답은 '급식 없음'과 API 오류를 구별하기 어려우므로 성공 상태를 갱신하지 않음.
        LOG.warning("NEIS 급식 응답이 비어 있어 기존 DB를 유지합니다.")
    return len(records)


def _sync_schedule(school_code, start=None, end=None):
    start = start or datetime.now().date()
    end = end or (start + timedelta(days=180))
    rows = _fetch_json("SchoolSchedule", {
        "ATPT_OFCDC_SC_CODE": ATPT_CODE,
        "SD_SCHUL_CODE": school_code,
        "AA_FROM_YMD": start.strftime("%Y%m%d"),
        "AA_TO_YMD": end.strftime("%Y%m%d"),
    })
    records = []
    for row in rows:
        ymd = row.get("AA_YMD", "")
        if not re.fullmatch(r"\d{8}", ymd):
            continue
        date_label = f"{ymd[:4]}년 {int(ymd[4:6])}월 {int(ymd[6:])}일"
        title = row.get("EVENT_NM", "학교 일정")
        content = f"{date_label} {title}"
        if row.get("EVENT_CNTNT"):
            content += f" / 내용: {row['EVENT_CNTNT']}"
        grade = row.get("ONE_GRADE_EVENT_YN", "")
        if grade:
            content += f" / 1학년 행사 여부: {grade}"
        records.append({
            "category": "학사일정", "record_date": ymd, "title": title,
            "content": content, "url": "https://open.neis.go.kr/", "source": "NEIS SchoolSchedule",
        })
    if records:
        _upsert_records(records)
        _mark_success("neis_schedule", f"{len(records)}건 저장")
    else:
        LOG.warning("NEIS 학사일정 응답이 비어 있어 기존 DB를 유지합니다.")
    return len(records)


def _sync_timetable(school_code):
    """
    시간표는 학년/반별 API라 요청 반 목록을 환경변수로 설정해야 합니다.
    예: TIMETABLE_CLASSES=1:1,1:2,1:3,2:1,2:2,3:1
    과도한 호출을 피하기 위해 설정되지 않으면 건너뛰고 기존 DB를 유지합니다.
    """
    configured = os.getenv("TIMETABLE_CLASSES", "").strip()
    if not configured:
        LOG.warning("TIMETABLE_CLASSES 미설정: 시간표 API 수집을 건너뜁니다.")
        return 0
    pairs = []
    for part in configured.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", part)
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
    if not pairs:
        LOG.warning("TIMETABLE_CLASSES 형식이 올바르지 않습니다. 예: 1:1,1:2,2:1")
        return 0

    today = datetime.now().date()
    # 최근 5일~향후 14일의 시간표를 요청한다. 학교 API의 범위/파라미터 정책에 따라 조정 가능.
    start = today - timedelta(days=5)
    end = today + timedelta(days=14)
    records = []
    for grade, class_no in pairs:
        try:
            rows = _fetch_json("hisTimetable", {
                "ATPT_OFCDC_SC_CODE": ATPT_CODE,
                "SD_SCHUL_CODE": school_code,
                "GRADE": grade,
                "CLASS_NM": class_no,
                "TI_FROM_YMD": start.strftime("%Y%m%d"),
                "TI_TO_YMD": end.strftime("%Y%m%d"),
            })
            for row in rows:
                ymd = row.get("ALL_TI_YMD", "")
                if not re.fullmatch(r"\d{8}", ymd):
                    continue
                date_label = f"{ymd[:4]}년 {int(ymd[4:6])}월 {int(ymd[6:])}일"
                period = row.get("PERIO", "")
                subject = row.get("ITRT_CNTNT", "")
                room = row.get("CLRM_NM", "")
                title = f"{grade}학년 {class_no}반 {period}교시"
                content = f"{date_label} {title}: {subject}"
                if room:
                    content += f" / 교실: {room}"
                records.append({
                    "category": "시간표", "record_date": ymd, "title": title,
                    "content": content, "url": "https://open.neis.go.kr/", "source": "NEIS hisTimetable",
                })
            time.sleep(0.15)
        except Exception as exc:
            LOG.warning("시간표 수집 실패 (%s학년 %s반): %s", grade, class_no, exc)
    if records:
        _upsert_records(records)
        _mark_success("neis_timetable", f"{len(records)}건 저장")
    return len(records)


def _clean_html(html):
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()



def _crawl_official_site():
    """
    살레시오고등학교 공식 홈페이지의
    학생공지, 학교공지, 가정통신문을 수집합니다.
    """
    if not SCHOOL_HOME:
        return 0

    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 설치가 필요합니다.")

    headers = {"User-Agent": "Mozilla/5.0"}
    base_url = "https://salesio.jge.hs.kr"

    boards = [
        ("학생공지", 25),
        ("학교공지", 43),
        ("가정통신문", 44),
    ]

    records = []
    seen = set()

    for category, tbnum in boards:
        board_url = f"{base_url}/xboard/board.php?tbnum={tbnum}"

        try:
            req = Request(board_url, headers=headers)
            with urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                html = response.read().decode("utf-8", errors="replace")

            soup = BeautifulSoup(html, "html.parser")

            # 게시글 상세 페이지 링크 수집
            links = []

            for a in soup.select("a[href]"):
                href = a.get("href", "")

                if "mode=view" not in href or "number=" not in href:
                    continue

                detail_url = urljoin(board_url, href)

                if f"tbnum={tbnum}" not in detail_url:
                    continue

                if detail_url not in seen:
                    seen.add(detail_url)
                    links.append(detail_url)

            # 게시판별 최신 게시글 최대 15개
            for detail_url in links[:15]:
                try:
                    req = Request(detail_url, headers=headers)

                    with urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                        page = response.read().decode(
                            "utf-8", errors="replace"
                        )

                    detail = BeautifulSoup(page, "html.parser")

                    view = detail.select_one("#xb_view")
                    if not view:
                        continue

                    info = view.select_one(".new_board_view")
                    body = view.select_one(".board_content")

                    if not info:
                        continue


                    # 게시글 제목
                    title_tag = info.select_one("h4.title")
                    title = ""

                    if title_tag:
                        title_tag.select_one("p.indent").decompose() if title_tag.select_one("p.indent") else None
                        title = title_tag.get_text(" ", strip=True)

                    # 게시글 본문
                    content_tag = view.select_one(
                        ".board_content .board_img_view_con"
                    )

                    content = (
                        content_tag.get_text(" ", strip=True)
                        if content_tag else ""
                    )

                    # 작성일
                    record_date = ""

                    for li in info.select("ul.v_info li"):
                        label = li.find("span")
                        if label and "작성일" in label.get_text(strip=True):
                            date_match = re.search(
                                r"\d{4}-\d{2}-\d{2}",
                                li.get_text(" ", strip=True)
                            )
                            if date_match:
                                record_date = date_match.group(0)
                            break


                    if not title:
                        continue

                    records.append({
                        "category": category,
                        "record_date": record_date,
                        "title": title[:300],
                        "content": content[:12000],
                        "url": detail_url,
                        "source": "살레시오고등학교 공식 홈페이지",
                    })

                except Exception as exc:
                    LOG.warning(
                        "게시글 수집 실패 (%s): %s",
                        detail_url, exc
                    )

        except Exception as exc:
            LOG.warning(
                "%s 게시판 접속 실패: %s",
                category, exc
            )

    if records:
        _upsert_records(records)
        _mark_success(
            "official_site",
            f"{len(records)}개 게시글 저장"
        )
        LOG.info("홈페이지 게시글 %s개 수집 완료", len(records))
    else:
        LOG.warning("홈페이지에서 게시글을 수집하지 못했습니다.")

    return len(records)



def sync_all():
    """모든 출처를 수집. 개별 출처 실패는 다른 출처의 수집을 막지 않는다."""
    summary = {}
    school_code = None
    try:
        school_code = _sync_school_info()
        summary["학교기본정보"] = "성공"
    except Exception as exc:
        summary["학교기본정보"] = f"실패: {exc}"
        LOG.exception("학교 기본정보 수집 실패")

    if school_code:
        for label, fn in [
            ("급식", _sync_meals),
            ("학사일정", _sync_schedule),
            ("시간표", _sync_timetable),
        ]:
            try:
                count = fn(school_code)
                summary[label] = f"{count}건"
            except Exception as exc:
                summary[label] = f"실패: {exc}"
                LOG.exception("%s 수집 실패", label)
    else:
        summary.update({"급식": "건너뜀(학교 코드 없음)", "학사일정": "건너뜀(학교 코드 없음)",
                        "시간표": "건너뜀(학교 코드 없음)"})

    try:
        summary["학교 홈페이지"] = f"{_crawl_official_site()}건"
    except Exception as exc:
        summary["학교 홈페이지"] = f"실패: {exc}"
        LOG.exception("학교 홈페이지 수집 실패")

    LOG.info("자동 수집 결과: %s", summary)
    return summary


def get_records(category=None, limit=5000):
    conn = db_connect()
    try:
        if category:
            rows = conn.execute(
                "SELECT * FROM school_records WHERE category=? ORDER BY record_date, title LIMIT ?",
                (category, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM school_records ORDER BY category, record_date, title LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_context_texts():
    """기존 날짜 검색 코드에 합칠 수 있도록 DB 레코드를 두 개의 텍스트 묶음으로 변환."""
    records = get_records()
    info_categories = {"학교기본정보", "학사일정", "시간표", "학교공지", "가정통신문"}
    info_lines, menu_lines = [], []
    for row in records:
        content = row.get("content", "").strip()
        if not content:
            continue
        if row["category"] == "급식":
            menu_lines.append(content)
        elif row["category"] in info_categories:
            info_lines.append(content)
    return "\n".join(info_lines), "\n".join(menu_lines)


def last_sync_age_seconds():
    conn = db_connect()
    try:
        rows = conn.execute("SELECT last_success FROM sync_status").fetchall()
        if not rows:
            return None
        parsed = []
        for row in rows:
            try:
                parsed.append(datetime.fromisoformat(row["last_success"]))
            except Exception:
                pass
        return (datetime.now() - max(parsed)).total_seconds() if parsed else None
    finally:
        conn.close()


def sync_if_due(force=False):
    age = last_sync_age_seconds()
    if force or age is None or age >= SYNC_INTERVAL_SECONDS:
        return sync_all()
    LOG.info("마지막 성공 수집 후 %.1f시간 경과: 아직 갱신 시간이 아닙니다.", age / 3600)
    return {"상태": "최근 24시간 내 수집 기록이 있어 건너뜀"}


def start_daily_scheduler(refresh_callback=None):
    """
    실행 중인 Python 프로세스 안에서 하루마다 수집합니다.
    Replit 프로세스가 sleep/restart 되면 실행이 멈출 수 있으므로,
    별도의 Scheduled Deployment에서 `python sync_data.py`를 매일 실행하는 것을 권장합니다.
    """
    def worker():
        while True:
            time.sleep(SYNC_INTERVAL_SECONDS)
            try:
                result = sync_all()
                if refresh_callback:
                    refresh_callback()
                LOG.info("일일 백그라운드 수집 완료: %s", result)
            except Exception:
                LOG.exception("일일 백그라운드 수집 중 예외")
    thread = threading.Thread(target=worker, daemon=True, name="salebot-daily-sync")
    thread.start()
    return thread
