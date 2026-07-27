"""
fetch_law.py  (v2 — 메타데이터 버그 수정판)
---------------------------------------------------
국가법령정보센터 Open API(DRF)로 산업안전 관련 법령을 수집하고,
전처리 후 'RAG용 chunk'로 만들어 data/ 폴더에 저장한다.

수집 대상 (시행규칙 제외):
  - 산업안전보건법
  - 산업안전보건법 시행령
  - 산업안전보건기준에 관한 규칙

[v2 수정 사항]
  1) 짧은 조문 병합 제거
     - 조문 1개 = 최소 문서 단위. 서로 다른 조문을 합치지 않는다.
       (병합 시 메타데이터가 앞 조문 번호만 남아 출처가 오염되는 버그 수정)
  2) 조문 가지번호 보존
     - "제619조의2"의 '의2'를 조문가지번호로 따로 저장하고,
       조문표시("제619조의2")를 별도 필드로 제공. (기존: 619로 유실)
  3) 장(章) 메타데이터 실제 수집
     - 조문여부="전문" 행의 "제N장 ..." 을 읽어 이후 조문에 장 정보를 부여.
       (기존: 전부 빈 문자열이던 버그 수정)
  4) 별표 구조 보존
     - 중첩 리스트를 문자열로 덤프하던 버그 수정. 행 단위 텍스트로 정리하고
       별표번호·별표가지번호를 메타데이터로 저장. 긴 별표는 행 단위로 분할.

적용 전처리:
  - 긴 조문(MAX_CHARS 초과)은 '같은 조문 안에서' 항 단위로만 분할
    (조문 간 병합은 하지 않으므로 메타데이터 오염 없음)
  - 모든 chunk 앞에 "[법령명 조문표시(제목)]" 헤더 유지
  - <개정 ...> 표기 제거, 공백 정리

[사용법]
1) https://open.law.go.kr 에서 OPEN API 활용신청 → OC 값 발급
2) 아래 OC 변수에 발급받은 값(보통 본인 이메일 아이디)을 넣는다
3) python fetch_law.py 실행
"""

import json
import re
import time
from pathlib import Path

import requests

# =====================================================================
# 설정
# =====================================================================

# ★ 여기에 발급받은 OC 인증키를 넣으세요 (예: OC = "hong1234")
OC = "9933005555446622"

LAWS = [
    "산업안전보건법",
    "산업안전보건법 시행령",
    "산업안전보건기준에 관한 규칙",
]

# 이 글자 수를 넘는 조문만 '항 단위'로 분할 (조문 간 병합은 없음)
MAX_CHARS = 800
# 별표를 이 글자 수 기준으로 행 단위 분할
TABLE_MAX_CHARS = 1500

SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
SLEEP = 0.5


# =====================================================================
# 공통 유틸
# =====================================================================
def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


_RE_AMEND = re.compile(r"<개정[^>]*>|<신설[^>]*>|<삭제[^>]*>|<제목개정[^>]*>")
_RE_MULTISPACE = re.compile(r"[ \t]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")
_RE_CHAPTER = re.compile(r"제\s*(\d+)\s*장(?:의\s*\d+)?\s*([^\n<]*)")


def normalize(text) -> str:
    if not text:
        return ""
    t = str(text)
    t = t.replace("\xa0", " ").replace("　", " ")
    t = _RE_AMEND.sub("", t)
    t = _RE_MULTISPACE.sub(" ", t)
    t = _RE_MULTINEWLINE.sub("\n\n", t)
    return t.strip()


# =====================================================================
# 1) 법령명 → 일련번호(MST) + 시행일자
# =====================================================================
def search_law(name: str) -> dict | None:
    params = {"OC": OC, "target": "law", "type": "JSON",
              "query": name, "display": "20"}
    res = requests.get(SEARCH_URL, params=params, timeout=20)
    res.raise_for_status()
    items = _as_list(res.json().get("LawSearch", {}).get("law", []))
    if not items:
        return None
    for it in items:
        if it.get("법령명한글", "").strip() == name:
            return it
    return items[0]


def fetch_law_body(mst: str) -> dict:
    params = {"OC": OC, "target": "law", "type": "JSON", "MST": mst}
    res = requests.get(SERVICE_URL, params=params, timeout=20)
    res.raise_for_status()
    return res.json().get("법령", {})


# =====================================================================
# 2) 조문 파싱 — 가지번호·장 보존, 병합 없음
# =====================================================================
def make_jo_label(jo_no: str, jo_gaji: str) -> str:
    """조문표시 생성: (619, 2) → '제619조의2' / (37, '') → '제37조'"""
    label = f"제{jo_no}조"
    if jo_gaji and jo_gaji not in ("0", ""):
        label += f"의{jo_gaji}"
    return label


def parse_article(jo: dict, ctx: dict) -> list[dict]:
    """
    조문 1개 → chunk 리스트 (병합 없음, 긴 조문만 항 단위 분할).
    반환되는 모든 chunk는 '이 조문'의 메타데이터만 가진다.
    """
    jo_no = normalize(jo.get("조문번호"))
    jo_gaji = normalize(jo.get("조문가지번호"))
    jo_label = make_jo_label(jo_no, jo_gaji)          # 예: 제619조의2
    jo_title = normalize(jo.get("조문제목"))

    header = f"[{ctx['법령명']} {jo_label}({jo_title})]" if jo_title \
        else f"[{ctx['법령명']} {jo_label}]"

    # 본문 수집: 조문내용 + 항(호 포함)
    hang_texts = []
    head = normalize(jo.get("조문내용"))
    if head:
        hang_texts.append(head)
    for hang in _as_list(jo.get("항")):
        h = normalize(hang.get("항내용"))
        parts = [h] if h else []
        for ho in _as_list(hang.get("호")):
            ho_txt = normalize(ho.get("호내용"))
            if ho_txt:
                parts.append("  " + ho_txt)
        if parts:
            hang_texts.append("\n".join(parts))

    full_body = "\n".join(t for t in hang_texts if t).strip()
    if not full_body:
        return []

    meta = {
        "법령명": ctx["법령명"],
        "법령종류": ctx["법령종류"],
        "장": ctx["장"],                    # v2: 실제 장 정보
        "조문번호": jo_no,
        "조문가지번호": jo_gaji,             # v2: 가지번호 보존
        "조문표시": jo_label,               # v2: '제619조의2' 형태
        "조문제목": jo_title,
        "시행일자": ctx["시행일자"],
    }

    # 짧은 조문 → 그대로 1개 chunk (병합하지 않음!)
    if len(full_body) <= MAX_CHARS or len(hang_texts) <= 1:
        return [{"본문": f"{header}\n{full_body}", "내용": full_body, **meta}]

    # 긴 조문 → '같은 조문 안에서' 항 단위 분할 (메타데이터 동일)
    chunks = []
    for i, ht in enumerate(hang_texts, 1):
        chunks.append({
            "본문": f"{header} (항 {i})\n{ht}",
            "내용": ht,
            **meta,
        })
    return chunks


# =====================================================================
# 3) 별표 파싱 — 중첩 리스트를 행 단위 텍스트로 복원
# =====================================================================
def _flatten_table(value) -> list[str]:
    """
    별표내용은 문자열/리스트/중첩 리스트가 섞여 온다.
    재귀적으로 풀어 '행 단위 텍스트 리스트'로 만든다.
    (기존: str()로 덤프해 ['...', '...'] 형태의 깨진 문자열이 되던 버그 수정)
    """
    rows = []
    if value is None:
        return rows
    if isinstance(value, str):
        v = normalize(value)
        if v:
            rows.append(v)
    elif isinstance(value, list):
        # 리스트의 항목이 전부 문자열이면 '한 행'으로 합침 (표의 한 줄)
        if all(isinstance(x, str) for x in value):
            line = " | ".join(normalize(x) for x in value if normalize(x))
            if line:
                rows.append(line)
        else:
            for x in value:
                rows.extend(_flatten_table(x))
    else:
        v = normalize(value)
        if v:
            rows.append(v)
    return rows


def parse_tables(law: dict, ctx: dict) -> list[dict]:
    tables = []
    byl_root = law.get("별표", {})
    units = byl_root.get("별표단위") if isinstance(byl_root, dict) else None

    for byl in _as_list(units):
        no = normalize(byl.get("별표번호"))
        gaji = normalize(byl.get("별표가지번호"))
        label = f"별표{no}" + (f"의{gaji}" if gaji and gaji not in ("0", "") else "")
        title = normalize(byl.get("별표제목"))

        rows = _flatten_table(byl.get("별표내용"))
        if not rows:
            continue

        meta = {
            "법령명": ctx["법령명"],
            "법령종류": ctx["법령종류"],
            "장": "",
            "조문번호": label,               # 예: 별표2 (기존: '별표'로 뭉뚱그림)
            "조문가지번호": gaji,
            "조문표시": label,
            "조문제목": title,
            "시행일자": ctx["시행일자"],
        }
        header = f"[{ctx['법령명']} {label}: {title}]"

        # 긴 별표는 행 단위로 나눠 여러 chunk 생성 (표 맥락 유지 위해 헤더 반복)
        buf, size, part = [], 0, 1
        for row in rows:
            if size + len(row) > TABLE_MAX_CHARS and buf:
                content = "\n".join(buf)
                tables.append({
                    "본문": f"{header} (부분 {part})\n{content}",
                    "내용": content, **meta,
                })
                buf, size, part = [], 0, part + 1
            buf.append(row)
            size += len(row)
        if buf:
            content = "\n".join(buf)
            suffix = f" (부분 {part})" if part > 1 else ""
            tables.append({
                "본문": f"{header}{suffix}\n{content}",
                "내용": content, **meta,
            })
    return tables


# =====================================================================
# 4) 법령 1건 처리
# =====================================================================
def process_law(name: str) -> list[dict]:
    info = search_law(name)
    if not info:
        print(f"  ! MST를 찾지 못함 — 법령명 확인: {name}")
        return []
    mst = info.get("법령일련번호")
    ctx = {
        "법령명": name,
        "법령종류": normalize(info.get("법령구분명")),
        "시행일자": normalize(info.get("시행일자")),
        "장": "",
    }
    time.sleep(SLEEP)

    law = fetch_law_body(mst)

    chunks = []
    n_articles = 0
    for jo in _as_list(law.get("조문", {}).get("조문단위", [])):
        # 장 헤더 행: 조문여부가 '전문'이거나, 내용이 '제N장 ...' 형태
        #  → 장 컨텍스트만 갱신하고 chunk는 만들지 않음
        content = normalize(jo.get("조문내용"))
        if normalize(jo.get("조문여부")) == "전문" or (
            content and not jo.get("항") and _RE_CHAPTER.match(content)
        ):
            m = _RE_CHAPTER.search(content)
            if m:
                ctx["장"] = f"제{m.group(1)}장 {m.group(2).strip()}".strip()
            continue

        arts = parse_article(jo, ctx)
        if arts:
            n_articles += 1
            chunks.extend(arts)

    tbl = parse_tables(law, ctx)
    chunks.extend(tbl)

    n_chapter = sum(1 for c in chunks if c.get("장"))
    print(f"  → 조문 {n_articles}개 → chunk {len(chunks)}개 "
          f"(별표 {len(tbl)}개, 장 정보 보유 {n_chapter}개, MST={mst})")
    return chunks


def main():
    if not OC:
        raise SystemExit(
            "OC 인증키가 비어 있습니다. open.law.go.kr에서 발급 후 "
            "파일 상단의 OC 변수에 넣어주세요."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for name in LAWS:
        print(f"[수집] {name}")
        chunks = process_law(name)
        all_chunks.extend(chunks)

        safe = name.replace(" ", "_")
        with open(OUT_DIR / f"{safe}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        time.sleep(SLEEP)

    with open(OUT_DIR / "laws_all.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # 수집 품질 리포트
    gaji = sum(1 for c in all_chunks if c.get("조문가지번호") not in ("", "0", None))
    chapters = sum(1 for c in all_chunks if c.get("장"))
    tables = sum(1 for c in all_chunks if str(c.get("조문표시", "")).startswith("별표"))
    print(f"\n완료: 총 chunk {len(all_chunks)}개 → {OUT_DIR/'laws_all.json'}")
    print(f"  가지번호 보존: {gaji}개 / 장 정보 보유: {chapters}개 / 별표 chunk: {tables}개")
    print("각 chunk 필드: 본문(임베딩 대상) / 내용 / "
          "법령명·법령종류·장·조문번호·조문가지번호·조문표시·조문제목·시행일자")


if __name__ == "__main__":
    main()
