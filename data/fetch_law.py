"""
fetch_law.py  (원래 예시 구조의 load_pdf.py 자리)
---------------------------------------------------
국가법령정보센터 Open API(DRF)로 산업안전 관련 법령을 수집하고,
전처리 후 'RAG용 chunk'로 만들어 data/ 폴더에 저장한다.

수집 대상 (시행규칙 제외):
  - 산업안전보건법
  - 산업안전보건법 시행령
  - 산업안전보건기준에 관한 규칙

적용한 전처리:
  ① 조문 재분할 + 헤더 유지
     - 너무 긴 조문은 항(①②) 단위로 분할, 너무 짧은 조문은 다음 조문과 병합
     - 모든 chunk 맨 앞에 "[법령명 제N조(제목)]" 헤더를 붙여, 잘려도 맥락 유지
  ② 메타데이터 확장
     - 법령명 / 법령종류 / 장 / 절 / 조문번호 / 조문제목 / 시행일자
  ③ 별표(표) 별도 수집 (있으면)
  ④ 텍스트 정규화
     - <개정 ...> 표기 제거, 원문자/공백/특수문자 정리

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

# 수집할 법령 (시행규칙 제외 — 3개)
LAWS = [
    "산업안전보건법",
    "산업안전보건법 시행령",
    "산업안전보건기준에 관한 규칙",
]

# 재분할 기준: 이 글자 수를 넘는 조문은 항 단위로 쪼갠다
MAX_CHARS = 800
# 이 글자 수보다 짧은 조문은 다음 조문과 병합 시도
MIN_CHARS = 120

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


# ④ 텍스트 정규화 ------------------------------------------------------
_RE_AMEND = re.compile(r"<개정[^>]*>|<신설[^>]*>|<삭제[^>]*>|<제목개정[^>]*>")
_RE_MULTISPACE = re.compile(r"[ \t]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")


def normalize(text) -> str:
    if not text:
        return ""
    t = str(text)
    t = t.replace("\xa0", " ").replace("　", " ")  # 특수 공백 → 일반 공백
    t = _RE_AMEND.sub("", t)                           # <개정 2020. 1. 1.> 제거
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


# =====================================================================
# 2) MST → 조문 본문 조회
# =====================================================================
def fetch_law_body(mst: str) -> dict:
    params = {"OC": OC, "target": "law", "type": "JSON", "MST": mst}
    res = requests.get(SERVICE_URL, params=params, timeout=20)
    res.raise_for_status()
    return res.json().get("법령", {})


# 조문 파싱: 조문 하나를 (헤더, 본문, 항리스트, 메타) 형태로 정리
def parse_article(jo: dict, ctx: dict) -> dict | None:
    jo_no = normalize(jo.get("조문번호"))
    jo_title = normalize(jo.get("조문제목"))
    header = f"[{ctx['법령명']} 제{jo_no}조({jo_title})]" if jo_title \
        else f"[{ctx['법령명']} 제{jo_no}조]"

    # 항이 없으면 조문내용 자체가 본문
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

    body = "\n".join(t for t in hang_texts if t).strip()
    if not body:
        return None

    return {
        "header": header,
        "hang_texts": hang_texts,  # ① 재분할 시 사용
        "meta": {
            "법령명": ctx["법령명"],
            "법령종류": ctx["법령종류"],
            "장": normalize(jo.get("조문키") and ctx.get("장", "")) or ctx.get("장", ""),
            "조문번호": jo_no,
            "조문제목": jo_title,
            "시행일자": ctx["시행일자"],
        },
    }


# =====================================================================
# ① 재분할 + 헤더 유지 → 최종 chunk 리스트 생성
# =====================================================================
def build_chunks(articles: list[dict]) -> list[dict]:
    chunks = []
    buffer = None  # 짧은 조문 병합용

    def flush(buf):
        if buf:
            chunks.append(buf)

    for art in articles:
        full_body = "\n".join(art["hang_texts"])
        header = art["header"]
        meta = art["meta"]

        # (a) 긴 조문 → 항 단위 분할, 각 조각에 헤더 부착
        if len(full_body) > MAX_CHARS and len(art["hang_texts"]) > 1:
            flush(buffer)
            buffer = None
            for i, ht in enumerate(art["hang_texts"], 1):
                chunks.append({
                    "본문": f"{header} (항 {i})\n{ht}",
                    "내용": ht,
                    **meta,
                })
            continue

        # (b) 짧은 조문 → 버퍼에 모아 병합
        if len(full_body) < MIN_CHARS:
            piece = {"본문": f"{header}\n{full_body}", "내용": full_body, **meta}
            if buffer is None:
                buffer = piece
            else:
                buffer["본문"] += "\n\n" + piece["본문"]
                buffer["내용"] += "\n\n" + piece["내용"]
                if len(buffer["내용"]) >= MIN_CHARS:
                    flush(buffer)
                    buffer = None
            continue

        # (c) 적당한 길이 → 그대로 하나의 chunk
        flush(buffer)
        buffer = None
        chunks.append({"본문": f"{header}\n{full_body}", "내용": full_body, **meta})

    flush(buffer)
    return chunks


# =====================================================================
# ③ 별표 수집 (본문 응답에 포함된 경우)
# =====================================================================
def parse_tables(law: dict, ctx: dict) -> list[dict]:
    tables = []
    for byl in _as_list(law.get("별표", {}).get("별표단위") if isinstance(law.get("별표"), dict) else None):
        title = normalize(byl.get("별표제목"))
        content = normalize(byl.get("별표내용"))
        if not content:
            continue
        tables.append({
            "본문": f"[{ctx['법령명']} 별표: {title}]\n{content}",
            "내용": content,
            "법령명": ctx["법령명"],
            "법령종류": ctx["법령종류"],
            "장": "",
            "조문번호": "별표",
            "조문제목": title,
            "시행일자": ctx["시행일자"],
        })
    return tables


# =====================================================================
# 3) 실행
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

    # 장(章) 맥락을 조문에 전달하기 위해 조문단위를 순회
    articles = []
    for jo in _as_list(law.get("조문", {}).get("조문단위", [])):
        # 편/장/절 제목 행이면 맥락만 갱신하고 skip
        jo_title = normalize(jo.get("조문제목"))
        if normalize(jo.get("조문여부")) == "전문" or (jo_title and not jo.get("항") and not jo.get("조문내용")):
            if "장" in (jo_title or ""):
                ctx["장"] = jo_title
            continue
        a = parse_article(jo, ctx)
        if a:
            articles.append(a)

    chunks = build_chunks(articles)
    chunks += parse_tables(law, ctx)

    print(f"  → 조문 {len(articles)}개 → chunk {len(chunks)}개 (MST={mst})")
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

    print(f"\n완료: 총 chunk {len(all_chunks)}개 → {OUT_DIR/'laws_all.json'}")
    print("각 chunk 필드: 본문(임베딩 대상) / 내용 / 법령명·법령종류·장·조문번호·조문제목·시행일자(메타데이터)")


if __name__ == "__main__":
    main()
