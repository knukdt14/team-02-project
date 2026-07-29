"""
한국어-영어 법령 청크의 번역 무결성 판정을 공통으로 제공합니다.

이 모듈을 번역기와 최종 검증기가 함께 사용하도록 만든 이유는 두 프로그램의
합격 기준이 달라 같은 청크를 계속 재번역하는 문제를 막기 위해서입니다.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


# 숫자를 영어 단어로 자연스럽게 번역한 경우에 허용할 기본 별칭입니다.
_CARDINAL_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}

_ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}

_FREQUENCY_WORDS = {
    1: ("once",),
    2: ("twice",),
    3: ("thrice",),
}

_MONTH_WORDS = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}

# 법령 별표에서 사용하는 한글 목 번호입니다. 영어 번역문에 남아 있어도
# 내용 미번역이 아니라 행 식별자이므로 실패가 아닌 확인 경고로 분류합니다.
_KOREAN_ITEM_LABELS = set(
    "가나다라마바사아자차카타파하"
    "거너더러머버서어저처커터퍼허"
    "고노도로모보소오조초코토포호"
    "구누두루무부수우주추쿠투푸후"
    "그느드르므브스으즈츠크트프흐"
    "기니디리미비시이지치키티피히"
    "갸샤야"
)

_DIGIT_PATTERN = re.compile(
    # 뒤에 mg, ppm 같은 단위가 바로 붙어도 소수 전체를 잡아야 합니다.
    # 다만 img23019821처럼 영문 식별자 안에 붙은 숫자는 제외합니다.
    r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?(?!\d|[.,]\d)"
)

_KOREAN_NUMBER_ATOM = r"\d[\d,]*(?:\.\d+)?"
_KOREAN_SCALED_PATTERN = re.compile(
    rf"(?<!\d){_KOREAN_NUMBER_ATOM}"
    rf"(?:[억만천](?:{_KOREAN_NUMBER_ATOM})?)*[억만천](?!\d)"
)

_KOREAN_CURRENCY_PATTERN = re.compile(
    rf"(?<!\d)({_KOREAN_NUMBER_ATOM}"
    rf"(?:[억만천](?:{_KOREAN_NUMBER_ATOM})?)*[억만천]|"
    rf"{_KOREAN_NUMBER_ATOM})원"
)

_ENGLISH_CURRENCY_PATTERNS = (
    re.compile(
        r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*"
        r"(billion|million|thousand)?\s*(?:korean\s+)?(?:won|krw)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:krw)\s*(\d[\d,]*(?:\.\d+)?)\s*"
        r"(billion|million|thousand)?\b",
        re.IGNORECASE,
    ),
)


def _decimal_value(raw: str) -> Decimal | None:
    """쉼표와 불필요한 앞자리 0을 제거한 Decimal 값을 반환합니다."""
    try:
        return Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _value_key(value: Decimal) -> str:
    """동일한 수치가 같은 문자열 키를 갖도록 정규화합니다."""
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def number_values(text: str) -> set[str]:
    """본문에 직접 적힌 숫자를 수치 기준으로 정규화해 추출합니다."""
    result: set[str] = set()
    for match in _DIGIT_PATTERN.finditer(text):
        value = _decimal_value(match.group())
        if value is not None:
            result.add(_value_key(value))
    # 조문 번역에서 ①~⑳을 그대로 사용한 경우도 1~20과 같은 값입니다.
    for character in text:
        codepoint = ord(character)
        if 0x2460 <= codepoint <= 0x2473:
            result.add(str(codepoint - 0x2460 + 1))
    return result


def _parse_small_korean_number(text: str) -> Decimal:
    """'1천500'처럼 천 단위가 섞인 아라비아 숫자 표현을 계산합니다."""
    cleaned = text.replace(",", "")
    if not cleaned:
        return Decimal(0)
    if "천" not in cleaned:
        return Decimal(cleaned)
    left, right = cleaned.split("천", 1)
    thousands = Decimal(left or "1") * 1000
    remainder = Decimal(right or "0")
    return thousands + remainder


def _parse_korean_scaled_number(text: str) -> Decimal:
    """'3천억원', '1천500만원', '1만3천명'의 실제 숫자값을 계산합니다."""
    cleaned = text.replace(",", "")
    total = Decimal(0)

    if "억" in cleaned:
        left, cleaned = cleaned.split("억", 1)
        total += _parse_small_korean_number(left) * Decimal(100_000_000)
    if "만" in cleaned:
        left, cleaned = cleaned.split("만", 1)
        total += _parse_small_korean_number(left) * Decimal(10_000)
    total += _parse_small_korean_number(cleaned)
    return total


def _english_currency_values(text: str) -> set[str]:
    """영문 번역의 원화 금액을 원 단위 값으로 정규화합니다."""
    multipliers = {
        "": Decimal(1),
        "thousand": Decimal(1_000),
        "million": Decimal(1_000_000),
        "billion": Decimal(1_000_000_000),
    }
    result: set[str] = set()
    for pattern in _ENGLISH_CURRENCY_PATTERNS:
        for match in pattern.finditer(text):
            number = _decimal_value(match.group(1))
            if number is None:
                continue
            scale = str(match.group(2) or "").lower()
            result.add(_value_key(number * multipliers[scale]))
    return result


def _covered_source_number_spans(
    source_text: str,
    target_text: str,
) -> list[tuple[int, int]]:
    """
    표기는 달라도 같은 수치임을 확인한 한국어 숫자 표현의 범위를 반환합니다.

    예:
    - 10억원 ↔ 1 billion won
    - 1천500만원 ↔ 15 million won
    - 7천볼트 ↔ 7,000 volts
    """
    covered: list[tuple[int, int]] = []
    target_numbers = number_values(target_text)
    target_currency = _english_currency_values(target_text)

    for match in _KOREAN_CURRENCY_PATTERN.finditer(source_text):
        amount_text = match.group(1)
        if any(unit in amount_text for unit in ("억", "만", "천")):
            amount = _parse_korean_scaled_number(amount_text)
        else:
            amount = _decimal_value(amount_text) or Decimal(0)
        if _value_key(amount) in target_currency:
            covered.append(match.span(1))

    for match in _KOREAN_SCALED_PATTERN.finditer(source_text):
        # 금액 표현은 위에서 원 단위까지 확인하므로 여기서 중복 처리하지 않습니다.
        if match.end() < len(source_text) and source_text[match.end()] == "원":
            continue
        amount = _parse_korean_scaled_number(match.group())
        if _value_key(amount) in target_numbers:
            covered.append(match.span())

    return covered


def _inside_any_span(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _ignored_markup_spans(text: str) -> list[tuple[int, int]]:
    """HTML 태그와 URL 내부의 파일 식별 숫자를 법령 수치 비교에서 제외합니다."""
    spans = [match.span() for match in re.finditer(r"<[^>]+>", text)]
    spans.extend(match.span() for match in re.finditer(r"https?://\S+", text))
    return spans


def _source_date_spans(source_text: str, target_text: str) -> list[tuple[int, int]]:
    """'2024. 6. 28.'과 'June 28, 2024'를 같은 날짜로 판정합니다."""
    lowered_target = target_text.lower()
    covered: list[tuple[int, int]] = []
    date_pattern = re.compile(
        r"(?<!\d)(\d{4})\s*[년.]\s*(\d{1,2})\s*[월.]\s*(\d{1,2})\s*일?\.?"
    )
    for match in date_pattern.finditer(source_text):
        year, month, day = (int(value) for value in match.groups())
        month_word = _MONTH_WORDS.get(month, "")
        if (
            str(year) in number_values(target_text)
            and str(day) in number_values(target_text)
            and month_word
            and re.search(rf"\b{month_word}\b", lowered_target)
        ):
            covered.append(match.span())
    return covered


def _source_table_split_spans(
    source_text: str,
    target_text: str,
) -> list[tuple[int, int]]:
    """
    PDF 표 추출 때문에 35℃가 `3│다른 셀│5℃`로 갈라진 경우를 복구합니다.

    번역문에 결합된 값(35)이 실제로 있을 때만 두 조각을 보존된 것으로 봅니다.
    """
    target_numbers = number_values(target_text)
    covered: list[tuple[int, int]] = []
    pattern = re.compile(
        r"(?P<a>\d+)\s*[│┃][^℃\n]{0,120}?[│┃]\s*(?P<b>\d+)\s*℃"
    )
    for match in pattern.finditer(source_text):
        combined = str(int(match.group("a") + match.group("b")))
        if combined in target_numbers:
            covered.append(match.span("a"))
            covered.append(match.span("b"))
    return covered


def _is_allowed_alias(
    value: int,
    source_text: str,
    target_text: str,
    start: int,
    end: int,
) -> bool:
    """숫자가 영어 단어·월명·per 표현으로 바뀐 정상 번역을 허용합니다."""
    lowered_target = target_text.lower()
    aliases = []
    if value in _CARDINAL_WORDS:
        aliases.append(_CARDINAL_WORDS[value])
    if value in _ORDINAL_WORDS:
        aliases.append(_ORDINAL_WORDS[value])
    aliases.extend(_FREQUENCY_WORDS.get(value, ()))
    if any(re.search(rf"\b{re.escape(alias)}\b", lowered_target) for alias in aliases):
        return True

    # 6월 → June처럼 월 숫자가 이름으로 번역된 경우입니다.
    after = source_text[end : end + 2]
    if after.startswith("월") and value in _MONTH_WORDS:
        if re.search(rf"\b{_MONTH_WORDS[value]}\b", lowered_target):
            return True

    # 1개소당·1명당·1대당은 영어에서 보통 per location/person/unit입니다.
    around = source_text[max(0, start - 2) : min(len(source_text), end + 5)]
    if value == 1 and "당" in around and re.search(r"\bper\b", lowered_target):
        return True

    # 1분에·1시간에처럼 단위당 비율을 영어의 per minute/hour로 바꾼 경우입니다.
    unit_aliases = {
        "분": "minute",
        "시간": "hour",
        "일": "day",
        "주": "week",
        "개월": "month",
        "년": "year",
    }
    if value == 1:
        suffix = source_text[end : end + 5]
        for korean_unit, english_unit in unit_aliases.items():
            if suffix.startswith(korean_unit) and re.search(
                rf"\bper\s+{english_unit}\b",
                lowered_target,
            ):
                return True

    # PDF 표의 열 구분선 사이로 금액 단위가 떨어진 경우도 실제 원화값을 비교합니다.
    compact_tail = re.sub(
        r"[\s│┃|]+",
        "",
        source_text[start : min(len(source_text), end + 120)],
    ).replace(",", "")
    raw_number = source_text[start:end].replace(",", "")
    for korean_unit, multiplier in (("억원", 100_000_000), ("만원", 10_000), ("천원", 1_000)):
        if compact_tail.startswith(raw_number + korean_unit):
            target_amount = _value_key(Decimal(raw_number) * multiplier)
            if target_amount in _english_currency_values(target_text):
                return True

    # 원문의 ×106은 PDF에서 위첨자 6이 평문으로 붙은 10⁶ 표기일 수 있습니다.
    if value == 106:
        prefix = source_text[max(0, start - 2) : start]
        if "×" in prefix and re.search(r"10\s*[⁶6]", target_text):
            return True

    # 100분의 80은 영어에서 자연스럽게 80%로 축약됩니다.
    fraction = re.search(
        rf"(?<!\d){value}\s*분의\s*(\d+(?:\.\d+)?)",
        source_text[max(0, start - 2) : min(len(source_text), end + 15)],
    )
    if fraction and re.search(
        rf"(?<!\d){re.escape(fraction.group(1))}\s*(?:%|percent)\b",
        lowered_target,
    ):
        return True

    return False


def missing_number_values(source_text: str, target_text: str) -> list[str]:
    """
    번역문에서 의미상 보존되지 않은 숫자만 반환합니다.

    단순 문자열 일치가 아니라 금액 단위 환산, 한국식 만·억 단위, 날짜의 영문
    월명, 작은 수의 영단어 표기를 고려합니다.
    """
    target_numbers = number_values(target_text)
    semantic_spans = _covered_source_number_spans(source_text, target_text)
    semantic_spans.extend(_source_date_spans(source_text, target_text))
    semantic_spans.extend(_source_table_split_spans(source_text, target_text))
    ignored_spans = _ignored_markup_spans(source_text)
    missing: set[str] = set()

    # 기존 검증과 마찬가지로 숫자의 반복 횟수보다 서로 다른 법령 값의 보존을
    # 확인합니다. 같은 값의 여러 출현 중 하나가 의미상 보존되면 합격입니다.
    occurrences: dict[str, list[tuple[re.Match[str], Decimal]]] = {}
    for match in _DIGIT_PATTERN.finditer(source_text):
        value = _decimal_value(match.group())
        if value is not None:
            occurrences.setdefault(_value_key(value), []).append((match, value))

    for key, items in occurrences.items():
        if key in target_numbers:
            continue

        preserved = False
        for match, value in items:
            raw = match.group()
            if _inside_any_span(match.start(), match.end(), ignored_spans):
                preserved = True
                break
            if _inside_any_span(match.start(), match.end(), semantic_spans):
                preserved = True
                break

            # 별표0014의00과 같은 내부 식별자의 00은 내용상 숫자가 아닙니다.
            if value == 0 and set(raw.replace(",", "").replace(".", "")) == {"0"}:
                prefix = source_text[max(0, match.start() - 30) : match.start()]
                if re.search(r"(?:별표|별지|서식)\d+의$", prefix):
                    preserved = True
                    break

            if value == value.to_integral_value() and _is_allowed_alias(
                int(value),
                source_text,
                target_text,
                match.start(),
                match.end(),
            ):
                preserved = True
                break

        if not preserved:
            missing.add(key)

    return sorted(missing, key=lambda item: (Decimal(item), item))


def classify_hangul(text_en: str) -> tuple[list[str], list[str]]:
    """
    영어 본문의 한글을 실제 미번역 단어와 별표 항목 기호로 나눕니다.

    반환값:
        (실제 미번역 후보, 허용 가능한 표 항목 기호)
    """
    hard: set[str] = set()
    labels: set[str] = set()
    for token in re.findall(r"[가-힣]+", text_en):
        if len(token) == 1 and token in _KOREAN_ITEM_LABELS:
            labels.add(token)
        else:
            hard.add(token)
    return sorted(hard), sorted(labels)


def translation_issues(source_text: str, target_text: str) -> dict[str, list[str]]:
    """번역기와 검증기가 함께 사용하는 한 행의 오류 판정 결과입니다."""
    missing = missing_number_values(source_text, target_text)
    hard_hangul, label_hangul = classify_hangul(target_text)
    return {
        "missing_numbers": missing,
        "untranslated_hangul": hard_hangul,
        "table_label_hangul": label_hangul,
    }
