from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER_WITH_UNIT_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)\s*"
    r"(%|[kKmMbB]\b|тыс\.?|млн\.?|млрд\.?|тысяч\w*|миллион\w*|миллиард\w*|"
    r"thousand\w*|million\w*|billion\w*)?",
    re.IGNORECASE,
)
DIGIT_GROUP_SPACE_RE = re.compile(r"(?<=\d)[ \u00a0](?=\d{3}\b)")
COMMA_GROUP_RE = re.compile(r"(?<![\w.,])([1-9]\d{0,2}(?:,\d{3})+)(?![\d,])")
DECIMAL_SCALE_RE = re.compile(
    r"^\s*(?:%|[kmb]\b|тыс|млн|млрд|тысяч|миллион|миллиард|thousand|million|billion)",
    re.IGNORECASE,
)


def decimal_text(value: Decimal) -> str:
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_numeric_values(text: str) -> set[str]:
    compact = DIGIT_GROUP_SPACE_RE.sub("", text)
    # English grouped integers must not become decimals: 20,000 != 20.000.
    # Preserve decimal commas (0,125; 1,234%; 1,234 million), where a locale-free
    # conversion would be ambiguous. Dot-decimal grouped amounts remain valid.
    compact = COMMA_GROUP_RE.sub(
        lambda match: (
            match.group(0)
            if DECIMAL_SCALE_RE.match(compact[match.end() :])
            else match.group(0).replace(",", "")
        ),
        compact,
    )
    values: set[str] = set()
    for match in NUMBER_WITH_UNIT_RE.finditer(compact):
        raw_number = match.group(1).replace(",", ".")
        suffix = (match.group(2) or "").casefold()
        try:
            number = Decimal(raw_number)
        except InvalidOperation:  # pragma: no cover - regex already constrains input
            continue
        if suffix == "%":
            values.add(f"{decimal_text(number)}%")
            continue
        if suffix in {"k", "тыс", "тыс."} or suffix.startswith(("тысяч", "thousand")):
            number *= 1_000
        elif suffix in {"m", "млн", "млн."} or suffix.startswith(("миллион", "million")):
            number *= 1_000_000
        elif suffix in {"b", "млрд", "млрд."} or suffix.startswith(("миллиард", "billion")):
            number *= 1_000_000_000
        values.add(decimal_text(number))
    return values
