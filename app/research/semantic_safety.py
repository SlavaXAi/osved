"""Narrow source-conditioned guards against high-impact translation inversions."""

import re

CYBER_CAPABILITY_EN = re.compile(r"\b(?:cybersecurity|cyber)\s+capabilit\w*\b", re.I)
CYBER_CAPABILITY_RU = re.compile(
    r"\b(?:возможност\w*|способност\w*).{0,45}\bкибер\w*", re.I
)
CRITICAL = re.compile(r"\b(?:critical|критическ\w*)\b", re.I)
AMBIGUOUS_SECURITY_LEVEL = re.compile(
    r"\b(?:критическ\w*|critical)\s+(?:(?:уров\w*|порог\w*)\s+)?(?:в\s+)?"
    r"(?:кибербезопасност\w*|безопасност\w*|защищ[её]нност\w*|защит\w*)\b", re.I
)
RISK_ATTRIBUTION = re.compile(
    r"\b(?:по\s+(?:данным|оценке|классификации)|согласно|"
    r"классификац\w*|фреймворк\w*|framework|preparedness|"
    r"оцени\w*|отнес\w*|отн[её]с\w*)\b", re.I
)
RELATIVE_DAY = re.compile(
    r"\b(?:сегодня\w*|вчера\w*|завтра\w*|позавчера|послезавтра|"
    r"today|yesterday|tomorrow)\b", re.I,
)


def relative_publication_time_errors(text: str) -> list[str]:
    """Generated prose must be stable after the source's publication day.

    Literal evidence excerpts are separate fields and are not passed here. No date
    is guessed: the writer may use a cited absolute date or omit the relative day.
    """
    if RELATIVE_DAY.search(text):
        return ["relative publication day is ambiguous; use a cited absolute date or omit it"]
    return []


def cyber_capability_translation_errors(text: str, cited_evidence: str) -> list[str]:
    if not CRITICAL.search(cited_evidence) or not (
        CYBER_CAPABILITY_EN.search(cited_evidence)
        or CYBER_CAPABILITY_RU.search(cited_evidence)
    ):
        return []
    errors = []
    if AMBIGUOUS_SECURITY_LEVEL.search(text):
        errors.append(
            "cybersecurity capability/risk is not the model's security strength; "
            "preserve capabilities in cyberoperations"
        )
    if (
        CRITICAL.search(text)
        and CYBER_CAPABILITY_RU.search(text)
        and not RISK_ATTRIBUTION.search(text)
    ):
        errors.append("critical cyber capability classification requires explicit attribution")
    return errors
