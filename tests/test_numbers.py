from app.pipeline.numbers import extract_numeric_values


def test_numeric_units_are_canonicalized_across_source_formats() -> None:
    assert "500000" in extract_numeric_values("500k tokens")
    assert "500000" in extract_numeric_values("500 000 токенов")
    assert "500000" in extract_numeric_values("500 тысяч токенов")
    assert "500000000" in extract_numeric_values("0.5B input tokens")
    assert "500000000" in extract_numeric_values("0,5 миллиарда токенов")
    assert "1000000" in extract_numeric_values("$2 per 1M input tokens")
    assert "60%" in extract_numeric_values("60% cheaper")
    assert "500000000000" in extract_numeric_values("$500 billion financing platform")
    assert "6500000" in extract_numeric_values("6.5 million tokens")
    assert "500000000000" in extract_numeric_values("500 млрд долларов")
    assert "10000" in extract_numeric_values("10 тыс. сотрудников")
