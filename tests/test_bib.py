from backend.pipeline.bib import pick_bib


def _q(h):
    return [[0, 0], [50, 0], [50, h], [0, h]]


def test_pick_bib_prefers_tallest_digits():
    assert pick_bib([(_q(20), "1234", 0.9), (_q(60), "56", 0.9)]) == "56"


def test_pick_bib_strips_non_digits_and_filters():
    assert pick_bib([(_q(30), "No.123", 0.9)]) == "123"
    assert pick_bib([(_q(30), "ABC", 0.9)]) == ""
    assert pick_bib([(_q(30), "1234567", 0.9)]) == ""      # quá dài -> không phải BIB
    assert pick_bib([(_q(30), "99", 0.2)]) == ""           # tin cậy thấp
    assert pick_bib([]) == ""


def test_pick_bib_keeps_letter_prefix():
    assert pick_bib([(_q(30), "m5520", 0.9)]) == "M5520"
    assert pick_bib([(_q(30), "M 5520", 0.9)]) == "M5520"
