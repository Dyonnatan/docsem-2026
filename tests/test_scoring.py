from dfm.scoring import mcnemar, norm_answer, wilson


def test_normalisation():
    assert norm_answer("Final answer: 1,200.0") == "1200"
    assert norm_answer(" 2.50 ") == "2.5"


def test_exact_mcnemar_matches_paper_value():
    a = {i: i < 73 for i in range(89)}
    b = {i: i >= 73 for i in range(89)}
    n, x, y, p = mcnemar(a, b)
    assert (x, y) == (73, 16) and abs(p - 7.24e-10) < 1e-11


def test_wilson():
    lo, hi = wilson(139, 300)
    assert round(lo, 1) == 40.8 and round(hi, 1) == 52.0
