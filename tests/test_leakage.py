from nsss_vg.evaluation.leakage import explain_no_lookahead_rule


def test_no_lookahead_rule_text_exists():
    text = explain_no_lookahead_rule()
    assert "trailing" in text.lower()
    assert "future" in text.lower()
