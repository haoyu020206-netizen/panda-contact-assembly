from scripts.evaluate_stall_trigger_holdout import PROTOCOL, apply_decision, load_json, summarize


def test_holdout_decision_requires_a_primary_gain() -> None:
    protocol = load_json(PROTOCOL)
    conditions = {
        condition: {"net_rescues": 0, "new_safety_terminations": 0}
        for condition in protocol["condition_order"]
    }
    assert apply_decision(conditions, protocol)["policy"] == "search_off"
    conditions[protocol["primary_condition"]]["net_rescues"] = 1
    assert apply_decision(conditions, protocol)["policy"] == "stall_trigger"
    conditions["oracle"]["net_rescues"] = -1
    assert apply_decision(conditions, protocol)["policy"] == "search_off"


def test_paired_safety_detects_new_abort_even_if_totals_match() -> None:
    rows = [
        {
            "search_off_success": False,
            "stall_trigger_success": False,
            "search_off_safety_violation": a,
            "stall_trigger_safety_violation": b,
            "search_off_search_triggered": False,
            "stall_trigger_search_triggered": True,
            "search_off_peak_force_n": 200.0 if a else 50.0,
            "stall_trigger_peak_force_n": 200.0 if b else 50.0,
        }
        for a, b in ((False, True), (True, False))
    ]
    summary = summarize(rows)
    assert summary["policies"]["search_off"]["safety_terminations"] == 1
    assert summary["policies"]["stall_trigger"]["safety_terminations"] == 1
    assert summary["new_safety_terminations"] == 1
