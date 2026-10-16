import copy

from credit_risk.config import check_no_forbidden, load_config


def test_default_config_passes_guard():
    check_no_forbidden(load_config())


def test_guard_blocks_leaky_feature():
    cfg = copy.deepcopy(load_config())
    cfg["features"]["numeric"].append("MonthsInRepayment")
    try:
        check_no_forbidden(cfg)
    except ValueError as e:
        assert "MonthsInRepayment" in str(e)
        return
    raise AssertionError("guard did not block a forbidden feature")


def test_band_percentages_cover_population():
    cfg = load_config()
    ups = [b["upper_pct"] for b in cfg["risk_bands"]]
    assert ups == sorted(ups) and ups[-1] == 100
