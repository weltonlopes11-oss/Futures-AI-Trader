from __future__ import annotations

from types import SimpleNamespace

from telemetry.open_interest_statistics import OpenInterestStatistics


def _record(*, price, oi, decision, regime="STRONG_BULL"):
    return SimpleNamespace(
        timestamp="2026-08-16",
        close_price=price,
        decision=SimpleNamespace(action=decision),
        market=SimpleNamespace(
            regime=regime,
            direction="LONG",
            metadata={
                "open_interest": oi,
                "open_interest_change_pct": None,
            },
        ),
    )


def test_counts_real_oi_observations_instead_of_forward_filled_candles():
    records = [
        _record(price=100, oi=1000, decision="LONG"),
        _record(price=101, oi=1000, decision="LONG"),
        _record(price=102, oi=1000, decision="LONG"),
        _record(price=103, oi=1100, decision="LONG"),
        _record(price=104, oi=1100, decision="LONG"),
        _record(price=102, oi=990, decision="SHORT", regime="STRONG_BEAR"),
    ]

    stats = OpenInterestStatistics(records)

    assert stats.total_observations == 3
    assert stats.oi_direction_distribution() == {
        "UP": 1,
        "DOWN": 1,
    }


def test_crosses_open_interest_with_decision_and_regime():
    records = [
        _record(price=100, oi=1000, decision="LONG"),
        _record(price=105, oi=1100, decision="LONG"),
        _record(price=100, oi=990, decision="SHORT", regime="STRONG_BEAR"),
    ]

    stats = OpenInterestStatistics(records)

    assert stats.by_decision()["LONG"]["UP"] == 1
    assert stats.by_decision()["SHORT"]["DOWN"] == 1
    assert stats.by_regime()["STRONG_BULL"]["UP"] == 1
    assert stats.by_regime()["STRONG_BEAR"]["DOWN"] == 1


def test_price_and_oi_quadrants_use_changes_between_real_oi_observations():
    records = [
        _record(price=100, oi=1000, decision="LONG"),
        _record(price=110, oi=1100, decision="LONG"),
        _record(price=105, oi=990, decision="SHORT", regime="STRONG_BEAR"),
    ]

    stats = OpenInterestStatistics(records)
    combinations = stats.price_oi_combinations()

    assert combinations["PRICE_UP__OI_UP"] == 1
    assert combinations["PRICE_DOWN__OI_DOWN"] == 1


def test_change_statistics_include_percentiles_without_thresholds():
    records = [
        _record(price=100, oi=1000, decision="LONG"),
        _record(price=101, oi=1010, decision="LONG"),
        _record(price=102, oi=1030.2, decision="LONG"),
        _record(price=101, oi=1019.898, decision="NO_TRADE"),
    ]

    stats = OpenInterestStatistics(records)
    distribution = stats.change_statistics()["oi"]

    assert distribution["count"] == 3
    assert distribution["min"] < 0
    assert distribution["max"] > 0
    assert distribution["p25"] <= distribution["median"] <= distribution["p75"]
