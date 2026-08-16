from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from backtest.engine import BacktestEngine


def _engine_with_history(history):
    engine = object.__new__(BacktestEngine)
    engine.loader = Mock()
    engine.loader.load.return_value = history
    engine.feature_engine = Mock()
    return engine


def test_backtest_stops_before_feature_engine_when_history_is_none():
    engine = _engine_with_history(None)

    with pytest.raises(RuntimeError, match="Histórico vazio"):
        engine.run(limit=1000, warmup=200)

    engine.feature_engine.enrich.assert_not_called()


def test_backtest_stops_before_feature_engine_when_history_is_empty():
    engine = _engine_with_history(pd.DataFrame())

    with pytest.raises(RuntimeError, match="Histórico vazio"):
        engine.run(limit=1000, warmup=200)

    engine.feature_engine.enrich.assert_not_called()


def test_backtest_rejects_incompatible_history_type():
    engine = _engine_with_history([])

    with pytest.raises(TypeError, match="Histórico inválido"):
        engine.run(limit=1000, warmup=200)

    engine.feature_engine.enrich.assert_not_called()
