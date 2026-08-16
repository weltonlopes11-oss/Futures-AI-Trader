from __future__ import annotations

from collections import Counter


class BacktestSimulator:
    """
    Simulador institucional.

    Nesta primeira versão não simula execução financeira.

    Apenas consolida as decisões produzidas pelo pipeline.
    """

    def simulate(
        self,
        trades: list,
    ) -> dict:

        total = len(trades)

        # ----------------------------------------
        # Decisões operacionais
        # ----------------------------------------

        longs = [
            t
            for t in trades
            if t["decision"].action == "LONG"
        ]

        shorts = [
            t
            for t in trades
            if t["decision"].action == "SHORT"
        ]

        no_trade = [
            t
            for t in trades
            if t["decision"].action == "NO_TRADE"
        ]

        approved_trades = len(longs) + len(shorts)

        rejected_trades = len(no_trade)

        # ----------------------------------------
        # Distribuições
        # ----------------------------------------

        qualities = Counter(
            t["signal"].quality
            for t in trades
        )

        regimes = Counter(
            t["context"].regime
            for t in trades
        )

        directions = Counter(
            t["context"].direction
            for t in trades
        )

        recommendations = Counter(
            t["signal"].recommendation
            for t in trades
        )

        # ----------------------------------------
        # Percentuais
        # ----------------------------------------

        approval_rate = (
            round(
                approved_trades * 100 / total,
                2,
            )
            if total
            else 0
        )

        rejection_rate = (
            round(
                rejected_trades * 100 / total,
                2,
            )
            if total
            else 0
        )

        return {

            "total_candles": total,

            "approved_trades": approved_trades,

            "rejected_trades": rejected_trades,

            "long_trades": len(longs),

            "short_trades": len(shorts),

            "no_trade": len(no_trade),

            "quality_distribution": dict(qualities),

            "market_regimes": dict(regimes),

            "market_directions": dict(directions),

            "recommendations": dict(recommendations),

            "approval_rate": approval_rate,

            "rejection_rate": rejection_rate,
        }