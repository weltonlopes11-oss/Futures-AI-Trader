from __future__ import annotations

import time
from typing import Any

import requests

from logs.logger import setup_logger


logger = setup_logger()


class OpenInterestClient:
    """Cliente público para histórico de Open Interest da Binance Futures."""

    BASE_URL = "https://fapi.binance.com"
    ENDPOINT = "/futures/data/openInterestHist"
    MAX_LIMIT = 500

    VALID_PERIODS = {
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "12h",
        "1d",
    }

    PERIOD_MS = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    def get_history(
        self,
        *,
        symbol: str,
        period: str = "5m",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if period not in self.VALID_PERIODS:
            raise ValueError(f"Período de Open Interest inválido: {period}")

        params: dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "limit": min(max(int(limit), 1), self.MAX_LIMIT),
        }

        if start_time is not None:
            params["startTime"] = int(start_time)

        if end_time is not None:
            params["endTime"] = int(end_time)

        try:
            response = requests.get(
                f"{self.BASE_URL}{self.ENDPOINT}",
                params=params,
                timeout=20,
            )
            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise ValueError(
                    "Resposta inválida da Binance para Open Interest."
                )

            logger.info(
                f"Open Interest recebido: {len(data)} registros"
            )

            return data

        except Exception as error:
            logger.error(
                f"Erro Binance Open Interest: {error}"
            )
            raise

    def get_range(
        self,
        *,
        symbol: str,
        start_time: int,
        end_time: int,
        period: str = "5m",
        pause_seconds: float = 0.15,
    ) -> list[dict[str, Any]]:
        """
        Baixa uma janela temporal completa em blocos limitados ao máximo
        de observações aceito pelo endpoint.

        O recorte por tempo evita depender do comportamento de paginação
        implícita do endpoint quando uma janela inteira possui mais de
        MAX_LIMIT observações.
        """

        if start_time > end_time:
            raise ValueError("start_time não pode ser maior que end_time.")

        if period not in self.VALID_PERIODS:
            raise ValueError(f"Período de Open Interest inválido: {period}")

        period_ms = self.PERIOD_MS[period]

        # Uma janela com no máximo 500 observações possui 499 intervalos
        # entre o primeiro e o último timestamp.
        max_window_span = period_ms * (self.MAX_LIMIT - 1)

        records: list[dict[str, Any]] = []
        window_start = int(start_time)

        while window_start <= end_time:
            window_end = min(
                window_start + max_window_span,
                int(end_time),
            )

            batch = self.get_history(
                symbol=symbol,
                period=period,
                start_time=window_start,
                end_time=window_end,
                limit=self.MAX_LIMIT,
            )

            if batch:
                records.extend(batch)

            if window_end >= end_time:
                break

            # Avança para além da janela já consultada. O +1 evita
            # repetir a observação exatamente na borda do bloco.
            window_start = window_end + 1

            time.sleep(pause_seconds)

        unique: dict[int, dict[str, Any]] = {}

        for item in records:
            if "timestamp" not in item:
                continue
            unique[int(item["timestamp"])] = item

        return [
            unique[key]
            for key in sorted(unique)
            if start_time <= key <= end_time
        ]
