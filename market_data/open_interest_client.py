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
        """Baixa uma janela temporal completa usando paginação progressiva."""

        if start_time > end_time:
            raise ValueError("start_time não pode ser maior que end_time.")

        records: list[dict[str, Any]] = []
        cursor = int(start_time)

        while cursor <= end_time:
            batch = self.get_history(
                symbol=symbol,
                period=period,
                start_time=cursor,
                end_time=end_time,
                limit=self.MAX_LIMIT,
            )

            if not batch:
                break

            records.extend(batch)

            timestamps = [
                int(item["timestamp"])
                for item in batch
                if "timestamp" in item
            ]

            if not timestamps:
                raise ValueError(
                    "Open Interest sem campo timestamp."
                )

            newest = max(timestamps)

            if newest < cursor:
                break

            cursor = newest + 1

            if len(batch) < self.MAX_LIMIT:
                break

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
