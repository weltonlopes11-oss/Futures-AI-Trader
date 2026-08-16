from __future__ import annotations

from telemetry.telemetry_models import TelemetryRecord


class TelemetryStorage:
    """
    Buffer em memória.

    Futuramente poderá salvar em:

    - SQLite
    - PostgreSQL
    - Parquet
    - CSV
    """

    def __init__(self):

        self._records: list[TelemetryRecord] = []

    def add(
        self,
        record: TelemetryRecord,
    ):

        self._records.append(record)

    def clear(self):

        self._records.clear()

    @property
    def records(self):

        return self._records

    def __len__(self):

        return len(self._records)