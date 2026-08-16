from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from filters.filter_result import FilterResult


class BaseFilter(ABC):
    """
    Classe base para todos os filtros institucionais.

    Todos os filtros do projeto devem herdar desta classe.

    Exemplo:

        class TimeframeFilter(BaseFilter):

            def evaluate(self, data):

                ...

                return FilterResult(...)
    """

    def __init__(self, config: Dict[str, Any] | None = None):

        self.config = config or {}

    @property
    def name(self) -> str:
        """
        Nome do filtro.
        """

        return self.__class__.__name__

    @abstractmethod
    def evaluate(self, data: Any) -> FilterResult:
        """
        Executa o filtro.

        Deve retornar obrigatoriamente um FilterResult.
        """
        raise NotImplementedError

    def approve(
        self,
        score: float,
        direction: str,
        grade: str,
        metadata: Dict[str, Any] | None = None,
    ) -> FilterResult:
        """
        Cria um FilterResult aprovado.
        """

        return FilterResult(
            approved=True,
            filter_name=self.name,
            direction=direction,
            score=score,
            grade=grade,
            metadata=metadata or {},
        )

    def reject(
        self,
        reason: str,
        metadata: Dict[str, Any] | None = None,
    ) -> FilterResult:
        """
        Cria um FilterResult reprovado.
        """

        return FilterResult(
            approved=False,
            filter_name=self.name,
            direction="NONE",
            score=0.0,
            grade="C",
            reason=reason,
            metadata=metadata or {},
        )