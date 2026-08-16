from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class FilterResult:
    """
    Resultado padronizado retornado por qualquer filtro institucional.
    Todos os filtros do projeto devem retornar uma instância desta classe.
    """

    # Indica se o filtro aprovou ou não o cenário
    approved: bool

    # Nome do filtro
    filter_name: str

    # Direção encontrada
    # Valores esperados:
    # LONG | SHORT | NONE
    direction: str = "NONE"

    # Score individual do filtro (0-100)
    score: float = 0.0

    # Classificação institucional
    # A+, A, B, C
    grade: str = "C"

    # Motivo da reprovação
    reason: str = ""

    # Informações extras para auditoria
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o resultado para dicionário.
        """

        return {
            "approved": self.approved,
            "filter_name": self.filter_name,
            "direction": self.direction,
            "score": round(self.score, 2),
            "grade": self.grade,
            "reason": self.reason,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"FilterResult("
            f"approved={self.approved}, "
            f"filter='{self.filter_name}', "
            f"direction='{self.direction}', "
            f"score={self.score:.2f}, "
            f"grade='{self.grade}')"
        )