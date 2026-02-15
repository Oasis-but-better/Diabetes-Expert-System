import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

FEATURE_WEIGHTS = {
    "age": 0.15,
    "bmi": 0.15,
    "fpg": 0.20,
    "a1c": 0.15,
    "random_glucose": 0.15,
    "ketones": 0.10,
    "has_classic_symptoms": 0.10,
}

NUMERIC_RANGES = {
    "age": (0, 100),
    "bmi": (10, 60),
    "fpg": (50, 500),
    "a1c": (3.0, 15.0),
    "random_glucose": (50, 600),
}

CLASSIC_SYMPTOMS = {"polyuria", "polydipsia", "weight-loss"}


@dataclass
class CaseFeatures:
    age: Optional[float] = None
    bmi: Optional[float] = None
    fpg: Optional[float] = None
    a1c: Optional[float] = None
    random_glucose: Optional[float] = None
    fpg_repeat: Optional[float] = None
    ketones: Optional[str] = None
    symptoms: list = field(default_factory=list)

    @property
    def has_classic_symptoms(self) -> bool:
        return bool(CLASSIC_SYMPTOMS.intersection(set(self.symptoms)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CaseFeatures":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CaseSolution:
    diagnosis: str
    status: str
    recommendations: list = field(default_factory=list)
    classification: Optional[str] = None   
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CaseSolution":
        return cls(**data)


@dataclass
class Case:
    id: str
    features: CaseFeatures
    solution: CaseSolution
    outcome: str
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "features": self.features.to_dict(),
            "solution": self.solution.to_dict(),
            "outcome": self.outcome,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Case":
        return cls(
            id=data["id"],
            features=CaseFeatures.from_dict(data["features"]),
            solution=CaseSolution.from_dict(data["solution"]),
            outcome=data.get("outcome", "pending"),
            notes=data.get("notes", ""),
        )


class CaseLibrary:
    """
    Loads cases from JSON, exposes retrieval interface, and handles persistence
    when new cases are retained after successful diagnosis cycles.
    """

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.cases: list[Case] = []
        self._load()

    def _load(self):
        if not self.filepath.exists():
            return
        with open(self.filepath) as f:
            raw = json.load(f)
        self.cases = [Case.from_dict(item) for item in raw]

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump([c.to_dict() for c in self.cases], f, indent=2)

    def add_case(self, case: Case):
        if any(c.id == case.id for c in self.cases):
            return
        self.cases.append(case)
        self.save()

    def all_cases(self) -> list[Case]:
        return self.cases

    def __len__(self) -> int:
        return len(self.cases)
