from typing import TypedDict, List


class ConfusionMatrixInfo(TypedDict):
    columns: List[str]
    caption: str
    label: str
    row_labels: List[str]
    data: List[List[float]]


class AttackVector(TypedDict):
    NETWORK: int
    ADJACENT_NETWORK: int
    LOCAL: int
    PHYSICAL: int


class AttackComplexity(TypedDict):
    LOW: int
    HIGH: int


class PrivilegesRequired(TypedDict):
    NONE: int
    LOW: int
    HIGH: int


class UserInteraction(TypedDict):
    NONE: int
    REQUIRED: int


class Scope(TypedDict):
    UNCHANGED: int
    CHANGED: int


class Impact(TypedDict):
    NONE: int
    LOW: int
    HIGH: int


class Metrics(TypedDict):
    attackVector: AttackVector
    attackComplexity: AttackComplexity
    privilegesRequired: PrivilegesRequired
    userInteraction: UserInteraction
    scope: Scope
    confidentialityImpact: Impact
    integrityImpact: Impact
    availabilityImpact: Impact
