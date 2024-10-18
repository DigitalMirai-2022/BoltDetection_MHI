from dataclasses import dataclass
from typing import List


@dataclass
class MarkerSize:
    length: float


@dataclass
class Axis:
    direction: str
    marker_no: int


@dataclass
class YRange:
    lower_y: float


@dataclass
class CodeSpecification:
    flg: str
    code: List[int]


@dataclass
class BoltStatus:
    loading_interval: float
    status_change_count: int


@dataclass
class Config_Data:
    marker_size: MarkerSize
    axis: Axis
    y_range: YRange
    code_specification: CodeSpecification
    bolt_status: BoltStatus
