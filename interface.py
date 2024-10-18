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


# # JSONデータロード
# with open("config.json", "r") as f:
#     data_dict = json.load(f)

# # データクラスインスタンス生成
# marker_size = MarkerSize(length=data_dict["marker size"]["length"])
# axis = Axis(
#     direction=data_dict["axis"]["direction"], marker_no=data_dict["axis"]["marker no"]
# )
# y_range = YRange(lower_y=data_dict["y range"]["Lower y"])
# code_specification = CodeSpecification(
#     flg=data_dict["code specification"]["flg"],
#     code=data_dict["code specification"]["code"],
# )
# bolt_status = BoltStatus(
#     loading_interval=data_dict["bolt status"]["loading interval"],
#     status_change_count=data_dict["bolt status"]["status change count"],
# )

# # コンフィグデータ
# config_data = Config_Data(
#     marker_size=marker_size,
#     axis=axis,
#     y_range=y_range,
#     code_specification=code_specification,
#     bolt_status=bolt_status,
# )
