from dataclasses import dataclass


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
    code: int


@dataclass
class BoltStatus:
    loading_interval: float
    status_change_count: int


@dataclass
class BoltDistance:
    distance: float


@dataclass
class Device:
    use_device: str


@dataclass
class Config:
    marker_size: MarkerSize
    axis: Axis
    y_range: YRange
    code_specification: CodeSpecification
    bolt_status: BoltStatus
    bolt_distance: BoltDistance
    device: Device

    # JSONファイルを読み込み、設定をDataclassにロード
    def read_json(self):
        try:
            # JSONファイル読み込み
            with open(self.config_path, "r") as f:
                setting_data = json.load(f)

            # 設定データをDataclassにロード
            self.setting = Config(
                marker_size=MarkerSize(length=setting_data["marker size"]["length"]),
                axis=Axis(
                    direction=setting_data["axis"]["direction"],
                    marker_no=setting_data["axis"]["marker no"],
                ),
                y_range=YRange(lower_y=setting_data["y range"]["lower y"]),
                code_specification=CodeSpecification(
                    flg=setting_data["code specification"]["flg"],
                    code=setting_data["code specification"]["code"],
                ),
                bolt_status=BoltStatus(
                    loading_interval=setting_data["bolt status"]["loading interval"],
                    status_change_count=setting_data["bolt status"][
                        "status change count"
                    ],
                ),
                bolt_distance=BoltDistance(
                    distance=setting_data["bolt distance"]["distance"]
                ),
                device=Device(use_device=setting_data["device"]["use device"]),
            )

            # ArUCo Marker サイズ設定
            self.marker_length = self.setting.marker_size.length
            self.m_Marker.marker_length = self.marker_length
            self.Cam.marker_length = self.marker_length
            self.log.info(
                __file__, f"Marker length: {self.marker_length * 1000:.0f} mm"
            )

            # 座標系設定用マーカー
            self.m_Marker.axis_dirct = (
                0 if self.setting.axis.direction.lower() == "x" else 1
            )
            self.m_Marker.axis_no = self.setting.axis.marker_no

            # Y 下方向許容幅設定
            self.m_Marker.lower_y = self.setting.y_range.lower_y
            self.log.info(
                __file__,
                f"Initial Y lower range: {self.m_Marker.lower_y * 1000:.0f} mm",
            )

            # ArUCo No 指定
            if self.setting.code_specification.flg.lower() == "yes":
                self.m_Marker.f_marker_no_fixed = True
                self.m_Marker.set_ini_code_no(self.setting.code_specification.code)
            else:
                self.m_Marker.f_marker_no_fixed = False

            # ボルト状況取得関連
            self.Time_interval = self.setting.bolt_status.loading_interval
            self.status_change_count = self.setting.bolt_status.status_change_count

            # 各ボルト間の距離設定
            self.bolt_distance = self.setting.bolt_distance.distance

            # 利用デバイス
            self.device = self.setting.device.use_device

        except FileNotFoundError:
            self.log.exception(__file__, f"{self.config_path} not found")
            raise ValueError(f"{self.config_path} not found")
        except KeyError as e:
            self.log.exception(__file__, f"Key missing in JSON: {e}")
            raise ValueError(f"Key missing in JSON: {e}")
