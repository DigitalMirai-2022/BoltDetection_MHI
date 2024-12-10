import numpy as np


class DataArUco:

    interval = 1e+6       # 撮影回数をカウントするインターバル（lim_Interval回数あけて、cntを+1する）

    def __init__(self, no,  corner):
        self.no = no
        self.corner = corner    # 4角の3D点座標

        self.cnt = 0            # 撮影回数
        self.status = -1        # マーカーの状態。-1=初期値、0=基底マーカーとのRt取得、1=一般マーカーとのRt取得
        self.virtual_no = -1    # status=1の場合、一般マーカーno
        self.R = np.zeros((3, 3))   # 全体座標系へ戻すための回転行列[3x3]
        self.t = np.zeros((3, 1))   # 　　〃　　　　　　　　並進行列[3x1]

