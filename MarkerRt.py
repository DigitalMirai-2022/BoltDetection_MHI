import numpy as np


class CalcRt:

    def __init__(self, corner0, corner1):
        self.corner0 = corner0  # 基準マーカーコーナー座標値
        self.corner1 = corner1  # コーナー座標値
        self.R = []             # 回転行列
        self.t = []             # 並進行列

    # 2つのマーカー座標値より回転角・平行移動量を算出
    # https://imagingsolution.net/imaging/affine-transformation/
    def get_Rt(self):

        # corner1からcorner0への回転量を求める
        # corner1の角点0→1ベクトル
        dx = self.corner1[1][0] - self.corner1[0][0]
        dy = self.corner1[1][1] - self.corner1[0][1]
        v1 = np.array([dx, dy])
        # corner0の角点0→1ベクトル
        dx = self.corner0[1][0] - self.corner0[0][0]
        dy = self.corner0[1][1] - self.corner0[0][1]
        v0 = np.array([dx, dy])
        angle = self.vector_to_vector_rotation_angle(v0, v1)
        r1 = np.identity(3)     # 3x3単位行列
        r1[0][0] = np.cos(angle)
        r1[0][1] = - np.sin(angle)
        r1[1][0] = np.sin(angle)
        r1[1][1] = np.cos(angle)
        self.R = r1

        # corner0→corner1へ平行移動
        # t1 = np.identity(3)     # 3x3単位行列
        t1 = np.zeros((3, 1))
        dx = 0.0
        dy = 0.0
        for i in range(4):
            dx += self.corner1[i][0]
            dy += self.corner1[i][1]
        dx /= 4.0
        dy /= 4.0
        # t1[0][2] = dx
        # t1[1][2] = dy
        t1[0][0] = dx
        t1[1][0] = dy
        self.t = t1

    @staticmethod
    # 任意のベクトルからの回転角(vector_1を基準とした回転角)
    # https://qiita.com/yusuke_s_yusuke/items/95e52c5e5932acd7e056
    def vector_to_vector_rotation_angle(vector_1, vector_2):
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        cos = np.dot(unit_vector_1, unit_vector_2)
        rad_angle = np.arccos(cos)
        # angle = rad_angle / np.pi * 180
        cross = np.cross(vector_1, vector_2)
        if cross < 0:
            rad_angle *= -1
        return rad_angle

    def rotate(self, r1):

        # R1[3][3]で回転させる
        r2 = np.dot(r1, self.R)
        t2 = np.dot(r1, self.t)

        # 反映
        self.R = r2
        self.t = t2

    @staticmethod
    def rotate_corner(cor1, r1):

        for i in range(4):
            cor1[i] = np.dot(r1, cor1[i])
