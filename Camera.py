import numpy as np
import cv2
import cv2.aruco as aruco


class Camera:

    def __init__(self, camera_mat, dist_coef, marker_length):
        self.camera_mat = camera_mat
        self.dist_coef = dist_coef
        self.marker_length = marker_length
        self.rvec = []
        self.tvec = []

    # カメラ外部パラメータR・tセット
    def set_Rt(self, rvec1, tvec1):
        self.rvec = rvec1
        self.tvec = tvec1

    # ディジタル座標2D(x,y)を入力して、カメラ座標ベクトル(カメラ位置からの向き)を取得
    def get_vec_3D(self, pt2d, kind):
        # kind:0 = 収差補正しない、kind:1 = 補正する

        # 収差補正前 → 後へ変換
        if kind == 1:
            pt2d = cv2.undistortPoints(pt2d, self.camera_mat, self.dist_coef, P=self.camera_mat)

        # カメラ座標へ変換
        # x = pt2d[0][0][0]
        # y = pt2d[0][0][1]
        x = pt2d[0]
        y = pt2d[1]
        cx = self.camera_mat[0, 2]
        cy = self.camera_mat[1, 2]
        fx = self.camera_mat[0, 0]
        fy = self.camera_mat[1, 1]
        pt3d = np.zeros([3])
        pt3d[0] = (x - cx) / fx
        pt3d[1] = (y - cy) / fy
        pt3d[2] = 1.0

        R = cv2.Rodrigues(self.rvec)[0]  # 回転ベクトル(1×3) -> 回転行列(3×3)
        R_T = R.T  # 転置行列
        v = np.dot(R_T, pt3d)
        if v[2] >= 0.0:
            return False, v
        return True, v

    @staticmethod
    # カメラ位置camtと向きvecで定義される直線とXY平面の交点を求める
    def get_crosspt_xy(camt, vec):

        xo = camt[0][0]
        yo = camt[0][1]
        zo = camt[0][2]
        a = vec[0]
        b = vec[1]
        c = vec[2]

        t = - zo / c
        x = xo + a * t
        y = yo + b * t

        pt3d = np.zeros([3])
        pt3d[0] = x
        pt3d[1] = y
        pt3d[2] = 0.0
        return pt3d

    # カメラ外部パラメータR・t取得
    def get_camera_Rt(self, corner):
        rvec1, tvec1, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corner, self.marker_length,
                                                                     self.camera_mat, self.dist_coef)

        R = cv2.Rodrigues(rvec1)[0]  # 回転ベクトル(1×3) -> 回転行列(3×3)
        return rvec1, tvec1

    # カメラ位置(x,y,z)取得
    def get_camera_pos(self, corners):
        rvec1, tvec1 = self.get_camera_Rt(corners)

        XYZ = []
        # RPY = []
        # V_x = []
        # V_y = []
        # V_z = []
        # imgpts_o_dst = []

        # カメラ位置算出（参考：https://qiita.com/namahoge/items/69b4e2c66f54+-+46dc8798）

        R = cv2.Rodrigues(rvec1)[0]  # 回転ベクトル(1×3) -> 回転行列(3×3)
        R_T = R.T  # 転置行列
        T = tvec1[0].T  # (1×3) →　(3×1)

        xyz = np.dot(R_T, - T).squeeze()  # R_T,-T の積
        XYZ.append(xyz)  # XYZ[]にxyzを付加

        return XYZ

    '''
    def main(self):

    if __name__ == '__main__':
        main()
    '''
