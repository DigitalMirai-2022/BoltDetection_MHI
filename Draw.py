import numpy as np
import cv2


class Draw:

    def __init__(self, rvec, tvec, camera_mat, dist_coef, marker_length):
        self.rvec = rvec
        self.tvec = tvec
        self.camera_mat = camera_mat
        self.dist_coef = dist_coef
        self.marker_length = marker_length
        self.__image_height = 0
        self.__image_width = 0
        self.__r_base = np.identity(3)
        self.__r_base_inv = np.identity(3)
        # 一時的なカメラによる撮影範囲
        self.shooting_rect = np.zeros((4, 3))

    @property
    def image_height(self):
        return self.__image_height

    @image_height.setter
    def image_height(self, height):
        self.__image_height = height

    @property
    def image_width(self):
        return self.__image_width

    @image_width.setter
    def image_width(self, width):
        self.__image_width = width

    @property
    def r_base(self):
        return self.__r_base

    @r_base.setter
    def r_base(self, r1):
        self.__r_base = r1
        self.__r_base_inv = np.linalg.inv(r1)

    @staticmethod
    def angle_calc(v1, v2):
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0.0
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if cos_theta < 0.0:
            cos_theta *= -1.0
        if np.abs(cos_theta - 1.0) <= 1.0e-6:
            return 0.0
        # print(cos_theta)
        theta = np.arccos(cos_theta) * 180 / np.pi
        if theta > 90.0:
            theta = 180.0 - theta
        return theta

    # 画像内2D座標を指定してライン描画
    @staticmethod
    def draw_line(frame, p0, p1, th1, color1):

        # 始点
        ip0 = np.zeros(2, dtype=np.int16)
        ip0[0] = int(p0[0])
        ip0[1] = int(p0[1])

        # 終点
        ip1 = np.zeros(2, dtype=np.int16)
        ip1[0] = int(p1[0])
        ip1[1] = int(p1[1])

        cv2.line(frame, ip0, ip1, color=color1, thickness=th1, lineType=cv2.LINE_AA)

    # 画像ｻｲｽﾞ設定
    def set_image_size(self, height1, width1):

        self.image_height = height1
        self.image_width = width1

    # 現在の撮影範囲。全体座標系でセットする。
    def set_shooting_range(self, rect):
        # rect[0][0-3]   : [左下][x,y,z]
        # rect[1]   : 右下
        # rect[2]   : 右上
        # rect[3]   : 左上

        self.shooting_rect = rect

        # print('shooting rect 0 = ({:.3f}, {:.3f}'.format(rect[0][0], rect[0][1]))
        # print('shooting rect 2 = ({:.3f}, {:.3f}'.format(rect[2][0], rect[2][1]))

    # 原点の描画
    def draw_origin(self, frame, kind):
        # kind:0 = 補正前、kind:1 = 補正後

        # 原点を画像に投影
        origin = np.float32([0, 0, 0])
        img_o, jac_o = cv2.projectPoints(
            origin, self.rvec, self.tvec, self.camera_mat, self.dist_coef
        )

        # 収差補正前 → 後へ変換
        if kind == 1:
            img_o = cv2.undistortPoints(
                img_o, self.camera_mat, self.dist_coef, P=self.camera_mat
            )

        # 原点描画
        img_o = img_o.astype(np.int64)
        img_o = np.squeeze(img_o)
        cv2.circle(frame, (img_o[0], img_o[1]), 2, (0, 0, 255), -1)  # 中心点描画

    # 座標軸の描画
    def draw_axis(self, frame, kind):
        # kind:0 = 補正前、kind:1 = 補正後

        # 座標軸の座標セット
        d = self.marker_length / 2
        axis = np.float32([[0, 0, 0], [d, 0, 0], [0, d, 0], [0, 0, d]])
        # 基底マーカーの回転分変換
        R_I = np.linalg.inv(self.r_base)
        for i in range(4):
            axis[i] = np.dot(R_I, axis[i])
        ax_pts, jac = cv2.projectPoints(
            axis, self.rvec, self.tvec, self.camera_mat, self.dist_coef
        )

        # 収差補正後の場合
        if kind == 1:
            ax_pts = cv2.undistortPoints(
                ax_pts, self.camera_mat, self.dist_coef, P=self.camera_mat
            )

        # 軸の座標値処理
        output = []
        for lp in ax_pts:
            lp_int = lp.astype(np.int64)
            output.append(tuple(lp_int.ravel()))

        # 座標軸描画
        if not self.is_within_image(output[0]):
            return
        if self.is_within_image(output[1]):
            cv2.line(frame, output[0], output[1], (255, 0, 0), 2, lineType=cv2.LINE_AA)
        if self.is_within_image(output[2]):
            cv2.line(frame, output[0], output[2], (0, 0, 255), 2, lineType=cv2.LINE_AA)
        # if self.is_within_image(output[3]):
        #     cv2.line(frame, output[0], output[3], (0, 255, 0), 2, lineType=cv2.LINE_AA)
        #     # print("o = ({:d}, {:d}), z = ({:d}, {:d})".format(output[0][0], output[0][1],output[3][0],output[3][1],))

    @staticmethod
    # ArUcoマーカーの四辺を描画
    def draw_marker_corner(frame, corner, color1):
        points = corner[0].astype(np.int32)
        # cv2.putText(frame, str("A"), tuple(points[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # cv2.putText(frame, str("B"), tuple(points[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # cv2.putText(frame, str("C"), tuple(points[2]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        # cv2.putText(frame, str("D"), tuple(points[3]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        cv2.line(
            frame,
            tuple(points[0]),
            tuple(points[1]),
            color=color1,
            thickness=1,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.line(
            frame,
            tuple(points[1]),
            tuple(points[2]),
            color=color1,
            thickness=1,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.line(
            frame,
            tuple(points[2]),
            tuple(points[3]),
            color=color1,
            thickness=1,
            lineType=cv2.LINE_AA,
            shift=0,
        )
        cv2.line(
            frame,
            tuple(points[3]),
            tuple(points[0]),
            color=color1,
            thickness=1,
            lineType=cv2.LINE_AA,
            shift=0,
        )

    @staticmethod
    # ボルト位置(x,y)をテキスト表示
    def draw_text_position(frame, crd_2d, crd_3d, color1):

        # 画像へ表示
        str1 = "{0:.3f}, {1:.3f}".format(crd_3d[0], crd_3d[1])  # 3点目
        pos1 = []
        pos1.append(int(crd_2d[0]))
        pos1.append(int(crd_2d[1]))
        cv2.putText(
            frame,
            str1,
            tuple(pos1),
            fontFace=cv2.FONT_ITALIC,
            fontScale=0.5,
            color=color1,
            thickness=1,
        )

    # 3D座標から画像に投影した2D座標取得
    def get_projected_2d(self, pt_3d0, r1, t1):
        # Input
        #   pt_3d[] : 3D座標値
        #   r1      : 現在のカメラR[3,3]
        #   t1      : 現在のカメラt[3]
        # Output
        #   bool
        #   pt_2d   : 2D投影座標

        pt_2d = np.zeros(2)

        pt_3d = np.copy(pt_3d0)

        # 撮影範囲内にあるかどうか
        if not self.check_pt_is_inside_rect(pt_3d):
            return False, pt_2d

        # 全体座標系を仮想マーカー座標系へ変換
        for j, val in enumerate(t1):
            pt_3d[j] += val
        pt_3d = np.dot(r1, pt_3d)
        # 投影座標へ変換
        pt_2d, jac_o = cv2.projectPoints(
            pt_3d, self.rvec, self.tvec, self.camera_mat, self.dist_coef
        )
        # 収差補正前 → 後へ変換
        pt_2d = cv2.undistortPoints(
            pt_2d, self.camera_mat, self.dist_coef, P=self.camera_mat
        )
        # 画像外にはみ出していないかチェック
        pt_2d = np.squeeze(pt_2d)
        if pt_2d[0] < 0.0 or pt_2d[0] > self.image_width:
            return False, pt_2d
        if pt_2d[1] < 0.0 or pt_2d[1] > self.image_height:
            return False, pt_2d

        pt_2d = np.squeeze(pt_2d)
        return True, pt_2d

    # 2D点へ点を描画
    def draw_point_by_2d(self, frame, x1, y1):
        # Input
        #   x1,y1 : 座標値

        # 画像内にあるかチェック
        if x1 < 0 or x1 > self.image_width:
            return False
        if y1 < 0 or y1 > self.image_height:
            return False

        op = np.zeros(2, dtype=np.int16)
        op[0] = int(x1)
        op[1] = int(y1)
        # cv2.circle(frame, center=op, radius=5, color=(0, 0, 255), thickness=-1)
        return True

    # 点か現在の撮影範囲内にあるかチェック
    def check_pt_is_inside_rect(self, pt):

        # x max.min 取得
        x_min = self.shooting_rect[0][0]
        x_max = self.shooting_rect[0][0]
        for i in range(4):
            x1 = self.shooting_rect[i][0]
            if x1 < x_min:
                x_min = x1
            if x1 > x_max:
                x_max = x1

        # x 範囲チェック
        if pt[0] < x_min or x_max < pt[0]:
            return False

        # y max.min 取得
        y_min = self.shooting_rect[0][1]
        y_max = self.shooting_rect[0][1]
        for i in range(4):
            y1 = self.shooting_rect[i][1]
            if y1 < y_min:
                y_min = y1
            if y1 > y_max:
                y_max = y1

        # y 範囲チェック
        if pt[1] < y_min or y_max < pt[1]:
            return False

        return True

    # 対象範囲を示す四角を描画
    def draw_rectangle(self, frame, x0, y0, x1, y1, r1, t1, color1):
        # Input
        #   x0,y0,x1,y1 : 矩形座標値
        #   r1[3,3] : 仮想マーカーのローカル座標系への回転行列
        #   t1[3]   : 　　　〃　　　　　　　　　　　　移動行列
        #   color1  : 色

        # 全体座標系を仮想マーカー座標系へ変換
        pt = np.zeros((4, 3))
        # 左下
        pt[0][0] = x0
        pt[0][1] = y0
        # 右下
        pt[1][0] = x1
        pt[1][1] = y0
        # 右上
        pt[2][0] = x1
        pt[2][1] = y1
        # 左上
        pt[3][0] = x0
        pt[3][1] = y1

        # 角度チェックの為の四角設定
        chk_rect = np.zeros((4, 3))
        len1 = self.marker_length
        # 左下
        chk_rect[0][0] = -t1[0]
        chk_rect[0][1] = -t1[1]
        # 右下
        chk_rect[1][0] = -t1[0] + len1
        chk_rect[1][1] = -t1[1]
        # 右上
        chk_rect[2][0] = -t1[0] + len1
        chk_rect[2][1] = -t1[1] + len1
        # 左上
        chk_rect[3][0] = -t1[0]
        chk_rect[3][1] = -t1[1] + len1

        # 座標変換
        pt_2d = np.zeros((4, 2))
        chk_rect_2d = np.zeros((4, 2))
        for i in range(4):
            # 全体座標系を仮想マーカー座標系へ変換
            for j, val in enumerate(t1):
                pt[i][j] += val
                chk_rect[i][j] += val
            pt[i] = np.dot(r1, pt[i])
            chk_rect[i] = np.dot(r1, chk_rect[i])
            # 投影座標へ変換
            pt_2d[i], jac_o = cv2.projectPoints(
                pt[i], self.rvec, self.tvec, self.camera_mat, self.dist_coef
            )
            chk_rect_2d[i], jac_o = cv2.projectPoints(
                chk_rect[i], self.rvec, self.tvec, self.camera_mat, self.dist_coef
            )
            # 収差補正前 → 後へ変換
            pt_2d[i] = cv2.undistortPoints(
                pt_2d[i], self.camera_mat, self.dist_coef, P=self.camera_mat
            )
            chk_rect_2d[i] = cv2.undistortPoints(
                chk_rect_2d[i], self.camera_mat, self.dist_coef, P=self.camera_mat
            )

        # X,Y角度チェック用
        vx_chk = chk_rect_2d[1] - chk_rect_2d[0]
        vy_chk = chk_rect_2d[3] - chk_rect_2d[0]

        # # チェック用四角描画
        # p1 = np.zeros(2, dtype=np.int16)
        # p2 = np.zeros(2, dtype=np.int16)
        # for i in range(2):
        #     p1[i] = chk_rect_2d[1][i]
        #     p2[i] = chk_rect_2d[2][i]
        # cv2.line(frame, p1, p2, color=color1, thickness=1, lineType=cv2.LINE_AA)
        # for i in range(2):
        #     p1[i] = chk_rect_2d[2][i]
        #     p2[i] = chk_rect_2d[3][i]
        # cv2.line(frame, p1, p2, color=color1, thickness=1, lineType=cv2.LINE_AA)

        # 四角描画
        p1 = np.zeros(2, dtype=np.int16)
        p2 = np.zeros(2, dtype=np.int16)
        d_ang = 10.0
        for i in range(3):
            # 点が画像内に収まるかどうか
            if not self.is_within_image(pt_2d[i]):
                continue
            if not self.is_within_image(pt_2d[i + 1]):
                continue
            # 左下・右下ライン(X座標比較)
            if i == 0:
                if not self.related_check_with_adjacent_point(
                    0, pt_2d[i], pt_2d[i + 1]
                ):
                    continue
                # X方向角度チェック
                vx1 = pt_2d[i + 1] - pt_2d[i]
                ang_x = self.angle_calc(vx1, vx_chk)
                if ang_x > d_ang:
                    print("下 over ang_x = {:.1f}".format(ang_x))
                    continue
                # print('下 ang_x = {:.1f}'.format(ang_x))
            # 右下・右上ライン(Y座標比較)
            elif i == 1:
                if not self.related_check_with_adjacent_point(
                    1, pt_2d[i + 1], pt_2d[i]
                ):
                    continue
                # Y方向角度チェック
                vy1 = pt_2d[i + 1] - pt_2d[i]
                ang_y = self.angle_calc(vy1, vy_chk)
                if ang_y > d_ang:
                    print("右 over ang_y = {:.1f}".format(ang_y))
                    continue
                # print('右 ang_y = {:.1f}'.format(ang_y))
            # 右上・左上ライン(X座標比較)
            elif i == 2:
                if not self.related_check_with_adjacent_point(
                    0, pt_2d[i + 1], pt_2d[i]
                ):
                    continue
                # X方向角度チェック
                vx1 = pt_2d[i + 1] - pt_2d[i]
                ang_x = self.angle_calc(vx1, vx_chk)
                if ang_x > d_ang:
                    print("上 over ang_x = {:.1f}".format(ang_x))
                    continue
                # print('上 ang_x = {:.1f}'.format(ang_x))

            for j in range(2):
                p1[j] = pt_2d[i][j]
                p2[j] = pt_2d[i + 1][j]
            cv2.line(frame, p1, p2, color=color1, thickness=2, lineType=cv2.LINE_AA)

        # 左縦線描画
        # 点が画像内に収まるかどうか
        if not self.is_within_image(pt_2d[3]):
            return
        if not self.is_within_image(pt_2d[0]):
            return
        if not self.related_check_with_adjacent_point(1, pt_2d[3], pt_2d[0]):
            return
        # Y方向角度チェック
        vy1 = pt_2d[3] - pt_2d[0]
        ang_y = self.angle_calc(vy1, vy_chk)
        if ang_y > d_ang:
            print("左 over ang_y = {:.1f}".format(ang_y))
            return
        # print('左 ang_y = {:.1f}'.format(ang_y))
        for i in range(2):
            p1[i] = pt_2d[3][i]
            p2[i] = pt_2d[0][i]
        cv2.line(frame, p1, p2, color=color1, thickness=1, lineType=cv2.LINE_AA)

    # ヒストグラムより取得したボルト位置格子を描画(1行目、1列目だけではなく全点に座標値をもつ)
    def draw_bolt_arr(self, frame, row1, col1, r1, t1):
        # Input
        #   row1[]  : ボルト位置縦座標値Y
        #   col1[]  : 　〃　　　横　〃　X
        #   r1[3,3] : 仮想マーカーのローカル座標系への回転行列
        #   t1[3]   : 　　　〃　　　　　　　　　　　　移動行列

        # 全体座標系を仮想マーカー座標系へ変換
        crd = np.zeros((len(row1) * len(col1), 2))
        crd_g = np.zeros((len(row1) * len(col1), 2))
        xy = np.zeros(3)
        # Rに基底マーカー回転分考慮
        # r1 = np.dot(self.__r_base_inv, r1)
        count = 0
        for j, y1 in enumerate(row1):
            for i, x1 in enumerate(col1):
                xy[0] = x1
                xy[1] = y1
                xy[2] = 0.0
                crd_g[count][0] = x1
                crd_g[count][1] = y1
                for k, val in enumerate(t1):
                    xy[k] += val
                xy = np.dot(r1, xy)
                crd[count][0] = xy[0]
                crd[count][1] = xy[1]
                count += 1

        # 投影座標へ変換
        crd2 = np.zeros((len(row1) * len(col1), 2), dtype=np.int16)
        count = 0
        for y, dy in enumerate(row1):
            for x, dx in enumerate(col1):
                # 3D点を画像に投影
                adr = len(col1) * y + x
                crd_x = crd[adr][0]
                crd_y = crd[adr][1]
                pt_3d = np.float32([crd_x, crd_y, 0])
                pt, jac_o = cv2.projectPoints(
                    pt_3d, self.rvec, self.tvec, self.camera_mat, self.dist_coef
                )

                # 収差補正前 → 後へ変換
                pt = cv2.undistortPoints(
                    pt, self.camera_mat, self.dist_coef, P=self.camera_mat
                )
                # 保存
                pt = pt.astype(np.int32)
                pt = np.squeeze(pt)
                crd2[count] = pt
                count += 1

        # 中心点・線描画
        op = np.zeros(2, dtype=np.int16)
        for y, dy in enumerate(row1):
            for x, dx in enumerate(col1):
                adr = len(col1) * y + x
                op = crd2[adr]
                if not self.check_pt_is_inside_rect(crd_g[adr]):
                    continue  # 点が現在のカメラ撮影範囲にあるか
                if not self.is_within_image(op):  # 点が画像内にあるか
                    continue
                if x != len(col1) - 1:  # 最後列でない場合
                    p_nx = crd2[adr + 1]
                    if not self.related_check_with_adjacent_point(0, op, p_nx):
                        continue
                else:  # 最後列の場合
                    op_pre = crd2[adr - 1]
                    if not self.related_check_with_adjacent_point(0, op_pre, op):
                        continue
                if y != len(row1) - 1:  # 最終行の場合
                    p_nx = crd2[adr + len(col1)]
                    if not self.related_check_with_adjacent_point(1, op, p_nx):
                        continue
                else:  # 最終行の場合
                    op_pre = crd2[adr - len(col1)]
                    if not self.related_check_with_adjacent_point(1, op_pre, op):
                        continue
                cv2.circle(frame, center=op, radius=5, color=(0, 0, 255), thickness=-1)
                # str1 = "{0:d}, {1:d}".format(y+1, x+1)
                # cv2.putText(frame, str1, tuple(op),
                #             fontFace=cv2.FONT_ITALIC, fontScale=1.0, color=(0, 0, 255), thickness=1)

                if x != 0:
                    op_bef = crd2[adr - 1]
                    if self.is_within_image(op_bef):
                        cv2.line(
                            frame,
                            op_bef,
                            op,
                            color=(0, 0, 255),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )
                if y != 0:
                    op_up = crd2[adr - len(col1)]
                    if self.is_within_image(op_up):
                        cv2.line(
                            frame,
                            op_up,
                            op,
                            color=(0, 0, 255),
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

    @staticmethod
    # 隣の点との位置関係が正しいかチェック
    def related_check_with_adjacent_point(type1, pt1, pt2):
        # type1 : 0=X方向チェック、1=Y方向チェック
        # 1点目のx座標が2点目より大きい場合NG
        if type1 == 0:
            if pt1[0] >= pt2[0]:
                return False
        # 1点目のY座標が2点目より大きい場合NG
        elif type1 == 1:
            if pt1[1] >= pt2[1]:
                return False

        return True

    # ポイントが画像内に入っているかどうか
    def is_within_image(self, pt):
        if pt[0] < 0 or pt[0] > self.__image_width:
            return False
        if pt[1] < 0 or pt[1] > self.__image_height:
            return False

        return True

    def draw_bolt_order(self, frame, x1, y1, r1, t1, order, kind):
        # Input
        #   x1      : ボルト位置縦座標値Y
        #   y1      : 　〃　　　横　〃　X
        #   r1[3,3] : 仮想マーカーのローカル座標系への回転行列
        #   t1[3]   : 　　　〃　　　　　　　　　　　　移動行列
        #   order   : 締め付け順序
        #   kind    : 0=白スタンプ、1=黄スタンプ

        # ボルト位置が現在のカメラ撮影範囲に入っているかチェック
        pt1 = np.array([x1, y1])
        if not self.check_pt_is_inside_rect(pt1):
            return

        # 全体座標系を仮想マーカー座標系へ変換
        xy = np.zeros(3)
        xy[0] = x1
        xy[1] = y1
        xy[2] = 0.0
        # Rに基底マーカー回転分考慮
        # r1 = np.dot(self.__r_base_inv, r1)
        for k, val in enumerate(t1):
            xy[k] += val
        xy = np.dot(r1, xy)
        crd_x = xy[0]
        crd_y = xy[1]

        # 投影座標へ変換
        # 3D点を画像に投影
        pt_3d = np.float32([crd_x, crd_y, 0])
        pt, jac_o = cv2.projectPoints(
            pt_3d, self.rvec, self.tvec, self.camera_mat, self.dist_coef
        )
        pt_int = pt.astype(np.int64)
        pt_int = tuple(pt_int.ravel())
        if not self.is_within_image(pt_int):  # 点が画像内にあるか
            return
        # 収差補正前 → 後へ変換
        pt = cv2.undistortPoints(pt, self.camera_mat, self.dist_coef, P=self.camera_mat)
        # 保存
        pt = pt.astype(np.int64)
        pt = np.squeeze(pt)

        if not self.is_within_image(pt.astype(np.int64)):  # 点が画像内にあるか
            return

        # 数字描画
        # 画像へ表示
        text = "{0:d}".format(order)  # 締め付け順序
        fontFace = cv2.FONT_ITALIC  # フォント
        fontScale = 0.8  # 文字のサイズ
        thickness = 1  # 文字の太さ
        # 文字の大きさを測る
        size, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
        # 四角描画
        pt_st = pt.copy()
        pt_end = pt.copy()
        pt_st[1] -= size[1] + baseLine  # Y方向始点
        pt_end[1] += baseLine  # 〃　 終点

        # 白スタンプ
        if kind == 0:
            pt[0] -= size[0]  # 文字X方向
            pt_st[0] -= size[0]  # 四角X方向始点
            mat_color = (255, 255, 255)
            if not self.is_within_image(pt_st):
                return
            if not self.is_within_image(pt_end):
                return
            cv2.rectangle(frame, pt_st, pt_end, mat_color, -1)  # 白で塗りつぶし
        # 黄スタンプ
        elif kind == 1:
            pt_end[0] += size[0]  # 四角X方向終点
            mat_color = (0, 217, 255)
            if not self.is_within_image(pt_st):
                return
            if not self.is_within_image(pt_end):
                return
            cv2.rectangle(frame, pt_st, pt_end, mat_color, -1)  # 黄で塗りつぶし

        # 黒四角ライン描画
        if not self.is_within_image(pt_st):
            return
        if not self.is_within_image(pt_end):
            return
        cv2.rectangle(frame, pt_st, pt_end, color=(0, 0, 0), thickness=0)
        # 締め付け順序数字描画
        if not self.is_within_image(pt):
            return
        cv2.putText(
            frame,
            text,
            pt,
            fontFace=fontFace,
            fontScale=fontScale,
            color=(0, 0, 0),
            thickness=thickness,
        )

    def draw_bolt_OK(
        self, frame, x_left, y_left, x_right, y_right, r1, t1, order, kind
    ):
        # Input
        #   x1      : ボルト位置縦座標値Y
        #   y1      : 　〃　　　横　〃　X
        #   r1[3,3] : 仮想マーカーのローカル座標系への回転行列
        #   t1[3]   : 　　　〃　　　　　　　　　　　　移動行列
        #   order   : 締め付け順序
        #   kind    : 0=白スタンプ、1=黄スタンプ

        # ボルト位置が現在のカメラ撮影範囲に入っているかチェック
        pt1 = np.array([x_left, y_left])
        # pt2 =
        if not self.check_pt_is_inside_rect(pt1):
            return

        # 全体座標系を仮想マーカー座標系へ変換
        xy_left = np.zeros(3)
        xy_left[0] = x_left
        xy_left[1] = y_left
        xy_left[2] = 0.0
        xy_right = np.zeros(3)
        xy_right[0] = x_right
        xy_right[1] = y_right
        xy_right[2] = 0.0
        # Rに基底マーカー回転分考慮
        # r1 = np.dot(self.__r_base_inv, r1)
        for k, val in enumerate(t1):
            xy_left[k] += val
        xy_left = np.dot(r1, xy_left)
        crd_x_left = xy_left[0]
        crd_y_left = xy_left[1]

        # 投影座標へ変換
        # 3D点を画像に投影
        pt_3d_left = np.float32([crd_x_left, crd_y_left, 0])
        pt, jac_o = cv2.projectPoints(
            pt_3d_left, self.rvec, self.tvec, self.camera_mat, self.dist_coef
        )
        pt_int = pt.astype(np.int64)
        pt_int = tuple(pt_int.ravel())
        if not self.is_within_image(pt_int):  # 点が画像内にあるか
            return
        # 収差補正前 → 後へ変換
        pt = cv2.undistortPoints(pt, self.camera_mat, self.dist_coef, P=self.camera_mat)
        # 保存
        pt = pt.astype(np.int64)
        pt = np.squeeze(pt)

        if not self.is_within_image(pt.astype(np.int64)):  # 点が画像内にあるか
            return

        # 数字描画
        # 画像へ表示
        # text = "{0:d}".format(order)  # 締め付け順序
        text = order
        fontFace = cv2.FONT_ITALIC  # フォント
        fontScale = 0.8  # 文字のサイズ
        thickness = 1  # 文字の太さ
        # 文字の大きさを測る
        size, baseLine = cv2.getTextSize(text, fontFace, fontScale, thickness)
        # 四角描画
        pt_st = pt.copy()
        pt_end = pt.copy()
        pt_st[1] -= size[1] + baseLine  # Y方向始点
        pt_end[1] += baseLine  # 〃　 終点

        # 白スタンプ
        if kind == 0:
            pt[0] -= size[0]  # 文字X方向
            pt_st[0] -= size[0]  # 四角X方向始点
            mat_color = (255, 255, 255)
            if not self.is_within_image(pt_st):
                return
            if not self.is_within_image(pt_end):
                return
            cv2.rectangle(frame, pt_st, pt_end, mat_color, -1)  # 白で塗りつぶし
        # 黒四角ライン描画
        if not self.is_within_image(pt_st):
            return
        if not self.is_within_image(pt_end):
            return
        cv2.rectangle(frame, pt_st, pt_end, color=(0, 0, 0), thickness=0)
        # 締め付け順序数字描画
        if not self.is_within_image(pt):
            return
        cv2.putText(
            frame,
            text,
            pt,
            fontFace=fontFace,
            fontScale=fontScale,
            color=(0, 0, 0),
            thickness=thickness,
        )
