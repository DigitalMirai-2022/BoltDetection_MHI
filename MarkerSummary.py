from threading import Lock
import numpy as np
import cv2.aruco as aruco

# 自作
from ArUcoMarker import DataArUco
from MarkerRt import CalcRt
from Output_Log import LogMan


class MarkerSummary:

    # 条件1. 同じ型のインスタンスをprivate なクラス変数として定義する。
    #       1インスタンスしか生成しないことを保証するために利用する。
    _unique_instance = None
    _lock = Lock()  # クラスロック（マルチスレッド対応）

    # 条件2. コンストラクタの可視性をprivateとする。
    #        pythonの場合、コンストラクタをprivate定義できない。
    #        コンストラクタ呼び出しさせず、インスタンス取得をget_instanceに限定する。
    #        get_instanceからインスタンス取得を可能にするため、__init__は使用しない。
    #        初期化時に、__new__が__init__よりも先に呼び出される。
    def __new__(cls):
        raise NotImplementedError("Cannot initialize via Constructor")

    # インスタンス生成
    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    # 条件3:同じ型のインスタンスを返す `getInstance()` クラスメソッドを定義する。
    @classmethod
    def get_instance(cls):
        # インスタンス未生成の場合
        if not cls._unique_instance:
            with cls._lock:
                if not cls._unique_instance:
                    cls._unique_instance = cls.__internal_new__()
        return cls._unique_instance

    ############################################################################
    # 宮地様向け変数
    # __ をつけて隠蔽していることを示す ↓
    __nArUco = 0  # 認識したマーカー数
    __list_ArU = []  # 上記の詳細。ArUcoMaker.DataArUco型変数
    __base_no = 0  # 基底マーカーNo（デフォルト = 0 ）
    __f_baseMarker = False  # 基底マーカーを認識したかフラグ
    __baseMarker = None  # 基底マーカー詳細。ArUcoMaker.DataArUco型変数
    __axis_no = -1  # 軸方向決定マーカーNo
    __axis_dirct = 0  # 基底マーカーと__axis_noによって決定される軸方向。0=x,1=y
    __f_axisMarker = False  # 軸方向決定用マーカーを認識したかフラグ
    __f_axis_comp = False  # 2マーカーを使用した座標軸作成完了フラグ
    __r_base = np.identity(3)  # 基底マーカーと全体座標の回転行列
    __marker_length = 0.0  # マーカー1辺サイズ
    __lim_Interval = 10  # インターバル数
    __area_x = [0.0, 0.0]  # ArUcoMakerのx座標min,max
    __area_y = [0.0, 0.0]  # 〃          y
    __lower_y = 0.0  # 基底ArUcoマーカーよりY負方向へ設ける余白サイズ(mm)
    __f_marker_no_fixed = False  # 使用ArUco noを限定するか否かフラグ
    __use_code_no = []  # 〃　　　　　　リスト

    log = LogMan.get_instance()  # logインスタンスの生成

    @staticmethod
    # ArUcoマーカー設定
    def set_ar_marker_param():

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        parameters.cornerRefinementMethod = (
            aruco.CORNER_REFINE_SUBPIX
        )  # AruCoマーカーの検出されたコーナーをサブピクセル精度で修正
        parameters.minMarkerPerimeterRate = 0.125  # 検出されるマーカー輪郭の最小周囲長。入力画像の最大寸法に対する割合として定義(デフォルトは 0.03)。GoProの場合0.11とする。
        parameters.polygonalApproxAccuracyRate = (
            0.05  # 正方形への近似プロセス最小精度(デフォルトは 0.03)
        )

        # parameters.minMarkerPerimeterRate = 0.155
        # parameters.minMarkerPerimeterRate = 0.175

        # # パラメータ調整で検出精度向上を図る
        # parameters.adaptiveThreshWinSizeMin = 5
        # parameters.adaptiveThreshWinSizeMax = 21
        # parameters.adaptiveThreshWinSizeStep = 7
        # parameters.minMarkerPerimeterRate = 0.125
        # parameters.maxMarkerPerimeterRate = 4.0
        # parameters.minCornerDistanceRate = 0.05
        # parameters.minMarkerDistanceRate = 0.02
        # parameters.polygonalApproxAccuracyRate = 0.04
        # parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        return aruco_dict, parameters

    @property
    def nArUco(self):
        return self.__nArUco

    @property
    def f_baseMarker(self):
        return self.__f_baseMarker

    @property
    def base_no(self):
        return self.__base_no

    @property
    def axis_no(self):
        return self.__axis_no

    @axis_no.setter
    def axis_no(self, no):
        self.__axis_no = no

    @property
    def axis_dirct(self):
        return (
            self.__axis_dirct
        )  # 基底マーカーと__axis_noによって決定される軸方向。0=x,1=y

    @axis_dirct.setter
    def axis_dirct(self, no):
        self.__axis_dirct = no

    @property
    def f_axisMarker(self):
        return self.__f_axisMarker  # 軸方向決定用マーカーを認識したかフラグ

    @property
    def f_axis_comp(self):
        return self.__f_axis_comp  # 2マーカーを使用した座標軸作成完了フラグ

    @property
    def min_x(self):
        return self.__area_x[0]

    @property
    def max_x(self):
        return self.__area_x[1]

    @property
    def min_y(self):
        return self.__area_y[0]

    @property
    def max_y(self):
        return self.__area_y[1]

    @property
    def marker_length(self):
        return self.__marker_length

    @marker_length.setter
    def marker_length(self, length):
        if length <= 0:
            message = "marker_length value error : %d" % length
            self.log.exception(__file__, message)
            raise ValueError("marker_length value error")
        self.__marker_length = length

    @property
    def lower_y(self):
        return self.__lower_y

    @lower_y.setter
    def lower_y(self, y):
        self.__lower_y = y

    @property
    def f_marker_no_fixed(self):
        return self.__f_marker_no_fixed

    @f_marker_no_fixed.setter
    def f_marker_no_fixed(self, f_marker_no_fixed):
        self.__f_marker_no_fixed = f_marker_no_fixed

    @property
    def r_base(self):
        return self.__r_base

    @r_base.setter
    def r_base(self, r1):
        self.__r_base = r1

    # 設定ファイルで指定されたArUcoマーカーno登録
    def set_ini_code_no(self, list):
        for i, no in enumerate(list):
            self.__use_code_no.append(no)

    # クラス変数ArUcoマーカーリスト（self.__list_ArU[]）より基底マーカーのアドレス取得
    def get_adr_bese_in_list_aruco(self):

        for i, listd in enumerate(self.__list_ArU):
            if listd.no == self.__base_no:
                return i
        return -1

    # 基底マーカーセット
    def add_marker_base(self, corner_ax):
        # 引数
        #   corner_ax : 軸方向マーカー四隅座標値(基底マーカー基準)
        # 返り値 True,False

        # 四隅の回転前座標値セット
        d = self.marker_length / 2.0
        crd_o = np.zeros([4, 3])
        crd_o[0][0] = -d
        crd_o[0][1] = d
        crd_o[0][2] = 0.0
        crd_o[1][0] = d
        crd_o[1][1] = d
        crd_o[1][2] = 0.0
        crd_o[2][0] = d
        crd_o[2][1] = -d
        crd_o[2][2] = 0.0
        crd_o[3][0] = -d
        crd_o[3][1] = -d
        crd_o[3][2] = 0.0

        # 基底マーカーが未設定の場合、回転前基底マーカーセット（self.__baseMarker）
        iadr = -1
        if not self.__f_baseMarker:
            # クラス変数へマーカー追加
            a = DataArUco(self.__base_no, crd_o)
            a.status = 0
            self.__baseMarker = a

        # 認識済みの場合、平均化するタイミングか確認
        else:
            # self.__list_ArU[]内基底マーカー取得
            iadr = self.get_adr_bese_in_list_aruco()
            if iadr < 0:
                return False
            # インターバル回数チェック
            base_in_list = self.__list_ArU[iadr]
            base_in_list.interval += 1
            if base_in_list.interval < self.__lim_Interval:
                return True

        # 軸指定マーカーの中心点算出
        sumx = 0.0
        sumy = 0.0
        for i in range(4):
            sumx += corner_ax[i][0]
            sumy += corner_ax[i][1]
        sumx /= 4.0
        sumy /= 4.0
        pt_ax = np.zeros(2)
        pt_ax[0] = sumx
        pt_ax[1] = sumy

        # 基底マーカーは原点とし、軸指定マーカーとの距離算出
        pt_org = np.zeros(2)
        d1 = np.linalg.norm(pt_ax - pt_org)

        # 傾きの算出
        if self.__axis_dirct == 0:  # x軸方向指定の場合
            sin1 = pt_ax[1] / d1
            cos1 = pt_ax[0] / d1
        else:  # y軸方向指定の場合
            sin1 = -pt_ax[0] / d1
            cos1 = pt_ax[1] / d1

        # 回転行列の作成
        R = np.zeros((3, 3))
        R[0][0] = cos1
        R[0][1] = -sin1
        R[0][2] = 0.0
        R[1][0] = sin1
        R[1][1] = cos1
        R[1][2] = 0.0
        R[2][0] = 0.0
        R[2][1] = 0.0
        R[2][2] = 1.0
        R_I = np.linalg.inv(R)

        # 4角を回転
        crd_R = np.zeros([4, 3])
        for i in range(4):
            crd1 = np.dot(R_I, crd_o[i])
            crd_R[i] = crd1

        # 基底マーカーが未設定の場合
        if not self.__f_baseMarker:
            # リストへ追加
            b = DataArUco(self.__base_no, crd_R)
            b.status = 0  # 全体座標系とのRt取得
            b.interval = 0
            b.cnt = 1
            b.R = R_I
            self.__list_ArU.append(b)
            self.__nArUco += 1
            message = "set base marker %d, num = %d" % (b.no, self.__nArUco)
            self.log.info(__file__, message)
            # クラス変数の基底マーカー回転量セット
            self.__r_base = R_I
            # フラグセット
            self.__f_baseMarker = True
        # 平均化する場合
        else:
            # 既存データに加える
            base_in_list = self.__list_ArU[iadr]
            base_in_list.R, sita1 = self.averaging_R(
                base_in_list.cnt, base_in_list.R, R_I
            )  # Rの平均化
            base_in_list.cnt += 1
            base_in_list.interval = 0
            message = (
                f" (Averaging : base marker no , Z axis rotation+(deg) : {sita1:.3f})"
            )
            self.log.debug(__file__, message)
            # クラス変数の基底マーカー回転量セット
            self.__r_base = base_in_list.R

        return True

    # ArUcoマーカーを追加する(一般マーカー)
    def add_marker(self, no, corner):
        # 返り値　0:一般マーカー追加、1:基底マーカー追加、2:既存マーカーあり、-1:エラー

        # 基底マーカーが見つけられていない場合、無視
        if not self.__f_baseMarker:
            return -1

        # 基底マーカーは除外
        if no == self.__base_no:
            return 2

        # jsonファイルで使用マーカーが指定されている場合
        if self.__f_marker_no_fixed:
            f1 = False
            for i, no1 in enumerate(self.__use_code_no):
                if no == no1:
                    f1 = True
                    break
            if not f1:
                message = f"Marker {no} is not specified"
                self.log.info(__file__, message)
                return -1

        # 既存マーカーと重複する場合、無視
        if self.is_exists_marker_no(no):
            if no == self.__base_no:
                return 1
            return 2

        # クラス変数のリストへ追加
        b = DataArUco(no, corner)
        self.__list_ArU.append(b)
        self.__nArUco += 1
        message = "added marker no. %d, Number of Codes = %d" % (b.no, self.__nArUco)
        self.log.info(__file__, message)

        return 0

    # ArUcoMarker座標の最大・最小値初期化
    def init_marker_max_min(self):
        self.__area_x[0] = 0.0
        self.__area_x[1] = 0.0
        self.__area_y[0] = 0.0
        self.__area_y[1] = 0.2

    # ArUcoMarker座標の最大・最小値セット
    def set_marker_max_min(self):

        # 基底マーカーが認識されていない場合、無視
        if not self.__f_baseMarker:
            return

        # max・min取得
        min_x = 1e6
        max_x = -1.0e6
        min_y = 1e6
        max_y = -1.0e6
        for i, mark in enumerate(self.__list_ArU):
            if mark.status == -1:
                continue
            # 余白＋側
            x1 = mark.t[0][0]
            y1 = mark.t[1][0]
            if x1 < min_x:
                min_x = x1
            if x1 > max_x:
                max_x = x1
            if y1 < min_y:
                min_y = y1
            # if self.__lower_y > max_y:
            #     max_y = self.__lower_y
            # min_y = 0
            max_y = self.__lower_y
        self.__area_x[0] = min_x
        self.__area_x[1] = max_x
        self.__area_y[0] = min_y
        self.__area_y[1] = max_y
        # message = f"Target range : x = {self.__area_x[0]} - {self.__area_x[1]}, y = {self.__area_y[0]} - {self.__area_y[1]}"
        # self.log.debug(__file__, message)

    # 番号noのマーカーか存在するかチェック
    def is_exists_marker_no(self, no):
        for w in self.__list_ArU:
            if w.no == no:
                return True
        return False

    # _list_ArU[].corner座標値セット、及びR・t設定
    def set_corner(self, no, crd, kind, no_vir_base=-1):
        # no : マーカーno
        # crd[] : 4角三次元座標値
        # kind:0 = 基底マーカーを原点としたcrd、kind:1 = 一般マーカー"no_vir_base"を原点としたcrd(基底マーカー基準のR･tを算出し直し)
        # no_vir_base : kind=1の場合一般マーカーno

        r_base = self.__r_base

        for i, listd in enumerate(self.__list_ArU):
            if listd.no != no:
                continue
            # 4角三次元座標値登録
            listd.corner = crd

            # 基底マーカーとのR.t算出・登録
            if kind == 0:
                # R,t算出
                M_Rt = CalcRt(self.__baseMarker.corner, crd)
                M_Rt.get_Rt()
                # 基底マーカーの傾き考慮
                M_Rt.rotate(r_base)
                # 既存データと10mm異なる場合、無視
                if listd.cnt != 0 and np.linalg.norm(M_Rt.t - listd.t) > 0.1:
                    continue
                # 四隅三次元座標も基底マーカー傾き考慮
                M_Rt.rotate_corner(listd.corner, r_base)
                # 既存データの場合、基底マーカーとの関係性取得
                istatus = self.get_status(no)
                # 基底マーカーとのRtがない場合
                if istatus == 1 or istatus == -1:
                    listd.R = M_Rt.R
                    listd.t = M_Rt.t
                    listd.cnt = 1
                    listd.interval = 0
                    listd.virtual_no = -1
                    listd.status = 0  # 基底マーカーとのRt取得
                    message = f" (marker no = {no:.0f} : Get relationship with base )"
                    self.log.info(__file__, message)
                    message = "marker t = ( %.1f, %.1f)" % (listd.t[0], listd.t[1])
                    self.log.info(__file__, message)

                # 基底マーカーとのRtがある場合、平均値算出
                else:
                    listd.interval += 1
                    if listd.interval < self.__lim_Interval:
                        return True
                    # 既存データに加える
                    sita1 = 0.0
                    del1 = 0.0
                    listd.R, sita1 = self.averaging_R(
                        listd.cnt, listd.R, M_Rt.R
                    )  # Rの平均化
                    listd.t, del1 = self.averaging_t(
                        listd.cnt, listd.t, M_Rt.t
                    )  # tの平均化
                    listd.cnt += 1
                    listd.interval = 0
                    message = f" (Averaging : marker no = {no:.0f}, Z axis rotation+(deg) : {sita1:.3f}, xy+(mm) : {del1:.3f})"
                    self.log.debug(__file__, message)
                listd.status = 0
                # 軸指定マーカーが設定されたら座標系OKフラグをたてる
                if no == self.__axis_no:
                    self.__f_axis_comp = True
                break
            # 一般マーカーとのR・ｔ算出・登録
            elif kind == 1:
                if listd.virtual_no >= 0 and listd.virtual_no != no_vir_base:
                    return False
                if listd.virtual_no < 0:
                    message = f" (virtual base no = {no_vir_base:.0f} )"
                    self.log.info(__file__, message)
                # 既存データの場合、基底マーカーとの関係性取得
                istatus = self.get_status(no)
                # 既存データがあり、interval以下の場合抜ける
                if istatus == 1:
                    listd.interval += 1
                    if listd.interval < self.__lim_Interval:
                        return True
                # 一般マーカーとのRt算出
                M_Rt = CalcRt(self.__baseMarker.corner, crd)
                M_Rt.get_Rt()
                R1 = M_Rt.R
                t1 = M_Rt.t
                R2, t2 = self.get_Rt_by_no(no_vir_base)
                det2 = np.linalg.det(R2)
                if det2 == 0.0:
                    return False
                add_R = np.dot(R2, R1)
                add_t = np.dot(R2, t1) + t2
                # 既存データに加える
                # 新規データの場合
                if istatus == -1:
                    listd.R = add_R
                    listd.t = add_t
                # 既存データと平均化する
                elif istatus == 1:
                    # 既存データと10mm異なる場合、無視
                    if listd.cnt != 0 and np.linalg.norm(add_t - listd.t) > 0.1:
                        continue
                    listd.R, sita2 = self.averaging_R(
                        listd.cnt, listd.R, add_R
                    )  # Rの平均化
                    listd.t, del2 = self.averaging_t(
                        listd.cnt, listd.t, add_t
                    )  # tの平均化
                    message = f" (Averaging : marker no = {no:.0f}, vir_base = {no_vir_base:.0f}, Z axis rotation+(deg) : {sita2:.3f}, xy+(mm) : {del2:.3f})"
                    self.log.debug(__file__, message)
                else:
                    return False
                listd.cnt += 1
                # 属性
                listd.interval = 0
                listd.status = 1
                listd.virtual_no = no_vir_base
                break

        # ArUcoMakerの座標max・minセット
        self.set_marker_max_min()

    # 点座標(x,y,z)に基底マーカー回転分をかける
    def rotate_base_arc(self, pt):

        return np.dot(self.__r_base, pt)

    # リストに１つデータを加えて平均化する(t[3][1])
    @staticmethod
    def averaging_t(n1, d1, add):
        # Input
        #   n1  : 元のデータ数
        #   d1[]: 元のデータ配列
        #   add : 加える新データ
        # Output
        #   d_out: 更新後のデータ配列
        #   delta  : 変化量

        p1 = n1 * d1
        p1 = p1 + add
        d_out = p1 / (n1 + 1)

        # 差分取得
        row, col = d1.shape
        delta = 0.0
        for i in range(3):
            a = d_out[i][0] - d1[i][0]
            delta += a * a
        delta = np.sqrt(delta)
        delta *= 1000.0  # m → mmへ変換

        return d_out, delta

    # リストに１つデータを加えて平均化する(R[3][3])
    @staticmethod
    def averaging_R(n1, d1, add):
        # Input
        #   n1  : 元のデータ数
        #   d1[]: 元のデータ配列
        #   add : 加える新データ
        # Output
        #   d_out: 更新後のデータ配列
        #   sita_out : 新回転角
        #   delta  : 変化量

        # 元データの回転角を求める
        sin1 = d1[1][0]
        cos1 = d1[0][0]
        sita1 = 0
        if 0 <= sin1 <= np.pi:
            sita1 = np.arccos(cos1)
        elif -np.pi <= sin1 < 0:
            sita1 = np.arccos(cos1) * -1.0

        # 加えるデータの回転角を求める
        sin2 = add[1][0]
        cos2 = add[0][0]
        sita2 = 0
        if 0 <= sin2 <= np.pi:
            sita2 = np.arccos(cos2)
        elif -np.pi <= sin2 < 0:
            sita2 = np.arccos(cos2) * -1.0

        # 平均化
        sum1 = n1 * sita1
        sum1 += sita2
        sita_out = sum1 / (n1 + 1)

        # 回転行列セット
        d_out = np.identity(3)  # 3x3単位行列
        d_out[0][0] = np.cos(sita_out)
        d_out[0][1] = np.sin(sita_out) * -1.0
        d_out[1][0] = np.sin(sita_out)
        d_out[1][1] = np.cos(sita_out)

        # 差分取得
        delta = np.degrees(sita2 - sita1)

        return d_out, delta

    def get_Rt_by_no(self, no):

        for i, listd in enumerate(self.__list_ArU):
            if listd.no != no:
                continue
            return listd.R, listd.t

        # 該当なしの場合
        r = np.zeros([3, 3])
        t = np.zeros(3)
        return r, t

    # _list_ArU[].corner座標値セット、及びR・t設定
    def get_status(self, no):
        # no : マーカーno

        for i, listd in enumerate(self.__list_ArU):
            if listd.no != no:
                continue
            return listd.status

    # 3D座標から画像内2Dへの投影座標値を求める
    def get_nearest_marker(self, ids, pt_3d):
        # Input
        #   ids[]   : 現在画像内に存在するマーカーNo
        #   pt_3d[] : 3D座標値
        # Output
        #   2D投影座標

        R_out = np.zeros([3, 3])
        t_out = np.zeros(3)

        if self.__nArUco <= 0:
            return False, -1, R_out, t_out

        # 撮影されているマーカーで最も近いマーカー取得
        d_min = 1.0e10
        no_min = -1
        for i, ino in enumerate(ids):
            # マーカーの存在確認
            if not self.is_exists_marker_no(ino):
                continue
            # マーカーの位置取得
            R1, t1 = self.get_Rt_by_no(ino)
            t1 = np.squeeze(t1)
            # 中心点との距離確認
            d = np.linalg.norm(pt_3d - t1)
            if d < d_min:
                d_min = d
                no_min = ino
        if no_min < 0:
            return False, -1, R_out, t_out

        R_out, t_out = self.get_Rt_by_no(no_min)
        R_out = np.linalg.inv(R_out)
        t_out = t_out * -1.0
        t_out = np.squeeze(t_out)
        return True, no_min, R_out, t_out
