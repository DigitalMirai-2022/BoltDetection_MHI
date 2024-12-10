import numpy as np
import cv2
from cv2 import aruco
import json
import tkinter as tk
import threading
import queue
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image, ImageDraw, ImageFont

import csv
import os
from datetime import datetime
import time
import math
from collections import defaultdict
from typing import Dict, Union

# YOLOX検出用
from yolox.detect import YOLOX
from yolox.coco_classes import COCO_CLASSES

# 自作
from Camera import Camera
from Draw import Draw
from MarkerSummary import MarkerSummary
from Output_Log import LogMan
from Message_Dialog import QR_INFO, msgbox_edit_config
from yolox.visualize import vis_counted_position
from config_class import (
    MarkerSize,
    Axis,
    YRange,
    CodeSpecification,
    BoltStatus,
    BoltDistance,
    Device,
    Config,
)

# tkinterのルートウィンドウ非表示
root = tk.Tk()
root.attributes("-topmost", True)
root.withdraw()


class Measure:
    def __init__(self, video_input, camera_mat, dist_coef, width1, height1):
        self.video_input = video_input
        self.camera_mat = camera_mat
        self.dist_coef = dist_coef
        self.image_width = width1
        self.image_height = height1
        self.marker_length = 0.0  # 単位=m
        # 収差補正後画像
        self.undistorted_image = None
        # マーカー関連初期化
        self.rvec_base = []  # 基底マーカー用
        self.tvec_base = []  # 〃
        self.cam_pos_base = []  # 〃
        self.list_no = []  # 一般マーカー用リスト
        self.list_rvec = []  # 〃
        self.list_tvec = []  # 〃
        self.list_cam_pos = []  # 〃
        self.no_vir_base = -1  # 仮の基底マーカー用
        self.rvec_vir = []  # 〃
        self.tvec_vir = []  # 〃
        self.cam_pos_vir = []  # 〃
        # 一時的なカメラによる撮影範囲
        self.shooting_rect = np.zeros((4, 3))

        # Cameraクラス定義
        self.Cam = Camera(self.camera_mat, self.dist_coef, self.marker_length)

        # シングルトンデータクラス
        self.m_Marker = MarkerSummary.get_instance()
        self.log = LogMan.get_instance()

        # ボルト状況取得関連
        self.status_change_count = 0  # ボルトクラス状況のカウント
        self.Time_interval = 1.0  # # ボルト状況取得 時間間隔の初期値
        self.bolt_statue_time = 0  # ボルト状況タイムスタンプ用
        self.bolt_distance = None  # 各ボルト間の距離

        # スレッド
        self.thread = threading.Thread(target=self.frame_update, daemon=True)
        self.stop_threads = False  # スレッド終了フラグ
        # フレーム共用のキュー
        self.q_frames = queue.Queue(maxsize=5)

        # 設定ファイルパス
        self.config_path = "config.json"
        # 設定情報初期化
        self.setting = None

        # 設定情報修正して、次のステップに移行のフラグ
        self.next = False

        # QRコード読み取った際のフラグ
        self.clicked = False
        # QRコード情報初期化
        self.qr_info = None
        # 使用デバイス
        self.device = None

        # 描画関連
        self.count_display = []  # カウント表示
        self.text_display = ["OK", "NG", "PIN", "判定不可", "合計"]  # 表示テキスト

        # 検出モデル関連
        self.bolt_pos = []  # 全ての位置リスト（中心点）
        self.bolt_cls = []  # 全てのクラスリスト
        self.all_left_top_xy = []  # 全ての位置リスト（左上）
        self.previous_status_count_tracker = (
            {}
        )  # 前のフレーム：各クラスごとの位置の連続検出回
        self.status_count_tracker = defaultdict(
            lambda: defaultdict(int)
        )  # 現在のフレーム：各クラスごとの位置の連続検出回数
        self.valid_centers = defaultdict(
            list
        )  # 各クラスごとの有効な位置リスト（中心点）
        self.valid_boxes = defaultdict(list)  # 各クラスごとの有効な位置リスト（左上）
        self.fixed_positions = []  # OKクラス位置固定用

    def main(self):

        # json読み込み
        self.read_json()

        # YOLOX検出準備
        predictor = self.set_yolo()

        # Set AR Marker Param
        aruco_dict, parameters = self.m_Marker.set_ar_marker_param()

        # OpenCV画像取得用スレッドの開始
        self.thread.start()

        # AIモデル既読フラグ
        f_read_model = False

        ids = None  # マーカーIDリスト初期化
        corners = None  # マーカーの隅リスト初期化

        while not self.stop_threads:

            # キューより画像取得
            # queueが空の場合は無視
            if self.q_frames.empty():
                continue
            # キューより画像取得
            self.undistorted_image = self.q_frames.get()

            if not self.next:
                # 設定情報修正
                edit_config = msgbox_edit_config(root, self.config_path, self.setting)
                root.wait_window(edit_config.dialog)  # ダイアログが閉じるまで待機
                if edit_config.save:  # OKボタンクリックした場合
                    self.setting = edit_config.config
                    # ユーザー入力した値を更新して、利用
                    self.Time_interval = self.setting["bolt status"]["loading interval"]
                    self.status_change_count = self.setting["bolt status"][
                        "status change count"
                    ]
                    self.m_Marker.lower_y = self.setting["Y range"]["Lower y"]

                    self.log.info(__file__, f"判定間隔変更後: {self.Time_interval} 秒")
                    self.log.info(
                        __file__, f"クラス固定回数変更後: {self.status_change_count} 回"
                    )
                    self.log.info(
                        __file__,
                        f"ArUco垂直方向距離変更後: {self.m_Marker.lower_y * 1000:.0f} mm",
                    )

                    # 次のステップに移行フラグ
                    self.next = True

                else:  # キャンセルが押された場合、ダイアログのXをクリックした場合
                    self.stop()  # システム終了
                    break

            else:
                # QR検知とユーザーの操作により処理
                if self.qr_info is None:
                    self.qr_info = self.detect_QR_pyzbar(self.undistorted_image)
                else:
                    if not self.clicked:
                        self.qr_info = json.loads(self.qr_info.replace("'", '"'))
                        display_infor = f"添接番号: {self.qr_info['添接番号']}, ボルト本数: {self.qr_info['ボルト本数']}, 首下長さ: {self.qr_info['首下長さ']}, 種類: {self.qr_info['種類']}"
                        # QR_INFOのウィンドウをフレームの中央に配置するためのX, Y座標を計算
                        dialog_qr_info = QR_INFO(root, display_infor)
                        root.wait_window(
                            dialog_qr_info.dialog
                        )  # ダイアログが閉じるまで待機
                        if dialog_qr_info.run:  # OKが押された場合
                            self.clicked = True

                        else:  # キャンセルが押された場合、ダイアログのXをクリックした場合、QR検知へ戻るため初期化
                            self.clear_queue()
                            self.clicked = False
                            self.qr_info = None

            # 一度AIモデルを読んでおく。初回読み込み時だけ時間がかかる為。
            if not f_read_model:
                dmy, dmy1 = predictor.inference(self.undistorted_image)
                f_read_model = True

            # クラスデータ初期化
            self.init_data()

            #########################################################################################
            # QR検知終了後、AI判定スタート
            if self.clicked:
                # マーカー関連
                # マーカー情報取得(補正前)
                corners0, ids0, _ = aruco.detectMarkers(
                    self.undistorted_image, aruco_dict, parameters=parameters
                )
                # マーカー情報取得(補正後)
                self.undistorted_image = cv2.undistort(
                    self.undistorted_image, self.camera_mat, self.dist_coef
                )
                corners, ids, _ = aruco.detectMarkers(
                    self.undistorted_image, aruco_dict, parameters=parameters
                )
                # 認識されたマーカーを補正前後で共通Noだけ残す
                ids0, corners0, ids, corners = self.integrity_marker_no(
                    ids0, corners0, ids, corners
                )

                # 収差補正前画像処理(カメラR,t等取得)
                self.process_distort(self.undistorted_image, corners0, ids0, ids)

                # 収差補正後画像処理
                # 座標系セット
                self.set_coordinate(corners, ids)
                # マーカー関連セット
                self.set_marker(corners, ids)  # 基底マーカーあり
                self.set_marker_no_base(corners, ids)  # 　　〃　　　なし
                # 撮影範囲関連
                self.get_shooting_range_with_base()  # 基底マーカーあり
                self.get_shooting_range_no_base(ids)  # 　　〃　　　なし

                # YOLOによる物体検出
                self.detection_by_yolo(predictor, ids, corners)

            # 描画関連
            self.draw_rectangle(ids)  ##エリア描画
            self.draw_marker_position(corners, ids)  # マーカー位置描画
            self.draw_count_by_class(ids)  ##各クラスの総カウント描画
            self.draw_valid_positions(ids)  ##確定した有効な位置描画
            # 画像表示
            self.show_image()
            #########################################################################################

            ##########################################################################################
            # キー入力の際の処理
            # OpenCVウィンドウのクローズ
            c = cv2.waitKey(1) & 0xFF
            if (
                c == 27  # ESCキー
                or cv2.getWindowProperty("undistorted", cv2.WND_PROP_VISIBLE)
                == 0  # 画面閉じる
            ):
                self.stop()  # 出力なし、システム終了
                break
            elif c == 13:  # Enter
                if self.clicked:
                    self.export_count_in_csv(ids)  # 　CSVファイル出力
                    self.stop()  # システム終了
                    break
            ##########################################################################################
        cv2.destroyAllWindows()
        return 0

    # クラスデータ初期化
    def init_data(self):
        # マーカー関連初期化
        # self.rvec_base = []        # 基底マーカー用(全て収差補正前より取得)
        # self.tvec_base = []        # 〃
        self.rvec_base = np.delete(self.rvec_base, np.s_[:])
        self.tvec_base = np.delete(self.tvec_base, np.s_[:])
        self.cam_pos_base.clear()  # 〃
        self.list_no.clear()  # 一般マーカー用リスト(全て収差補正前より取得)
        self.list_rvec.clear()  # 〃
        self.list_tvec.clear()  # 〃
        self.list_cam_pos.clear()  # 〃
        self.no_vir_base = -1  # 仮の基底マーカー用
        # self.rvec_vir = []       # 〃
        # self.tvec_vir = []       # 〃
        self.rvec_vir = np.delete(self.rvec_vir, np.s_[:])
        self.tvec_vir = np.delete(self.tvec_vir, np.s_[:])
        self.cam_pos_vir.clear()  # 〃

        # 検出モデル関連
        self.bolt_pos.clear()
        self.bolt_cls.clear()
        self.all_left_top_xy.clear()

    # キューの中身をすべて取り出してクリア
    def clear_queue(self):
        while not self.q_frames.empty():
            try:
                self.q_frames.get_nowait()  # キューから要素を取り出して捨てる
            except queue.Empty:
                break

    # スレッド化した画像の取り込み
    def frame_update(self):
        # キューへ追加
        counting_number = 0
        while not self.stop_threads:
            # 画像取り込み
            ret, frame = self.video_input.read()
            if not ret:
                self.stop_threads = True
                break
            if frame is None:
                continue
            # フレーム取り込み(5フレームに1回)
            counting_number += 1
            if self.image_height == 720 and counting_number % 5 != 0:
                continue
            elif self.image_height == 1080 and counting_number % 10 != 0:
                continue
            elif self.image_height != 720 and self.image_height != 1080:
                if counting_number % 5 != 0:
                    continue

            # キューが満杯でなければフレームを追加
            try:
                self.q_frames.put(
                    frame, block=True, timeout=1
                )  # キューにフレームを追加
            except queue.Full:
                message = "キューが満杯です。フレームを破棄します。。。"
                self.log.info(__file__, message)

        # json設定ファイルを読む

    def read_json(self):

        try:
            # JSONファイル読み込み
            with open(self.config_path, "r") as f:
                self.setting = json.load(f)

            # ArUCo Marker サイズ
            length = self.setting["marker size"]["length"]
            self.marker_length = length
            self.m_Marker.marker_length = length
            self.Cam.marker_length = length
            length *= 1000.0
            message = f"marker length : %.0f mm" % length
            self.log.info(__file__, message)

            # 座標系設定用マーカー
            axis_type = self.setting["axis"]["direction"]
            if axis_type == "x" or axis_type == "X":
                self.m_Marker.axis_dirct = 0
            else:
                self.m_Marker.axis_dirct = 1
            axis_no = self.setting["axis"]["marker no"]
            self.m_Marker.axis_no = axis_no

            # Y 下方向許容幅指定
            y = self.setting["Y range"]["Lower y"]
            self.m_Marker.lower_y = y
            message = f"initial setting lower y : %d mm" % (y * 1000)
            self.log.info(__file__, message)

            # ArUco No 指定
            f = self.setting["code specification"]["flg"]
            if f == "YES" or f == "Yes" or f == "yes":
                self.m_Marker.f_marker_no_fixed = True
                self.m_Marker.set_ini_code_no(
                    self.setting["code specification"]["code"]
                )
            else:
                self.m_Marker.f_marker_no_fixed = False

            # ボルト状況取得関連
            interval = self.setting["bolt status"]["loading interval"]
            count = self.setting["bolt status"]["status change count"]
            distance = self.setting["bolt distance"]["distance"]
            self.Time_interval = interval
            self.status_change_count = count
            self.bolt_distance = distance  # 各ボルト間の距離設定
            # 利用デバイス
            device = self.setting["device"]["use device"]
            self.device = device

        except FileNotFoundError:

            message = f"{self.config_path} not found"
            self.log.exception(__file__, message)
            raise ValueError(message)

    @staticmethod
    # 認識されたマーカーを補正前後で共通Noだけ残す
    def integrity_marker_no(ids0, corners0, ids1, corners1):
        # Input
        #   ids0        : 補正前マーカーNo
        #   corners0    : 　〃　四隅データ
        #   ids1        : 補正後マーカーNo
        #   corners1    : 　〃　四隅データ

        ids0_new = np.zeros(0)
        list0_new = []
        ids_new = np.zeros(0)
        list_new = []

        if ids0 is None or ids1 is None:
            return ids0_new, list0_new, ids_new, list_new

        # 補正前マーカー整理
        for i, no0 in enumerate(ids0):
            # 補正後画像で認識されているかマーカーかチェック
            f_exist = False
            for j, no1 in enumerate(ids1):
                if no0 == no1:
                    f_exist = True
                    break
            if not f_exist:
                continue
            # 追加
            ids0_new = np.append(ids0_new, no0)
            list0_new.append(corners0[i])

        # 補正後マーカー整理
        for i, no1 in enumerate(ids1):
            # 補正前画像で認識されているかマーカーかチェック
            f_exist = False
            for j, no0 in enumerate(ids0):
                if no1 == no0:
                    f_exist = True
                    break
            if not f_exist:
                continue
            # 追加
            ids_new = np.append(ids_new, no1)
            list_new.append(corners1[i])

        return ids0_new, list0_new, ids_new, list_new

    # 物体検出をするか否か判定
    # ArUcoマーカーが撮影されていなければ物体検出を行わない
    def check_existence_marker(self, corners, ids):
        # Input
        #   corners : マーカーのコーナー座標（画像内）
        #   ids     : マーカーNo

        # 座標系未設定の場合、NG
        if not self.m_Marker.f_axis_comp:
            return False

        # 基底マーカー関連データが存在する場合、OK
        if (
            len(self.rvec_base) != 0
            or len(self.tvec_base) != 0
            or len(self.cam_pos_base) != 0
        ):
            return True

        # 一般マーカー内で基底マーカーとのR・tが計算されているものを取得
        for i, corner in enumerate(corners):
            # マーカーが未登録の場合、無視
            if not self.m_Marker.is_exists_marker_no(ids[i]):
                continue
            # if self.m_Marker.get_status(ids[i]) != 0:  # 基底マーカーとの相互関係なしの場合、無視
            #     continue
            # # 仮想マーカーデータセット
            # if not self.set_vir_base(ids[i]):
            #     continue
            # if self.no_vir_base >= 0:
            #     break
            return True

        return False

    # 収差補正後画像処理（基底マーカーと一緒に撮影された場合）
    # 現在の撮影範囲を取得
    def get_shooting_range_with_base(self):

        # 基底マーカーが認識されていない場合、抜ける
        if (
            len(self.rvec_base) == 0
            or len(self.tvec_base) == 0
            or len(self.cam_pos_base) == 0
        ):
            return

        # 基底マーカーでCamクラスセット
        self.Cam.set_Rt(self.rvec_base, self.tvec_base)

        # 座標系未設定の場合、抜ける
        if not self.m_Marker.f_axis_comp:
            return

        # 画像の端4点
        xy1 = np.zeros((4, 2))
        xy1[0] = [0.0, self.image_height]
        xy1[1] = [self.image_width, self.image_height]
        xy1[2] = [self.image_width, 0.0]
        xy1[3] = [0.0, 0.0]

        for i, pos in enumerate(xy1):

            # 画像上の位置をマーカー座標値へ変換
            f1, v1 = self.Cam.get_vec_3D(pos, 0)
            if not f1:
                continue
            xy = self.Cam.get_crosspt_xy(self.cam_pos_base, v1)
            # 基底マーカー回転分をかける
            xy = self.m_Marker.rotate_base_arc(xy)
            # クラス変数セット
            self.shooting_rect[i] = np.copy(xy)

    # 収差補正後画像処理（基底マーカーが認識されていない(基底マーカ―基準の座標系がない場合)
    # 現在の撮影範囲を取得
    def get_shooting_range_no_base(self, ids):
        # Input
        #   ids     : マーカーNo

        # マーカーが全く認識されていない場合
        if ids is None:
            return

        # 基底マーカー関連データが存在する場合抜ける
        if (
            len(self.rvec_base) != 0
            or len(self.tvec_base) != 0
            or len(self.cam_pos_base) != 0
        ):
            return

        # 座標系未設定の場合、抜ける
        if not self.m_Marker.f_axis_comp:
            return

        # 仮想マーカー（基底マーカーとの相対関係あり）が存在しない場合、仮想マーカー（基底マーカーとの相対関係なし）を使用
        if (
            self.no_vir_base < 0
            or len(self.rvec_vir) == 0
            or len(self.tvec_vir) == 0
            or len(self.cam_pos_vir) == 0
        ):
            for i, no1 in enumerate(ids):
                if not self.m_Marker.is_exists_marker_no(no1):
                    continue
                if self.set_vir_base(no1):
                    break
        # 仮想マーカー関連データが存在しない場合、抜ける
        if (
            self.no_vir_base < 0
            or len(self.rvec_vir) == 0
            or len(self.tvec_vir) == 0
            or len(self.cam_pos_vir) == 0
        ):
            return

        # 仮想マーカーでCamクラスセット
        self.Cam.set_Rt(self.rvec_vir, self.tvec_vir)

        # 画像の端4点
        xy1 = np.zeros((4, 2))
        xy1[0] = [0.0, self.image_height]
        xy1[1] = [self.image_width, self.image_height]
        xy1[2] = [self.image_width, 0.0]
        xy1[3] = [0.0, 0.0]

        for i, pos in enumerate(xy1):  # 画像上の位置をマーカー座標値へ変換
            # f1 = False
            f1, v1 = self.Cam.get_vec_3D(pos, 0)
            if not f1:
                continue
            xy = self.Cam.get_crosspt_xy(self.cam_pos_vir, v1)
            # 仮想 → 基底マーカー座標系へ変換
            Rl, tl = self.m_Marker.get_Rt_by_no(self.no_vir_base)
            xy = np.dot(Rl, xy)
            for j, val in enumerate(tl):
                xy[j] += val
            # クラス変数セット
            self.shooting_rect[i] = np.copy(xy)

    # 最寄りのマーカーNo取得
    def get_nearest_marker_no(self, pt_2d, ids, corners):
        # Input
        #   pt_2d   : 画像内2D座標
        #   ids     : マーカーNo
        #   corners : マーカーのコーナー座標（画像内）
        # Output
        #   最寄りのマーカーNo

        # 検出されたマーカーが1つの場合
        if len(ids) == 1:
            # マーカーが未登録の場合、NG
            if not self.m_Marker.is_exists_marker_no(ids[0]):
                return False, -1
            return True, ids[0]

        corners1 = np.copy(corners)
        corners1 = np.squeeze(corners1)

        # 初期値
        no = -1
        dmin = self.image_width + self.image_height

        # 最寄りマーカー取得
        # 各マーカまでの距離取得
        for i, corn in enumerate(corners1):
            # マーカーが未登録の場合、無視
            if not self.m_Marker.is_exists_marker_no(ids[i]):
                continue
            # 収差前画像よりカメラ位置(R,t)を取得できたマーカーなのかチェック
            if not self.check_exist_in_list_no(ids[i]):
                continue
            pt1 = np.zeros(2)
            for j in range(4):
                pt1[0] += corn[j][0]
                pt1[1] += corn[j][1]
            # 平均値
            pt1[0] /= 4.0
            pt1[1] /= 4.0
            # 距離算出
            dlen = np.linalg.norm(pt_2d - pt1)
            if dlen < dmin:
                dmin = dlen
                no = ids[i]

        if no < 0:
            return False, no

        return True, no

    # 収差前画像よりカメラ位置(R,t)を取得できたマーカーなのかチェック
    def check_exist_in_list_no(self, ino):

        for i, id1 in enumerate(self.list_no):
            if ino == id1:
                return True

        return False

    # ボルト位置がマーカー内かチェック
    def check_bolt_position(self, xy):

        # ボルト位置簡易チェック
        if not self.m_Marker.f_axis_comp:
            return False
        if (
            self.m_Marker.min_x == self.m_Marker.max_x
            or self.m_Marker.min_y == self.m_Marker.max_y
        ):
            return False
        if xy[0] < self.m_Marker.min_x or self.m_Marker.max_x < xy[0]:
            return False
        if xy[1] < self.m_Marker.min_y or self.m_Marker.max_y < xy[1]:
            return False

        return True

    # 収差補正後処理
    # 基底マーカーと軸方向指定マーカーで座標系作成
    def set_coordinate(self, corners, ids):
        # Input
        #   corners : マーカーのコーナー座標（画像内）
        #   ids     : マーカーNo

        # 基底マーカーが認識されていない場合、抜ける
        if (
            len(self.rvec_base) == 0
            or len(self.tvec_base) == 0
            or len(self.cam_pos_base) == 0
        ):
            return

        if len(corners) == 0:
            return

        self.Cam.set_Rt(self.rvec_base, self.tvec_base)

        # 基底マーカーと軸方向マーカーの四隅座標値取得
        corner0 = []
        corner1 = []
        for i, corner in enumerate(corners):
            # 基底マーカーの場合
            if ids[i] == self.m_Marker.base_no:
                corner0 = corner[0]
            elif ids[i] == self.m_Marker.axis_no:
                corner1 = corner[0]

        if len(corner0) == 0 or len(corner1) == 0:
            return

        # 軸方向マーカーのXY座標算出
        crd = np.zeros([4, 3])
        f1 = False
        for i in range(4):
            # カメラ位置から2D座標へ向かうベクトル取得
            f1, v = self.Cam.get_vec_3D(corner1[i], 0)
            if not f1:
                break
            # カメラ位置と向きvで定義される直線とXY平面の交点を求める
            pt = self.Cam.get_crosspt_xy(self.cam_pos_base, v)
            crd[i] = pt
        if not f1:
            return

        # 基底マーカー登録（軸指定マーカーとの傾き考慮）
        self.m_Marker.add_marker_base(crd)

        # 軸指定マーカーの追加
        ires = self.m_Marker.add_marker(self.m_Marker.axis_no, crd)
        if ires == -1:
            return
        self.m_Marker.set_corner(self.m_Marker.axis_no, crd, 0)

    # 収差補正後画像処理（基底マーカーと一緒に撮影された場合）
    # ArUcoマーカー認識
    def set_marker(self, corners, ids):

        # Input
        #   corners : マーカーのコーナー座標（画像内）
        #   ids     : マーカーNo

        # 基底マーカーが認識されていない場合、抜ける
        if (
            len(self.rvec_base) == 0
            or len(self.tvec_base) == 0
            or len(self.cam_pos_base) == 0
        ):
            return

        # 座標系未設定の場合、抜ける
        if not self.m_Marker.f_axis_comp:
            return

        # 描画クラス
        Drw = Draw(
            self.rvec_base,
            self.tvec_base,
            self.camera_mat,
            self.dist_coef,
            self.marker_length,
        )
        Drw.image_width = self.image_width
        Drw.image_height = self.image_height
        Drw.r_base = self.m_Marker.r_base  # 基底マーカーの回転分考慮

        self.Cam.set_Rt(self.rvec_base, self.tvec_base)

        for i, corner in enumerate(corners):
            # 基底マーカーの場合
            if ids[i] == self.m_Marker.base_no and self.clicked:
                # 座標軸・原点の描画
                Drw.draw_axis(self.undistorted_image, 1)
                Drw.draw_origin(self.undistorted_image, 1)
                continue
            if ids[i] == self.m_Marker.axis_no:
                continue
            # マーカーの追加
            iret = self.m_Marker.add_marker(ids[i], corner)
            if iret == -1:  # -1:エラー
                continue
            # 一般マーカの座標値算出
            # カメラ位置から2D座標へ向かうベクトル取得
            crd = np.zeros([4, 3])
            f1 = False
            for j in range(4):
                f1, v = self.Cam.get_vec_3D(corner[0][j], 0)
                if not f1:
                    break
                pt = self.Cam.get_crosspt_xy(self.cam_pos_base, v)
                crd[j] = pt
            if not f1:
                continue
            self.m_Marker.set_corner(ids[i], crd, 0)

    # 収差補正後画像処理（基底マーカーが認識されていない(基底マーカ―基準の座標系がない場合)場合）
    # ArUcoマーカー認識
    def set_marker_no_base(self, corners, ids):

        # Input
        #   corners : マーカーのコーナー座標（画像内）
        #   ids     : マーカーNo
        # Output
        #   no_vir_base : 基底マーカーの代わりとするマーカーNo
        #   rvec_vir    : 上記マーカーの回転行列
        #   tvec_vir    : 　　〃　　　　並進行列
        #   cam_pos_vir :    〃　　　　カメラ位置

        # 座標系未設定の場合、抜ける
        if not self.m_Marker.f_axis_comp:
            return

        # 基底マーカー関連データが存在する場合抜ける
        if (
            len(self.rvec_base) != 0
            or len(self.tvec_base) != 0
            or len(self.cam_pos_base) != 0
        ):
            # print("base code exists")
            return

        # 一般マーカー内で基底マーカーとのR・tが計算されているものを取得
        for i, corner in enumerate(corners):
            # マーカーが未登録の場合、無視
            if not self.m_Marker.is_exists_marker_no(ids[i]):
                continue
            if (
                self.m_Marker.get_status(ids[i]) != 0
            ):  # 基底マーカーとの相互関係なしの場合、無視
                continue
            # 仮想マーカーデータセット
            if not self.set_vir_base(ids[i]):
                continue
            if self.no_vir_base >= 0:
                break

        # 上記で仮想マーカーが見つからなかった場合
        if (
            self.no_vir_base < 0
            or len(self.rvec_vir) == 0
            or len(self.tvec_vir) == 0
            or len(self.cam_pos_vir) == 0
        ):
            # 一般マーカー内で基底マーカーとのR・tがないものを選択
            for i, corner in enumerate(corners):
                # マーカーが未登録の場合、無視
                if not self.m_Marker.is_exists_marker_no(ids[i]):
                    continue
                # 仮想マーカーデータセット
                if not self.set_vir_base(ids[i]):
                    continue
                if self.no_vir_base >= 0:
                    break

        # 全ての方法で仮想マーカーが見つからなかった場合
        if (
            self.no_vir_base < 0
            or len(self.rvec_vir) == 0
            or len(self.tvec_vir) == 0
            or len(self.cam_pos_vir) == 0
        ):
            return

        # 未登録のマーカー登録
        # 一般マーカーと未登録マーカーのR・tを求める
        for i, corner in enumerate(corners):
            # 軸指定マーカーの場合無視
            if ids[i] == self.m_Marker.axis_no:
                continue
            # マーカーの追加
            if self.m_Marker.get_status(ids[i]) == 0:
                continue
            iret = self.m_Marker.add_marker(ids[i], corner)
            # 一般マーカの座標値算出
            if iret == 1 or iret == -1:  # 1:基底マーカー追加、-1:エラー
                continue
            # カメラ位置から2D座標へ向かうベクトル取得
            crd = np.zeros([4, 3])
            self.Cam.set_Rt(self.rvec_vir, self.tvec_vir)
            f1 = False
            for j in range(4):
                f1, v = self.Cam.get_vec_3D(corner[0][j], 0)
                if not f1:
                    break
                pt = self.Cam.get_crosspt_xy(self.cam_pos_vir, v)
                crd[j] = pt
            if not f1:
                continue
            self.m_Marker.set_corner(
                ids[i], crd, 1, self.no_vir_base
            )  # no_vir_baseとの関係性をもつ

    # 収差補正前画像処理
    def process_distort(self, frame, corners0, ids0, ids_undistorted):
        # Input
        #   corners0[]      : 収差補正前画像より検出されたマーカーコーナー
        #   ids0[]          : 〃　　　　　　　　　　　　　マーカーNo
        #   ids_distorted[] : 収差補正後画像より検出されたマーカーNo
        #

        if ids_undistorted is None or ids0 is None:
            return

        for i, corner0 in enumerate(corners0):
            # 補正後画像で認識されていない場合、無視
            f_exist = False
            for j, no1 in enumerate(ids_undistorted):
                if ids0[i] == no1:
                    f_exist = True
                    break
            if not f_exist:
                continue
            # 基底マーカーの場合
            if ids0[i] == self.m_Marker.base_no:
                # カメラパラメータ取得
                self.rvec_base, self.tvec_base = self.Cam.get_camera_Rt(
                    corner0
                )  # カメラR,t取得
                self.cam_pos_base = self.Cam.get_camera_pos(corner0)  # カメラ位置取得

                # 座標軸・原点描画
                Drw0 = Draw(
                    self.rvec_base,
                    self.tvec_base,
                    self.camera_mat,
                    self.dist_coef,
                    self.marker_length,
                )
                Drw0.image_width = self.image_width
                Drw0.image_height = self.image_height
                Drw0.r_base = self.m_Marker.r_base  # 基底マーカーの回転分考慮
                Drw0.draw_axis(frame, 0)
                Drw0.draw_origin(frame, 0)

            # 一般マーカーのカメラ外部パラメータ取得
            rvec, tvec = self.Cam.get_camera_Rt(corner0)  # カメラR,t取得
            cam_pos = self.Cam.get_camera_pos(corner0)  # カメラ位置取得
            self.list_no.append(ids0[i])
            self.list_rvec.append(rvec)
            self.list_tvec.append(tvec)
            self.list_cam_pos.append(cam_pos)

    # クラス内カメラ関連リストよりデータ取得
    def set_vir_base(self, no1):

        # 仮想マーカー関連初期化
        self.no_vir_base = -1  # 仮の基底マーカー用
        # self.rvec_vir.clear()      # 〃
        # self.tvec_vir.clear()      # 〃
        self.rvec_vir = np.delete(self.rvec_vir, np.s_[:])
        self.tvec_vir = np.delete(self.tvec_vir, np.s_[:])
        self.cam_pos_vir.clear()  # 〃

        # カメラパラメータ取得
        adr = -1
        for i, l_no in enumerate(self.list_no):
            if l_no == no1:
                adr = i
                break
        if adr < 0:
            return False
        self.rvec_vir = self.list_rvec[adr].copy()  # カメラR,t取得
        self.tvec_vir = self.list_tvec[adr].copy()
        self.cam_pos_vir = self.list_cam_pos[adr].copy()  # カメラ位置取得
        self.no_vir_base = no1
        # print("no_vir_base = {}".format(self.no_vir_base))

        if self.no_vir_base < 0:
            return False

        return True

    # QRコード読み取り
    def detect_QR_pyzbar(self, img_bgr):
        dec_inf = None
        try:
            value = decode(img_bgr, symbols=[ZBarSymbol.QRCODE])
            if value:
                for qrcode in value:
                    dec_inf = qrcode.data.decode("utf-8")
                return dec_inf  # QRコードが成功した場合
            return None  # QRコードが読み取れなかった場合
        except Exception as e:
            print(e)
            return None

    # YOLOX使用
    def set_yolo(self):
        model_path = f"models/1203_bolt_hiro_m.onnx"
        predictor = YOLOX(
            model_path,
            conf_thres=0.5,
            iou_thres=0.5,
            coco_classes=COCO_CLASSES,
        )
        predictor.make_session(self.device)

        return predictor

    # yoloによる物体検出
    def detection_by_yolo(self, predictor, ids, corners):
        # Input
        #   ids     : マーカーNo
        #   corners : マーカーのコーナー座標（画像内）
        # マーカーが全く認識されていない場合
        if ids is None:
            return

        # 座標系未設定の場合、抜ける
        if not self.m_Marker.f_axis_comp:
            return

        # 対象範囲が極端に狭い場合、無視
        x0 = self.m_Marker.min_x
        x1 = self.m_Marker.max_x
        y0 = self.m_Marker.min_y
        y1 = self.m_Marker.max_y
        if abs(x1 - x0) < self.marker_length or abs(y1 - y0) < self.marker_length:
            return

        # マーカーが撮影されているかチェック
        if not self.check_existence_marker(corners, ids):
            return

        # Onnxによる物体検出
        outputs, img_info = predictor.inference(self.undistorted_image)
        if outputs is not None:
            # 有効な検出結果はボルト位置として登録、無効な検出結果はリスト化
            n, list1 = self.selection_valid_bbox(outputs, img_info, ids, corners)

            if n > 0:
                outputs = self.delete_elm_by_list(outputs, list1)

            self.undistorted_image = predictor.visual(
                outputs,
                img_info,
                cls_conf=0.35,
            )

    # 無効な検出結果削除
    @staticmethod
    def delete_elm_by_list(outputs, list1):
        # 範囲内の削除インデックスのみを使用
        valid_del_indices = [i for i in list1 if 0 <= i < outputs.shape[0]]

        # 有効なインデックスを削除
        for del_adr in sorted(valid_del_indices, reverse=True):
            outputs = np.delete(outputs, del_adr, axis=0)

        return outputs

    # 2D座標から3D座標変換
    def process_point(self, point, ids, corners):
        # 最寄りのマーカーNo取得
        found, marker_no = self.get_nearest_marker_no(point, ids, corners)
        if not found or marker_no < 0:
            return None

        if not self.set_vir_base(marker_no):
            return None

        # 仮想マーカーでCamクラスセット
        self.Cam.set_Rt(self.rvec_vir, self.tvec_vir)

        # 画像上の位置をマーカー座標値へ変換
        found_vec, vec_3D = self.Cam.get_vec_3D(point, 0)
        if not found_vec:
            return None

        xy = self.Cam.get_crosspt_xy(self.cam_pos_vir, vec_3D)

        # 仮想 → 基底マーカー座標系へ変換
        Rl, tl = self.m_Marker.get_Rt_by_no(self.no_vir_base)
        xy = np.dot(Rl, xy)
        for j, val in enumerate(tl):
            xy[j] += val

        return xy

    # 有効な検出結果はボルト位置として登録、無効な検出結果はリスト化
    def selection_valid_bbox(self, output, img_info, ids, corners, cls_conf=0.35):
        # Input
        #   pos         : 検出結果データ
        #   img_info    : 画像データ
        #   ids         : マーカーNo
        #   corners     : マーカーのコーナー座標（画像内）
        # 返り値
        #   n                   : len(list_del)
        #   list_del  : 削除するアドレスの配列

        list_del = np.zeros(0, dtype=np.int16)

        ratio = img_info["ratio"]
        if output is None:
            return -1, list_del

        bboxes = output[:, :4]

        # # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 5]
        scores = output[:, 4]

        # 有効な検出結果は登録、無効なBBoxはリスト化
        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(cls[i])
            score = scores[i]
            if score < cls_conf:
                continue

            # 中心点の設定
            pt_2d = np.zeros(2)
            pt_2d[0] = (box[0] + box[2]) / 2.0
            pt_2d[1] = (box[1] + box[3]) / 2.0
            xy_center = self.process_point(pt_2d, ids, corners)

            # 描画固定のため
            pt_left = np.array([box[0], box[1]])
            # pt_right = np.array([box[2], box[3]])

            # left_top点の処理
            xy_l_t = self.process_point(pt_left, ids, corners)
            if xy_l_t is None:
                return
            # # right_bottom点の処理
            # xy_r_b = self.process_point(pt_right, ids, corners)
            # if xy_r_b is None:
            #     return

            ###############################################
            # BBox位置簡易チェック
            if not self.check_bolt_position(xy_center):
                # 削除対象リストへ追加
                list_del = np.append(list_del, i)
                continue

            # 描画・有効な位置フイルタのため、データ追加
            if self.check_bolt_position(xy_center):
                self.bolt_pos.append(xy_center)
                self.bolt_cls.append(cls_id)
                self.all_left_top_xy.append(xy_l_t)

            if self.OK_fixed_position(xy_center):
                # OKの位置に再判定しない、 削除対象リストへ追加
                list_del = np.append(list_del, i)
                continue

            if self.filter_position(xy_center):
                # 確定した位置、削除対象リストへ追加
                list_del = np.append(list_del, i)
                continue

            ###############################################

        # 削除対象がない場合
        if len(list_del) <= 0:
            return -1, list_del

        return len(list_del), list_del

    # マーカー位置描画
    def draw_marker_position(self, corners, ids):
        if not self.clicked:
            return
        # 描画クラス
        Drw = Draw(
            self.rvec_base,
            self.tvec_base,
            self.camera_mat,
            self.dist_coef,
            self.marker_length,
        )

        for i, corner in enumerate(corners):
            if (
                ids[i] != self.m_Marker.base_no
                and self.m_Marker.get_status(ids[i]) == -1
            ):
                continue
            # 登録されていれば座標値表示
            if self.m_Marker.is_exists_marker_no(ids[i]):
                Rl, tl = self.m_Marker.get_Rt_by_no(ids[i])
                center = np.average(corner[0], axis=0)
                crd_3d = np.zeros(2)
                crd_3d[0] = tl[0][0]
                crd_3d[1] = tl[1][0]
                # # 基底マーカーまたはバーチャル基底マーカーの場合、赤
                # if ids[i] == self.no_vir_base or ids[i] == self.m_Marker.base_no:
                #     Drw.draw_text_position(self.undistorted_image, center, crd_3d, (0, 0, 255))
                #     Drw.draw_marker_corner(self.undistorted_image, corner, (0, 0, 255))
                # # 一般マーカーの場合、緑
                # else:
                #     Drw.draw_text_position(self.undistorted_image, center, crd_3d, (0, 255, 0))
                #     Drw.draw_marker_corner(self.undistorted_image, corner, (0, 255, 0))
                # Drw.draw_text_position(self.undistorted_image, center, crd_3d, (0, 255, 0))
                Drw.draw_marker_corner(self.undistorted_image, corner, (0, 255, 0))
                continue
            # マーカー枠描画
            Drw.draw_marker_corner(self.undistorted_image, corner, (192, 192, 192))

    # 対象範囲の矩形描画
    def draw_rectangle(self, ids):
        # Input
        #   ids     : マーカーNo

        ##QRコード読み取れた時だけ描画
        if not self.clicked:
            return
        # マーカーが全く認識されていない場合
        if ids is None:
            return

        # 座標系未設定の場合、抜ける
        if not self.m_Marker.f_axis_comp:
            return

        # 対象エリア
        x0 = self.m_Marker.min_x
        x1 = self.m_Marker.max_x
        y0 = self.m_Marker.min_y
        y1 = self.m_Marker.max_y

        pt_3d = np.zeros([4, 3])
        # 左下
        pt_3d[0][0] = x0
        pt_3d[0][1] = y0
        # 右下
        pt_3d[1][0] = x1
        pt_3d[1][1] = y0
        # 右上
        pt_3d[2][0] = x1
        pt_3d[2][1] = y1
        # 左上
        pt_3d[3][0] = x0
        pt_3d[3][1] = y1

        # 分割の基準距離
        len_div = self.marker_length / 1.0

        # 各辺ごとに描画
        for i in range(4):
            # 各辺の始点・終点
            pt_st = pt_3d[i]
            if i < 3:
                pt_end = pt_3d[i + 1]
            else:
                pt_end = pt_3d[0]

            # 始点 → 終点ベクトルセット
            v = pt_end - pt_st  # ベクトル
            norm = np.linalg.norm(v)  # ベクトル長
            if norm == 0.0:
                continue
            v /= norm  # 単位ベクトル化

            # 辺を分割しながら描画
            pt1 = np.zeros([2, 3])
            pt1[0] = pt_st  # 分割した始点
            len1 = 0.0
            while len1 <= norm:
                len1 += len_div
                if len1 < norm:
                    pt1[1] = pt1[0] + v * len_div  # 分割した終点
                else:
                    pt1[1] = pt_end

                # 投影座標を算出
                f_judge = [False, False]
                pt_2d = np.zeros([2, 2])
                for j in range(2):
                    f, no1, R1, t1 = self.m_Marker.get_nearest_marker(ids, pt1[j])
                    if not f:
                        break
                    if not self.set_vir_base(no1):
                        break
                    # 描画クラスで投影座標を求める
                    Drw = Draw(
                        self.rvec_vir,
                        self.tvec_vir,
                        self.camera_mat,
                        self.dist_coef,
                        self.marker_length,
                    )
                    Drw.set_image_size(self.image_height, self.image_width)
                    Drw.set_shooting_range(self.shooting_rect)
                    f, pt_2d[j] = Drw.get_projected_2d(pt1[j], R1, t1)
                    if not f:
                        break
                    f_judge[j] = True
                if not f_judge[0] or not f_judge[1]:
                    pt1[0] = pt1[1]  # pt1[0]更新
                    continue
                # 投影2D座標の向きチェック
                if i == 0 and pt_2d[0][0] >= pt_2d[1][0]:  # 下 x比較
                    continue
                elif i == 1 and pt_2d[0][1] <= pt_2d[1][1]:  # 右　y比較
                    continue
                elif i == 2 and pt_2d[0][0] <= pt_2d[1][0]:  # 上 x比較
                    continue
                elif i == 3 and pt_2d[0][1] >= pt_2d[1][1]:  # 右　y比較
                    continue
                # 描画
                Drw = Draw(
                    self.rvec_vir,
                    self.tvec_vir,
                    self.camera_mat,
                    self.dist_coef,
                    self.marker_length,
                )
                Drw.draw_line(
                    self.undistorted_image, pt_2d[0], pt_2d[1], 1, (255, 0, 0)
                )

                # pt1[0]更新
                pt1[0] = pt1[1]

    # 3D座標から2D座標変換
    def get_project_point(self, point_3d, ids):
        # 最寄りのマーカーNo取得
        found, marker_no, R, t = self.m_Marker.get_nearest_marker(ids, point_3d)
        if not found:
            return None

        # 仮想ベース設定
        if not self.set_vir_base(marker_no):
            return None

        # 描画クラスで投影座標を求める
        Drw = Draw(
            self.rvec_vir,
            self.tvec_vir,
            self.camera_mat,
            self.dist_coef,
            self.marker_length,
        )
        Drw.set_image_size(self.image_height, self.image_width)
        Drw.set_shooting_range(self.shooting_rect)

        found, pt_2d = Drw.get_projected_2d(point_3d, R, t)
        if not found:
            return None

        return pt_2d

    # 位置が指定クラスの有効な位置に含まれるかチェック
    def is_valid_position(self, position, cls_id):
        for valid_position in self.valid_centers[cls_id]:
            if (
                np.linalg.norm(np.array(position) - np.array(valid_position))
                <= self.bolt_distance
            ):
                return True
        return False

    # 二重描画なしのため処理
    def filter_position(self, position):
        for cls_id, valid_position in self.valid_centers.items():
            for pos in valid_position:
                if (
                    np.linalg.norm(np.array(pos) - np.array(position))
                    <= self.bolt_distance
                ):
                    return True
        return False

    # 一度確定したOK位置は再判定しない
    def OK_fixed_position(self, position):
        for fixed_position in self.fixed_positions:
            if (
                np.linalg.norm(np.array(position) - np.array(fixed_position))
                <= self.bolt_distance
            ):
                return True
        return False

    # 座標を指定した精度で丸める
    @staticmethod
    def round_coordinates(center, precision=3):
        return tuple(round(c, precision) for c in center)

    # 距離チェック用のヘルパー関数
    def is_within_range(self, center1, center2):
        distance = math.sqrt(
            (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
        )
        return distance <= self.bolt_distance

    # 有効な位置確定処理
    def update_valid_positions(self):
        now_time = time.time()

        # 0.2秒以上経過していない場合はスキップ
        if now_time - self.bolt_statue_time < self.Time_interval:
            return

        # 前後フレームの結果を比較して増加しないカウントをリセット（現在のフレームは検出できていない）
        def reset_non_increasing_tracker(current, previous):
            for current_pos in list(current.keys()):
                matched = False
                for prev_pos in previous.keys():
                    if self.is_within_range(current_pos, prev_pos):
                        if current[current_pos] <= previous[prev_pos]:
                            current[current_pos] = 0  # カウントリセット
                        matched = True
                        break
                if not matched:
                    current[current_pos] = 0  # 一致なしの場合もリセット

        # カウントのリセット処理
        reset_non_increasing_tracker(
            self.status_count_tracker[1], self.previous_status_count_tracker.get(1, {})
        )  # NG
        reset_non_increasing_tracker(
            self.status_count_tracker[2], self.previous_status_count_tracker.get(2, {})
        )  # OK
        reset_non_increasing_tracker(
            self.status_count_tracker[3], self.previous_status_count_tracker.get(3, {})
        )  # PIN

        # 現在のトラッカーを過去フレームとして保存
        self.previous_status_count_tracker = {
            k: v.copy() for k, v in self.status_count_tracker.items()
        }

        # 位置をカウントして有効性をチェック
        for position, cls_id, left_top in zip(
            self.bolt_pos, self.bolt_cls, self.all_left_top_xy
        ):
            rounded_center = self.round_coordinates(position)  # 座標を丸める

            # 範囲内かをチェックしてキーを取得
            center_key = next(
                (
                    key
                    for key in self.status_count_tracker[cls_id]
                    if self.is_within_range(rounded_center, key)
                ),
                None,
            )

            # 既存のキーがなければ新しいキーとして登録、ステータスカウント初期値は0
            if center_key is None:
                center_key = rounded_center
                self.status_count_tracker.setdefault(cls_id, {}).setdefault(
                    center_key, 0
                )

            # すでにOK固定された位置ならスキップ
            if self.OK_fixed_position(center_key):
                continue

            # ステータスカウントを更新
            self.status_count_tracker[cls_id][center_key] += 1

            # ステータスカウントが閾値を超えた場合、有効位置として登録
            if (
                self.status_count_tracker[cls_id][center_key]
                >= self.status_change_count
            ):

                if not self.is_valid_position(center_key, cls_id):
                    # 　NG・PINからOKに変更の処理
                    for other_cls_id in [1, 3]:
                        if other_cls_id == cls_id:
                            continue
                        filtered_positions = [
                            (pos, box)
                            for pos, box in zip(
                                self.valid_centers[other_cls_id],
                                self.valid_boxes[other_cls_id],
                            )
                            if not self.is_within_range(center_key, pos)
                        ]
                        self.valid_centers[other_cls_id] = [
                            pos for pos, _ in filtered_positions
                        ]
                        self.valid_boxes[other_cls_id] = [
                            box for _, box in filtered_positions
                        ]

                    # 現在のクラスで同じボルトが存在しない場合にのみ追加
                    if not any(
                        self.is_within_range(center_key, pos)
                        for pos in self.valid_centers.get(cls_id, [])
                    ):
                        self.valid_centers.setdefault(cls_id, []).append(center_key)
                        self.valid_boxes.setdefault(cls_id, []).append(left_top)

                        # OKクラスの固定位置リスト追加
                        if cls_id == 2:
                            self.fixed_positions.append(center_key)

        # タイムスタンプ更新
        self.bolt_statue_time = now_time

    # 有効な位置描画
    def draw_valid_positions(self, ids):
        if ids is None:
            return

        # print(f"self.status_count_tracker NG: {self.status_count_tracker.get(1, [])}")
        # print(f"self.status_count_tracker OK: {self.status_count_tracker.get(2, [])}")
        # print(f"self.status_count_tracker PIN: {self.status_count_tracker.get(3, [])}")
        # print("=============================")
        for cls_id, positions in self.valid_boxes.items():
            for pt_left in positions:
                # pt_left, pt_right = position

                # # left_topの点を投影
                l_t_2d = self.get_project_point(pt_left, ids)
                if l_t_2d is None:
                    continue
                # # # right_bottomの点を投影
                # r_b_2d = self.get_project_point(pt_right, ids)
                # if r_b_2d is None:
                #     continue

                # 投影結果を使用して長方形を描画する
                l_t_2d = tuple(map(int, l_t_2d))  # 座標を整数に変換
                # r_b_2d = tuple(map(int, r_b_2d))  # 座標を整数に変換
                vis_counted_position(
                    self.undistorted_image, COCO_CLASSES, cls_id, l_t_2d
                )

    # クラスごとのカウント表示
    def draw_count_by_class(self, ids):
        try:
            if ids is None:
                return

            # 有効な位置更新
            self.update_valid_positions()

            # 各クラスの有効位置数をカウント
            count_ok = len(self.valid_centers.get(2, []))  # OK
            count_ng = len(self.valid_centers.get(1, []))  # NG
            count_pin = len(self.valid_centers.get(3, []))  # PIN
            self.count_display = [count_ok, count_ng, count_pin]

            if self.count_display is not None and self.clicked:
                unknown_class = int(self.qr_info["ボルト本数"]) - sum(
                    self.count_display
                )
                self.count_display.append(unknown_class)
                self.count_display.append(int(self.qr_info["ボルト本数"]))

                display_text = []
                for cls, count in zip(self.text_display, self.count_display):
                    text = f"{cls}: {count}"
                    display_text.append(text)
                result_text = ", ".join(display_text)

                font_info = cv2.getTextSize(
                    f"認識結果　{result_text}", cv2.FONT_HERSHEY_SIMPLEX, 1, 3
                )[0]

                whitespace = 5
                background_width = 2 * (self.undistorted_image.shape[1] // 3)
                background_height = font_info[1] + 10
                background_x1 = (
                    self.undistorted_image.shape[1] - background_width - whitespace
                )
                background_y1 = whitespace
                background_x2 = background_x1 + background_width
                background_y2 = background_y1 + background_height

                text_start_x = background_x1 + 10
                text_start_y = background_y1 + 5

                overlay = cv2.cvtColor(self.undistorted_image, cv2.COLOR_BGR2RGB)
                pil_overlay = Image.fromarray(overlay).convert("RGBA")
                transparent_overlay = Image.new("RGBA", pil_overlay.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(transparent_overlay)
                font_path = "Noto_Sans_JP/NotoSansJP-Bold.ttf"
                font = ImageFont.truetype(font_path, 15)
                draw.rectangle(
                    [(background_x1, background_y1), (background_x2, background_y2)],
                    fill=(255, 255, 255, 255),
                    outline=(0, 0, 0),
                    width=2,
                )
                draw.text(
                    (text_start_x, text_start_y),
                    f"認識結果　{result_text}",
                    font=font,
                    fill=(0, 0, 0),
                )
                combined = Image.alpha_composite(pil_overlay, transparent_overlay)
                self.undistorted_image = cv2.cvtColor(
                    np.array(combined), cv2.COLOR_RGBA2BGR
                )
        except Exception as e:
            print(e)

    # 画像を表示する
    def show_image(self):

        # Undistorted 画像表示
        cv2.namedWindow("undistorted", 0)
        cv2.resizeWindow("undistorted", 1280, 720)  # ウィンドウズリサイズ
        cv2.imshow("undistorted", self.undistorted_image)

    # CSVファイル出力
    def export_count_in_csv(self, ids):
        try:
            # マーカーが全く認識されていない場合
            if ids is None:
                return
            result_path = f"results"
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            now = datetime.now()
            output_path = (
                result_path
                + f"/添接番号_{self.qr_info['添接番号']}_{now.strftime('%Y_%m_%d_%H_%M_%S')}.csv"
            )

            # カラム名
            columns = self.text_display

            if self.count_display is not None and self.clicked:
                # CSVファイルへの出力
                output_file = output_path
                with open(
                    output_file, "w", encoding="utf-8-sig", newline=""
                ) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(columns)
                    writer.writerow(self.count_display)

        except Exception as e:
            print(e)
            return None

    # スレッドを終了させる処理
    def stop(self):
        self.stop_threads = True  # スレッド終了フラグOn
        self.thread.join(timeout=2)  # スレッド終了待ち
        self.log.info(__file__, "normal end")

    if __name__ == "__main__":
        main()
