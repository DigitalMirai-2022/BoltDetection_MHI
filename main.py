import supervision as sv
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
import tkinter as tk
from tkinter import messagebox
from logging import getLogger
import cv2
from PIL import Image, ImageTk
import json
import csv
import os
from datetime import datetime
from detect import YOLOX
from coco_classes import COCO_CLASSES
from visualize import vis_class_count
from interface import (
    MarkerSize,
    Axis,
    YRange,
    CodeSpecification,
    BoltStatus,
    Config_Data,
)
import threading

inference_target_ids = [0, 1, 2]
logger = getLogger()


class CameraApp:
    def __init__(
        self,
        window,
        window_title,
        model_path,
        config_json_path,
        video_source=0,
        color=sv.ColorPalette.DEFAULT,
    ):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # 初期のウィンドウサイズを設定
        self.window.geometry("800x600")

        # カメラキャプチャ
        self.vid = cv2.VideoCapture(self.video_source)

        # コンフィグデータ取得
        self.config = self.init_input(config_json_path)

        self.canvas = tk.Canvas(window, bg="orange")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 検出エリア
        polygons = [np.array([[10, 10], [600, 10], [600, 400], [10, 400]], np.int32)]

        # ポリゴンの配列取得し、ゾーンに書き込む
        self.zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
        # 各ゾーンのエリアを囲むため
        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=color.by_idx(index),
                thickness=1,
                text_thickness=1,
                text_scale=1,
            )
            for index, zone in enumerate(self.zones)
        ]
        # 各ゾーンにボックスを囲むため
        self.box_annotators = [
            sv.BoxAnnotator(
                color=color.by_idx(index),
            )
            for index in range(len(polygons))
        ]

        self.window.bind("<Configure>", self.resize)

        # YOLOXモデル
        self.model = self.init_model(model_path)

        self.change = False  # AI判定に移るためのフラグ
        self.inferenced_frame = None  # AI判定の結果初期化

        self.window.after(100, self.start_camera)  # カメラ表示0.1秒後、QRコード検知開始
        self.update()

        self.window.bind("<Return>", self.export_empty_csv)
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.window.mainloop()

    def resize(self, event):
        """ウィンドウサイズに合わせてCanvasサイズを更新"""
        self.canvas.config(width=event.width, height=event.height)

    def start_camera(self):
        """カメラ映像表示"""
        ret, frame = self.vid.read()
        if ret:
            # QRコードを読み取る
            self.qr_info = self.function_qrdec_pyzbar(frame)
            if self.qr_info is None:
                # QRコードが読み取れない場合、再度読み取り
                self.window.after(100, self.start_camera)
            else:
                # QRコードが読み取れた場合にOK/キャンセルダイアログを表示
                dialog_result = messagebox.askokcancel("QR情報", self.qr_info)
                if dialog_result:  # OKが押された場合
                    self.change = True
                    threading.Thread(
                        target=self.AI_test, daemon=True
                    ).start()  # 別のスレッドでAI判定開始
                else:  # キャンセルが押された場合
                    self.window.after(100, self.start_camera)  # QRコードを再度読み取る
        else:
            messagebox.showerror("エラー", "カメラが検出出来ませんでした。")
            self.quit()

    def update(self):
        """映像カメラ表示継続"""
        ret, frame = self.vid.read()
        if ret:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # ウィンドウサイズに合わせてフレームをリサイズ
                resized_frame = cv2.resize(
                    frame, (self.canvas.winfo_width(), self.canvas.winfo_height())
                )
                if self.inferenced_frame is None:
                    if self.change:
                        display_frame = resized_frame
                        # 非クリック型のダイアログ表示
                        self.show_waiting_dialog()
                    else:
                        display_frame = resized_frame

                elif self.inferenced_frame is not None and self.change:
                    display_frame = self.inferenced_frame
                    # AI処理が開始されたらダイアログを閉じる
                    self.window.after(0, self.close_waiting_dialog)

                self.image = ImageTk.PhotoImage(image=Image.fromarray(display_frame))
                self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)

            except Exception as e:
                logger.error(f"最新フレーム取得エラー:{str(e)}")
                raise ValueError("最新フレーム取得エラー")
        else:
            messagebox.showerror("エラー", "カメラが検出出来ませんでした。")
            exit()

        self.window.after(10, self.update)

    def AI_test(self):
        """AI判定"""
        while self.change:
            ret, frame = self.vid.read()
            if ret:
                try:
                    # 推論
                    inference_result = self.model.inference(frame)

                    if inference_result is not None:
                        output = (
                            inference_result[:, :4],
                            inference_result[:, 4],
                            inference_result[:, 5],
                        )
                        detections = sv.Detections.from_yolox_onnx(output)
                        detections = detections[
                            np.isin(detections.class_id, inference_target_ids)
                        ]
                        count_list = []
                        for zone, box_annotator in zip(self.zones, self.box_annotators):
                            detections_in_zone = detections[
                                zone.trigger(detections=detections)
                            ]
                            # バウンディングボックス描画
                            frame = box_annotator.annotate(
                                scene=frame, detections=detections_in_zone
                            )

                            # クラス毎に検出結果フイルタ
                            for cls_id in inference_target_ids:
                                detection_filter = detections_in_zone[
                                    np.isin(detections_in_zone.class_id, cls_id)
                                ]
                                count_list.append(len(detection_filter))

                    # エリア描画
                    for zone in self.zone_annotators:
                        frame = zone.annotate(scene=frame)
                    frame = vis_class_count(
                        frame,
                        inference_target_ids,
                        count_list,
                        self.config.bolt_status.status_change_count,
                    )

                    # ウィンドウサイズに合わせてフレームをリサイズ
                    frame = cv2.resize(
                        frame, (self.canvas.winfo_width(), self.canvas.winfo_height())
                    )
                    self.inferenced_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                except Exception as e:
                    logger.error(f"最新フレーム取得エラー:{str(e)}")
                    self.change = False
                    raise ValueError("最新フレーム取得エラー")
            else:
                logger.error("エラー", "カメラが検出出来ませんでした。")
                self.quit()
                break

    def init_model(self, model_path):
        """モデルロード"""
        model = None
        if model_path is not None:
            model = YOLOX(
                model_path,
                conf_thres=0.5,
                iou_thres=0.5,
                coco_classes=COCO_CLASSES,
            )
            model.make_session("gpu")
        return model

    def init_input(self, config_json_path):
        """設定ファイルからデータ取得"""
        config_data = None
        try:
            with open(config_json_path, "r") as f:
                data_dict = json.load(f)

            marker_size = MarkerSize(length=data_dict["marker size"]["length"])
            axis = Axis(
                direction=data_dict["axis"]["direction"],
                marker_no=data_dict["axis"]["marker no"],
            )
            y_range = YRange(lower_y=data_dict["y range"]["Lower y"])
            code_specification = CodeSpecification(
                flg=data_dict["code specification"]["flg"],
                code=data_dict["code specification"]["code"],
            )
            bolt_status = BoltStatus(
                loading_interval=data_dict["bolt status"]["loading interval"],
                status_change_count=data_dict["bolt status"]["status change count"],
            )

            config_data = Config_Data(
                marker_size=marker_size,
                axis=axis,
                y_range=y_range,
                code_specification=code_specification,
                bolt_status=bolt_status,
            )
            return config_data
        except json.decoder.JSONDecodeError:
            print(f"設定ファイルの値が間違っている可能性があります。")
            raise

    def function_qrdec_pyzbar(self, img_bgr):
        """QRコード読み取り"""
        dec_inf = None
        try:
            value = decode(img_bgr, symbols=[ZBarSymbol.QRCODE])
            if value:
                for qrcode in value:
                    dec_inf = qrcode.data.decode("utf-8")
                return dec_inf  # QRコードが成功した場合
            return None  # QRコードが読み取れなかった場合
        except Exception as e:
            logger.error(f"QRコードのデコード中にエラーが発生しました: {e}")
            return None

    def show_waiting_dialog(self):
        """AI処理待ちのダイアログを表示"""
        if not hasattr(self, "waiting_dialog") or self.waiting_dialog is None:
            # ダイアログがまだ存在しない場合のみ表示
            self.waiting_dialog = tk.Toplevel(self.window)
            self.waiting_dialog.title("AI処理待機中")
            tk.Label(self.waiting_dialog, text="AI処理を準備中です...").pack(
                padx=20, pady=20
            )

            # メインウィンドウの位置とサイズを取得
            self.window.update_idletasks()  # 必要に応じて更新
            main_window_x = self.window.winfo_x()
            main_window_y = self.window.winfo_y()
            main_window_width = self.window.winfo_width()

            # ダイアログの位置をメインウィンドウの右隣に設定
            dialog_x = main_window_x + main_window_width + 10  # 右隣に10pxの間隔
            dialog_y = main_window_y
            self.waiting_dialog.geometry(f"+{dialog_x}+{dialog_y}")

            # ダイアログを閉じられないように設定
            self.waiting_dialog.protocol("WM_DELETE_WINDOW", lambda: None)
            self.waiting_dialog.transient(self.window)  # メインウィンドウと関連づける
            self.waiting_dialog.grab_set()  # モーダルにする

    def close_waiting_dialog(self):
        """AI処理開始時にダイアログを自動的に閉じる"""
        if hasattr(self, "waiting_dialog") and self.waiting_dialog is not None:
            self.waiting_dialog.destroy()
            self.waiting_dialog = None

    def export_empty_csv(self, event):
        """CSVファイル出力"""
        try:
            result_path = "result"
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            now = datetime.now()
            output_path = result_path + f"/{now.strftime('%Y-%m-%d_%H_%M_%S')}.csv"
            with open(output_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([])  # 空の行を書き込む
            self.quit()
        except Exception as e:
            logger.error(f"CSVファイルの出力中にエラーが発生しました: {e}")

    def quit(self):
        """システム終了"""
        self.vid.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "カメラ映像表示", "model/yolox_x.onnx", "config.json")
