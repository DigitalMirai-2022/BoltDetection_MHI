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
from visualize import vis_class_count, vis_waiting_prepare
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

import time


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
        t1 = time.time()
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
        t2 = time.time()
        print(f"time prepare {t2-t1} s")
        print(f"config data {self.config}")
        print("#######################################")

        self.change = False
        self.inferenced_frame = None  # To store the AI inferenced frame

        self.window.after(5000, self.start_camera)
        self.update()

        self.window.bind("<Return>", self.export_empty_csv)
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        self.window.mainloop()

    def resize(self, event):
        """ウィンドウサイズに合わせてCanvasサイズを更新"""
        self.canvas.config(width=event.width, height=event.height)

    def start_camera(self):
        ret, frame = self.vid.read()
        if ret:
            self.qr_info = self.function_qrdec_pyzbar(frame)
            if self.qr_info is None:
                self.window.after(5000, self.start_camera)
            else:
                self.change = True
                # Start AI processing in a new thread
                threading.Thread(target=self.AI_test, daemon=True).start()
        else:
            messagebox.showerror("エラー", "カメラが検出出来ませんでした。")
            self.quit()

    def update(self):
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
                        # display_frame = resized_frame
                        display_frame = np.zeros(
                            (self.canvas.winfo_height(), self.canvas.winfo_width(), 3),
                            dtype=np.uint8,
                        )
                        display_frame = vis_waiting_prepare(display_frame)
                    else:
                        display_frame = resized_frame
                elif self.inferenced_frame is not None and self.change:
                    display_frame = self.inferenced_frame

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
                        final_boxes, final_scores, final_cls_inds = (
                            inference_result[:, :4],
                            inference_result[:, 4],
                            inference_result[:, 5],
                        )
                        output = final_boxes, final_scores, final_cls_inds
                        detections = sv.Detections.from_yolox_onnx(output)
                        detections = detections[
                            np.isin(detections.class_id, inference_target_ids)
                        ]
                        count_list = []
                        for zone, zone_annotator, box_annotator in zip(
                            self.zones, self.zone_annotators, self.box_annotators
                        ):
                            detections_in_zone = detections[
                                zone.trigger(detections=detections)
                            ]

                            # エリア、バウンディングボックス描画
                            frame = zone_annotator.annotate(scene=frame)
                            frame = box_annotator.annotate(
                                scene=frame, detections=detections_in_zone
                            )

                            # クラス毎に検出結果フイルタ
                            for cls_id in inference_target_ids:
                                detection_filter = detections_in_zone[
                                    np.isin(detections_in_zone.class_id, cls_id)
                                ]
                                count_list.append(len(detection_filter))

                            frame = vis_class_count(
                                frame,
                                inference_target_ids,
                                count_list,
                                self.config.bolt_status.status_change_count,
                            )
                            print(f"count_list {count_list}")
                            print("===============================================")

                    # ウィンドウサイズに合わせてフレームをリサイズ
                    frame = cv2.resize(
                        frame, (self.canvas.winfo_width(), self.canvas.winfo_height())
                    )
                    self.inferenced_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                except Exception as e:
                    logger.error(f"最新フレーム取得エラー:{str(e)}")
                    raise ValueError("最新フレーム取得エラー")

    def init_model(self, model_path):
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

    # def display_ai_preparation(self):
    #     """Display AI preparation message"""
    #     # self.canvas.delete("all")
    #     rect_x1 = self.canvas.winfo_width() // 2 - 120
    #     rect_y1 = self.canvas.winfo_height() // 2 - 20
    #     rect_x2 = self.canvas.winfo_width() // 2 + 120
    #     rect_y2 = self.canvas.winfo_height() // 2 + 20
    #     self.canvas.create_rectangle(
    #         rect_x1, rect_y1, rect_x2, rect_y2, fill="white", outline=""
    #     )
    #     self.canvas.create_text(
    #         self.canvas.winfo_width() // 2,
    #         self.canvas.winfo_height() // 2,
    #         text="AI判定の準備中。。。",
    #         fill="red",
    #         font=("Helvetica", 16),
    #     )
    #     self.canvas.update_idletasks()

    def function_qrdec_pyzbar(self, img_bgr):
        dec_inf = None
        try:
            value = decode(img_bgr, symbols=[ZBarSymbol.QRCODE])
            if value:
                for qrcode in value:
                    dec_inf = qrcode.data.decode("utf-8")
                messagebox.showinfo("QR情報", dec_inf)
            else:
                messagebox.showerror(
                    "失敗", "QRコード読み取り失敗しました。再度読み取ります。"
                )
            return dec_inf
        except Exception as e:
            logger.error(f"QRコードのデコード中にエラーが発生しました: {e}")
            return None

    def export_empty_csv(self, event):
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
        self.vid.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root, "カメラ映像表示", "model/yolox_x.onnx", "config.json")
