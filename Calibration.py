import numpy as np
import cv2
from tkinter import messagebox
import threading
import queue

# 自作
from Output_Log import LogMan


class Calibration:

    def __init__(self, video_input, camera_mat, dist_coef):
        self.square_side_length = 20.0  # チェスボード内の正方形の1辺のサイズ(mm)
        # self.grid_intersection_size = (18, 9)  # チェスボード内の格子数
        self.grid_intersection_size = (10, 7)  # チェスボード内の格子数
        self.video_input = video_input
        self.camera_mat = camera_mat
        self.dist_coef = dist_coef

        self.log = LogMan.get_instance()  # logインスタンスの生成
        # スレッド
        self.thread = threading.Thread(target=self.frame_update, daemon=True)
        # フレーム共用のキュー
        self.q_frames = queue.Queue(maxsize=5)

    # def __del__(self):
    # スレッド化した画像の取り込み
    def frame_update(self):

        # キューへ追加
        counting_number = 0
        while True:
            # 画像取り込み
            ret, frame = self.video_input.read()
            if not ret:
                break
            if frame is None:
                continue
            # フレーム取り込み(5フレームに1回)
            counting_number += 1
            if counting_number % 10 == 0:
                # 取得した画像をキューに追加
                self.q_frames.put(frame, block=True, timeout=None)
                print(f"qsize : {self.q_frames.qsize()}")

    def main(self):

        # 配列初期化
        pattern_points = np.zeros((np.prod(self.grid_intersection_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.grid_intersection_size).T.reshape(-1, 2)
        pattern_points *= self.square_side_length  # チェスボード格子サイズ[17,7,1]

        object_points = []
        image_points = []

        # OpenCV画像取得用スレッドの開始
        self.thread.start()

        capture_count = 0
        while True:
            # ret, frame = self.video_input.read()
            # if not ret:
            #     break

            # キューより画像取得
            # queueが空の場合は無視
            if self.q_frames.empty():
                continue
            # キューより画像取得
            frame = self.q_frames.get()

            # チェスボードのコーナーを検出
            found, corner = cv2.findChessboardCorners(
                frame, self.grid_intersection_size
            )

            if found:
                print("findChessboardCorners : True")
                cv2.drawChessboardCorners(
                    frame, self.grid_intersection_size, corner, found
                )
            if not found:
                print("findChessboardCorners : False")

            cv2.putText(
                frame,
                "Enter:Capture Chessboard(" + str(capture_count) + ")",
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                frame,
                "N    :Completes Calibration Photographing",
                (100, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                frame,
                "ESC  :terminate program",
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )
            cv2.imshow("original", frame)

            c = cv2.waitKey(50) & 0xFF
            if c == 13 and found is True:  # Enter
                # チェスボードコーナー検出情報を追加
                image_points.append(corner)
                object_points.append(pattern_points)
                capture_count += 1
            if c == 110:  # N
                if messagebox.askyesno(
                    "askyesno",
                    "チェスボード撮影を終了し、カメラ内部パラメータを求めますか？",
                ):
                    cv2.destroyAllWindows()
                    break
            if c == 27:  # ESC
                if messagebox.askyesno("askyesno", "プログラムを終了しますか？"):
                    self.video_input.release()
                    cv2.destroyAllWindows()
                    return -1

        if len(image_points) > 0:
            # カメラ内部パラメータを計算
            self.log.debug(__file__, "calibrateCamera() start")
            rms, K, d, r, t = cv2.calibrateCamera(
                object_points,
                image_points,
                (frame.shape[1], frame.shape[0]),
                None,
                None,
            )
            message = f"RMS = {rms} "
            self.log.debug(__file__, message)
            message = (
                f"K = [{K[0][0]}, {K[0][1]}, {K[0][2]}]"
                f" [{K[1][0]}, {K[1][1]}, {K[1][2]}]"
                f"[{K[2][0]}, {K[2][1]}, {K[2][2]}]"
            )
            self.log.debug(__file__, message)
            message = f"d = {d[0][0]}, {d[0][1]}, {d[0][2]}, {d[0][3]}, {d[0][4]}"
            self.log.debug(__file__, message)
            np.savetxt("K.csv", K, delimiter=",", fmt="%0.14f")  # カメラ行列の保存
            np.savetxt("d.csv", d, delimiter=",", fmt="%0.14f")  # 歪み係数の保存

            self.camera_mat = K
            self.dist_coef = d

            # 再投影誤差による評価
            mean_error = 0
            for i in range(len(object_points)):
                self.log.debug(__file__, f"{i}")
                image_points2, _ = cv2.projectPoints(
                    object_points[i], r[i], t[i], self.camera_mat, self.dist_coef
                )
                error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(
                    image_points2
                )
                mean_error += error
            message = f"total error:  {mean_error / len(object_points)} "  # 0に近い値が望ましい(魚眼レンズの評価には不適？)
            self.log.debug(__file__, message)
        else:
            self.log.debug(__file__, "findChessboardCorners() not be successful once")

        cv2.destroyAllWindows()

        # if self.camera_mat == [] or self.dist_coef == [] or capture_count <= 0:
        #     return -1

        return 0

    if __name__ == "__main__":
        main()
