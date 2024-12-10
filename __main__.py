import sys
import numpy as np
import cv2
from sys import exit

# 自作
from Calibration import Calibration

from Measure import Measure

from Output_Log import LogMan


def main():

    log = LogMan.get_instance()  # logインスタンスの生成

    # 動画取得
    video_input = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 実用時はカメラ名チェックが必要
    if not video_input.isOpened():
        log.error(__file__, "camera not open")
        exit()

    fps1 = video_input.get(cv2.CAP_PROP_FPS)
    log.info(__file__, f"FPS =  {fps1}")
    width1 = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(__file__, f"Original Size =  ({width1} x {height1})")

    ret = True
    camera_mat, dist_coef = [], []
    if not ret:  # チェスボードの撮影
        calib = Calibration(video_input, camera_mat, dist_coef)
        iret = calib.main()
        if iret != 0:
            video_input.release()
            exit(-1)
        camera_mat = calib.camera_mat
        dist_coef = calib.dist_coef
        dist_coef = np.squeeze(dist_coef)

    else:  # キャリブレーションデータの読み込み
        log.debug(__file__, f"Load calibration file for camera")
        camera_mat = np.loadtxt("K.csv", delimiter=",")
        dist_coef = np.loadtxt("d.csv", delimiter=",")

    # カメラ内部パラメータ出力
    message = (
        f"K = [{camera_mat[0][0]}, {camera_mat[0][1]}, {camera_mat[0][2]}]"
        f" [{camera_mat[1][0]}, {camera_mat[1][1]}, {camera_mat[1][2]}]"
        f"[{camera_mat[2][0]}, {camera_mat[2][1]}, {camera_mat[2][2]}]"
    )
    log.debug(__file__, message)
    message = f"d = {dist_coef[0]}, {dist_coef[1]}, {dist_coef[2]} , {dist_coef[3]}, {dist_coef[4]}"
    log.debug(__file__, message)

    # 内部パラメータがないとき
    if camera_mat.size != 9 or dist_coef.size != 5:
        video_input.release()
        exit(-1)

    # 撮影開始
    Mea = Measure(video_input, camera_mat, dist_coef, width1, height1)
    iret = Mea.main()

    video_input.release()

    return iret


if __name__ == "__main__":
    sys.exit(main())
