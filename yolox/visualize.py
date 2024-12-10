#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np
from logging import getLogger

logger = getLogger()
__all__ = ["vis"]


def vis(
    img,
    boxes,
    scores,
    cls_ids,
    inference_target_dict,
    conf=0.5,
    class_names=None,
):
    try:
        # 検出対象クラスIDと名前set取得
        inference_target_name_set = tuple(inference_target_dict.keys())

        for i in range(len(boxes)):
            box = boxes[i]
            class_id = int(cls_ids[i])
            class_name = class_names[class_id]
            score = scores[i]
            if score < conf:
                continue
            # 検出対象クラスが検出されたかどうか
            if inference_target_dict and class_name in inference_target_name_set:
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                # bbox描画
                bbox_color = (_COLORS[class_id] * 255).astype(np.uint8).tolist()

                text = "{}".format(class_names[class_id])
                txt_color = (0, 0, 0)
                # txt_color = (
                #     (0, 0, 0) if np.mean(_COLORS[class_id]) > 0.5 else (255, 255, 255)
                # )
                font = cv2.FONT_HERSHEY_SIMPLEX

                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

                txt_bk_color = (_COLORS[class_id] * 255 * 0.7).astype(np.uint8).tolist()
                cv2.rectangle(img, (x0, y0), (x1, y1), bbox_color, 1)
                cv2.rectangle(
                    img,
                    (x0, y0 + 1),
                    (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                    txt_bk_color,
                    -1,
                )
                cv2.putText(
                    img,
                    text,
                    (x0, y0 + txt_size[1]),
                    font,
                    0.4,
                    txt_color,
                    thickness=1,
                )
        return img
    except IndexError:
        logger.error(
            f"BBox情報描画Indexエラー:使用しているモデルとcoco_classes.pyの内容に相違がある可能性があります。"
        )
    except Exception as e:
        logger.error(f"BBox情報描画エラー:{str(e)}")


def vis_counted_position(img, class_names, class_id, pt1):
    try:
        text = "{}".format(class_names[class_id])
        txt_color = (
            255,
            255,
            255,
        )  #  (_COLORS[class_id] * 255).astype(np.uint8).tolist()
        # bbox_color = (_COLORS[class_id] * 255).astype(np.uint8).tolist()
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]

        txt_bk_color = (_COLORS[class_id] * 255 * 0.7).astype(np.uint8).tolist()
        # cv2.rectangle(img, pt1, pt2, bbox_color, 2)
        cv2.rectangle(
            img,
            (pt1[0], pt1[1] + 1),
            (pt1[0] + txt_size[0] + 1, pt1[1] + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (pt1[0], pt1[1] + txt_size[1]),
            font,
            0.4,
            txt_color,
        )
    except IndexError:
        logger.error(
            f"BBox情報描画Indexエラー:使用しているモデルとcoco_classes.pyの内容に相違がある可能性があります。"
        )
    except Exception as e:
        logger.error(f"BBox情報描画エラー:{str(e)}")


_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.000,
            0.000,
            1.000,  # NG (red)
            1.000,
            0.000,
            0.000,  # OK (blue)
            0.000,
            1.000,
            1.000,  # PIN (yellow)
            0.753,
            0.753,
            0.753,  # silver
            1.000,
            1.000,
            1.000,  # white
            0.000,
            1.000,
            1.000,  # yellow
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
