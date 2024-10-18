import cv2
from coco_classes import COCO_CLASSES


def vis_class_count(image, selected_classes, count_list, config_total):
    """
    各クラス名と検出数を描画
    """
    total_detected = sum(count_list)
    unknown_class = config_total - total_detected
    text_list = [COCO_CLASSES[i] for i in selected_classes]
    text_list.append("Unknown")
    text_list.append("total")
    count_list.append(unknown_class)
    count_list.append(config_total)

    # 表示テキストリファクタリング
    display_text = []
    for cls, count in zip(text_list, count_list):
        text = f"{cls}: {count}"
        display_text.append(text)
    result_text = ", ".join(display_text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(result_text, font, 0.5, 1)[0]
    text_position = image.shape[1]

    # 描画
    cv2.rectangle(
        image,
        (int(text_position - text_size[0] - 15), text_size[1] - 5),
        (int(text_position - 5), int(text_size[1] * 2 + 5)),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        img=image,
        text=str(result_text),
        org=(
            int(text_position - text_size[0] - 10),
            10 + text_size[1],
        ),
        fontFace=font,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return image
