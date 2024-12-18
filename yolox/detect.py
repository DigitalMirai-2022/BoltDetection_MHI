import numpy as np
import cv2
import onnxruntime
from yolox.coco_classes import COCO_CLASSES
from yolox.visualize import vis

class_names = COCO_CLASSES

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


class YOLOX:
    def __init__(self, model_path, conf_thres=0.7, iou_thres=0.5, coco_classes=None):
        self.common_process = CommonProcess()
        self.session = None
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.model_path = model_path
        self.coco_classes = coco_classes

    def make_session(self, device):
        if device == "gpu":
            self.session = onnxruntime.InferenceSession(
                self.model_path, providers=["CUDAExecutionProvider"]
            )
        else:
            self.session = onnxruntime.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )

        return self.session

    def inference(self, image):
        img_info = {"id": 0}
        img_info["raw_img"] = image
        input_shape = self.session.get_inputs()[0].shape
        ratio = min(input_shape[2] / image.shape[0], input_shape[3] / image.shape[1])
        img_info["ratio"] = ratio
        result = self.detect(
            session=self.session,
            frame=image,
            frame_size=(input_shape[2], input_shape[3]),
            score_thr=self.conf_threshold,
        )

        return result, img_info

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr, class_agnostic=True):
        """Multiclass NMS implemented in Numpy"""
        if class_agnostic:
            nms_method = self.multiclass_nms_class_agnostic
        else:
            nms_method = self.multiclass_nms_class_aware
        return nms_method(boxes, scores, nms_thr, score_thr)

    def multiclass_nms_class_aware(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-aware version."""
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.common_process.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def multiclass_nms_class_agnostic(self, boxes, scores, nms_thr, score_thr):
        """Multiclass NMS implemented in Numpy. Class-agnostic version."""
        dets = None

        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self.common_process.nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [
                    valid_boxes[keep],
                    valid_scores[keep, None],
                    valid_cls_inds[keep, None],
                ],
                1,
            )

        return dets

    def demo_postprocess(self, outputs, img_size, p6=False):

        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def preprocess(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = (
                np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
            )
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def detect(self, session, frame, frame_size, score_thr):
        input_shape = (frame_size[0], frame_size[1])
        img, ratio = self.preprocess(frame, input_shape)

        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}

        output = session.run(None, ort_inputs)

        predictions = self.demo_postprocess(output[0], input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=score_thr
        )

        return dets

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def visual(
        self,
        output,
        img_info,
        cls_conf=0.35,
    ):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img

        bboxes = output[:, :4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 5]
        scores = output[:, 4]
        inference_target_dict = {
            item: index for index, item in enumerate(COCO_CLASSES) if index != 0
        }

        vis_res = vis(
            img, bboxes, scores, cls, inference_target_dict, cls_conf, class_names
        )
        return vis_res


class CommonProcess:
    def __init__(self):
        pass

    def nms(self, boxes, scores, nms_thr):
        """Single class NMS implemented in Numpy."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep
