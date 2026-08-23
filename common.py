from __future__ import annotations

import numpy as np
import cv2

IMG_SIZE = 640

# ค่าเทาของ ultralytics — ต้องเป็น 114 เป๊ะ ไม่ใช่ 0 หรือ 128
# โมเดลเห็นขอบสีนี้ตอนเทรน เปลี่ยนค่าแล้ว mAP ขยับโดยไม่มีอะไรฟ้อง
# ตรวจแล้วบน cv2 5.0.0: letterbox() ให้ผลเท่ากับ ultralytics
# LetterBox(auto=False, scaleup=False) ทุก byte ครบ 8 อัตราส่วนที่ลอง
PAD_VALUE = 114

# สองชุดล่างนี้ห้ามสลับกัน
#
# VAL_* ใช้ตอนวัด mAP — conf 0.001 ต่ำจนดูไร้เหตุผล แต่ COCO AP คิดจากพื้นที่ใต้
# PR curve ต้องมีหางคะแนนต่ำมาลาก curve ไปจนสุด ถ้าตั้ง 0.25 mAP จะตกไปหลายจุด
# เพราะ recall ไปไม่ถึงปลาย และไม่มี error อะไรบอก
VAL_CONF = 0.001
VAL_IOU = 0.7
VAL_MAX_DET = 300

# DEPLOY_* คือค่าที่จะใช้จริง ใช้ตอนวัด latency เพราะ postprocess ถูกนับใน post_ms
# ถ้าวัด latency ด้วย conf 0.001 จะได้ NMS ที่ต้องกรอง box หลักพันทุกเฟรม
# ซึ่งไม่ใช่ภาระที่ระบบจริงเจอ ตัวเลขจะบวมโดยไม่มีความหมาย
#
# ผลคือคอลัมน์ mAP กับคอลัมน์ latency ในตารางมาจากคนละ threshold โดยตั้งใจ
# ต้องเขียนกำกับใน Analysis ไม่งั้นดูเหมือนวัดไม่สอดคล้องกัน
DEPLOY_CONF = 0.25
DEPLOY_IOU = 0.45


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------
def letterbox(img: np.ndarray, new_shape: int = IMG_SIZE, scaleup: bool = False):
    # scaleup=False ตาม ultralytics val — ภาพที่เล็กกว่า 640 อยู่แล้วไม่ขยาย
    # ขยายไม่ได้เพิ่มรายละเอียด มีแต่ทำให้ผลต่างจาก val ที่ใช้อ้างอิง
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # ±0.1 ไม่ใช่เลขมั่ว — dw/dh เป็นครึ่งของส่วนที่ขาด เลยลงท้าย .5 ได้
    # round(x-0.1) กับ round(x+0.1) แบ่ง pad คี่เป็น n กับ n+1 ผลรวมจึงเท่าส่วนที่ขาดพอดี
    # ใช้ int() ทั้งคู่จะขาด 1 px แล้ว TRT ฟ้อง shape ไม่ตรง profile
    # ใช้ round() ทั้งคู่จะเกิน 1 px
    # ลอกสูตรจาก ultralytics มาตรงๆ เพื่อให้ bbox mapping ตรงกัน ไม่ได้คิดเอง
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return img, r, (left, top)


def preprocess(bgr: np.ndarray, size: int = IMG_SIZE):
    """BGR uint8 HWC  ->  RGB float32 NCHW [0,1] (contiguous)

    ascontiguousarray ไม่ใช่การเผื่อไว้เฉยๆ — TensorRT copy ดิบจาก arr.ctypes.data
    ตาม nbytes (benchmark.py:203) ไม่ได้ดู stride ส่วน transpose() คืน view ที่
    stride ไม่เรียง ถ้าส่งเข้าไปตรงๆ GPU จะได้ขยะ แล้ว infer สำเร็จ ไม่มี error
    เหลือแค่ mAP ที่ต่ำผิดปกติให้ตามหาสาเหตุเอง
    """
    img, r, pad = letterbox(bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(img[None]), r, pad


def preprocess_batch(bgr_list: list[np.ndarray], size: int = IMG_SIZE):
    outs, metas = [], []
    for bgr in bgr_list:
        x, r, pad = preprocess(bgr, size)
        outs.append(x)
        metas.append((r, pad))
    return np.ascontiguousarray(np.concatenate(outs, axis=0)), metas


# --------------------------------------------------------------------------
# postprocess
# --------------------------------------------------------------------------
def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    half_w, half_h = x[:, 2] / 2, x[:, 3] / 2
    y[:, 0] = x[:, 0] - half_w
    y[:, 1] = x[:, 1] - half_h
    y[:, 2] = x[:, 0] + half_w
    y[:, 3] = x[:, 1] + half_h
    return y


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    # เขียนเองด้วย numpy แทน torchvision.ops.nms / cv2.dnn.NMSBoxes เพราะทั้งสาม
    # runtime ต้องใช้ตัวเดียวกัน ถ้า PyTorch ใช้ torchvision แล้ว TRT ใช้อีกตัว
    # คอลัมน์ mAP จะกลายเป็นการเทียบ NMS สองตัว ไม่ใช่เทียบ runtime
    # อีกเหตุผล: NMS ถูกนับใน post_ms ที่รายงาน ต้องเป็นภาระเดียวกันทุกแถว
    #
    # NOTE: O(n²) วนด้วย Python loop ตอนวัด mAP (conf 0.001) จะเหลือ box หลักพัน
    # ต่อภาพ ทำให้ evaluate.py ช้ากว่าที่ควร ยังไม่ได้จับเวลาจริงว่าเท่าไร
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


def postprocess(
    raw: np.ndarray,
    meta,
    conf_thres: float = DEPLOY_CONF,
    iou_thres: float = DEPLOY_IOU,
    max_det: int = 300,
):
    # แต่ละ runtime คาย layout ไม่เหมือนกัน (torch head / ONNX export / TRT engine)
    # เลยเดาจาก shape แทนฮาร์ดโค้ด: ด้านสั้นคือ channel (84 = 4 box + 80 class)
    # ด้านยาวคือ anchor (8400 ที่ input 640)
    # ตรวจแล้วครอบคลุม (1,84,8400) (1,8400,84) (84,8400) (8400,84)
    #
    # NOTE: heuristic นี้พังถ้า anchor < channel ซึ่งต้อง input เล็กกว่า ~200px
    # ไม่ใช่เคสของโปรเจกต์นี้ แต่ถ้าจะเพิ่ม imgsz เล็กตอน Phase 4 ต้องกลับมาดูตรงนี้
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] < raw.shape[1]:     
        raw = raw.T

    boxes_xywh = raw[:, :4]
    cls_scores = raw[:, 4:]

    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    m = scores > conf_thres
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int32)

    boxes = _xywh2xyxy(boxes_xywh[m])
    scores, class_ids = scores[m], class_ids[m]

    # class-offset NMS: ดัน box ของแต่ละคลาสไปคนละย่านพิกัดก่อน NMS เพื่อไม่ให้
    # box ต่างคลาสที่ทับกันจริง (คนถือกระเป๋า) กดกันเอง
    # 7680 = 640×12 — box ยังอยู่ในพิกัด letterbox (สูงสุด 640) ระยะห่างจึงเหลือเฟือ
    # ทำแบบนี้ได้ NMS รอบเดียวแทนที่จะวน 80 คลาส
    offset = class_ids.astype(np.float32) * 7680.0
    keep = nms_numpy(boxes + offset[:, None], scores, iou_thres)[:max_det]

    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    # กลับทาง letterbox — ลำดับสำคัญ ต้องลบ pad ก่อนแล้วค่อยหาร r
    # (letterbox resize ก่อนแล้วค่อย pad) สลับลำดับแล้วกล่องเลื่อนไปตามขนาด pad
    #
    # ไม่ clip ขอบที่นี่ เพราะ benchmark ไม่ต้องใช้ — evaluate.py:74 clip เองก่อนส่ง COCO
    #
    # NOTE: คำนวณด้วย dtype ที่ raw ส่งมา วัดแล้ว float32 คลาด 0.000 px
    # แต่ float16 คลาด 1.0 px (ภาพ 1080p, r=0.333) เพราะ float16 ที่ย่าน 640
    # ละเอียดแค่ 0.5 เข้าถึงเคสนี้ได้ผ่าน build_engine.py --fp16-head ที่ set_output_type
    # ให้ layer ท้ายเป็น float16 — ถ้าจะใช้โหมดนั้นจริงต้อง cast เป็น float32 ก่อนถึงตรงนี้
    r, (padx, pady) = meta
    boxes[:, [0, 2]] -= padx
    boxes[:, [1, 3]] -= pady
    boxes /= r
    return boxes.astype(np.float32), scores.astype(np.float32), class_ids.astype(np.int32)


# --------------------------------------------------------------------------
# COCO helpers
# --------------------------------------------------------------------------
# โมเดลคาย class index 0-79 ต่อเนื่อง แต่ instances_val2017.json ใช้ category_id
# 1-90 ที่มีรู (12, 26, 29, 30, 45, 66, 68, 69, 71, 83 ไม่ถูกใช้)
# ถ้าส่ง index ดิบเข้า COCOeval มันจับคู่คลาสผิดหมดแล้วได้ mAP ใกล้ 0 โดยไม่มี error
# สักบรรทัด — pycocotools ไม่เช็กว่า category_id ที่ส่งมาสมเหตุสมผลไหม
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
