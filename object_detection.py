import multiprocessing
from pathlib import Path
import sys
import time

import torch


yolo_root = Path(__file__).resolve().parent / "yolov9"
sys.path.append(str(yolo_root))

from yolov9.utils.general import check_imshow, check_img_size, Profile, LOGGER, non_max_suppression, scale_boxes
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.torch_utils import select_device

# testing modifying LoadStreams class
from custom_loader import customLoader

def yolo_detector(
        source: multiprocessing.Queue,
        centers_queue: multiprocessing.Queue,
        width,
        height,
        fps,
        frames,
        imgsz=(640, 480),
        weights='best.pt',
        data=yolo_root / 'data/coco.yaml',
        vid_stride=1,
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=5,  # maximum detections per image
        # line_thickness=3,  # bounding box thickness (pixels)
        ):

    device = select_device('cpu')
    weights = yolo_root / weights
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    view_img = check_imshow(warn=True)
    if not view_img:
        print('ERROR: environment unsuitable')
        exit()
    dataset = customLoader(source, width, height, fps, frames, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)


    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=False)

        # NMS
        with dt[2]:
            pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)
        
        # print(f'{len(pred)} predictions')
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0, frame = im0s[i].copy(), dataset.count
            s += f'{i}: '
           
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Convertir les coordonnées en pixels entiers
                    bbox_pixels = tuple(map(int, xyxy))  # (x1, y1, x2, y2)

                    # Calcul du centre
                    x1, y1, x2, y2 = bbox_pixels
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Afficher les résultats
                    print(f"Bounding box: {bbox_pixels}")
                    print(f"Object center: ({center_x}, {center_y})")

                    centers_queue.put(((center_x, center_y), bbox_pixels, conf))

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        time.sleep(0.1)
    
    print('YOLO detection shut down')
    return






