import torch
import numpy as np
import time
import math
from threading import Thread

from yolov9.utils.dataloaders import LOGGER
from yolov9.utils.augmentations import letterbox

class customLoader:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(
            self, 
            sources, 
            width,
            height,
            fps,
            frames,
            img_size=640, stride=32, 
            auto=True, transforms=None, 
            vid_stride=1,
            ):
        

        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride
        self.vid_stride = vid_stride  # video frame-rate stride
        source = sources

        # assert isinstance(source, type(multiprocessing.Queue))
        self.sources = source
        self.imgs, self.fps, self.frames, self.threads = None, 0, 0, None
        
        # Start thread to read frames from video stream
        # st = f'feed source: {s}... '

        # s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam

        # cap = cv2.VideoCapture(s)
        # assert cap.isOpened(), f'{st}Failed to open {s}'
        w = width
        h = height
        fps = fps  # warning: may return 0 or nan
        self.frames = frames  # infinite stream fallback
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

        _, self.imgs = source.get()  # guarantee first frame
        self.threads = Thread(target=self.update, args=(source, ), daemon=True)
        LOGGER.info(f" Success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)")
        self.threads.start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(self.imgs, img_size, stride=stride, auto=auto)[0].shape, ])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        self.auto = auto and self.rect
        self.transforms = transforms  # optional
        if not self.rect:
            LOGGER.warning('WARNING ⚠️ Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, source):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames  # frame number, frame array
        while True and n < f:
            n += 1
            # cap.grab()  # .read() = .grab() followed by .retrieve()
            if n % self.vid_stride == 0:
                success, im = source.get()
                while not source.empty():
                    success, im = source.get()
                if success:
                    self.imgs = im
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs = np.zeros_like(self.imgs)
                    # cap.open(stream)  # re-open stream if signal was lost
            time.sleep(0.0)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        # if not self.threads.is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
        if not self.threads.is_alive() or self.imgs is None:
            # cv2.destroyAllWindows()
            raise StopIteration

        im0 = [self.imgs.copy(), ]
        if self.transforms:
            im = np.stack([self.transforms(im0), ])  # transforms
        else:
            im = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0] for x in im0])  # resize
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            im = np.ascontiguousarray(im)  # contiguous

        return self.sources, im, im0, None, ''

    def __len__(self):
        return len([self.sources, ])  # 1E12 frames = 32 streams at 30 FPS for 30 years

