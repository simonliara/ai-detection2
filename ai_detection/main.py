import argparse

import cv2
import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.util import duration
from data_bus.topics.perception import ObjectDetection, get_object_detection_writer
from data_bus.topics.sensor_data import get_timed_image_reader
from data_bus.types.DataBus.SensorData import Encoding, TimedImage
from ultralytics import YOLO

# classes:
# person = 0
# car = 2
# truck = 7

person_classes = [0]
vehicle_classes = [2, 7]


class DetectionProcess:
    def __init__(self, domain_id, yolo_path, minimum_confidence, iou):
        self._detection_list = {}
        self._model = YOLO(yolo_path)
        self._minimum_confidence = minimum_confidence
        self._iou = iou
        self._participant = DomainParticipant(domain_id)
        self._image_reader = get_timed_image_reader(self._participant)
        self._ai_detection_writer = get_object_detection_writer(self._participant)

    def loop(self):
        while True:
            try:
                timed_image: TimedImage = self._image_reader.take_one(timeout=duration(seconds=1.0))
                image = None
                if timed_image.image.encoding == Encoding.JPEG:
                    image = cv2.imdecode(np.asarray(timed_image.image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if timed_image.image.encoding == Encoding.BGR24:
                    image = np.asarray(timed_image.image.data, dtype=np.uint8).reshape((timed_image.image.height, timed_image.image.width, 3))
                if image is None:
                    continue
                self.process_image(image, timed_image)

            except StopIteration:
                pass

    def process_image(self, image, timed_image: TimedImage):
        results = self._model.track(
            source=image, stream=True, persist=True, verbose=True, conf=self._minimum_confidence, classes=[0, 2, 7], device=0, iou=self._iou
        )
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = 0
                if len(box.conf):
                    confidence = box.conf[0].item()

                class_name = "unknown"
                if len(box.cls):
                    class_id = int(box.cls[0].item())
                    if class_id in person_classes:
                        class_name = "person"
                    elif class_id in vehicle_classes:
                        class_name = "vehicle"

                unique_id = 0
                if box.id is not None:
                    unique_id = int(box.id[0].item())

                detection_list_value = self._detection_list.get(unique_id, "unknown")
                if detection_list_value == "unknown":
                    if class_name != "unknown":
                        self._detection_list[unique_id] = class_name
                        detection_list_value = class_name
                    else:
                        continue

                box_center_x, box_center_y, box_width, box_height = box.xywhn[0]

                detection_msg = ObjectDetection(
                    sourceIdentity=timed_image.sourceIdentity,
                    time=timed_image.imageTime,
                    id=unique_id,
                    classification=class_name,
                    confidenceScore=confidence,
                    referenceFrame=timed_image.referenceFrame,
                    centerX=box_center_x,
                    centerY=box_center_y,
                    sizeX=box_width,
                    sizeY=box_height,
                )

                self._ai_detection_writer.write(detection_msg)


def main():
    parser = argparse.ArgumentParser(
        prog="Ai Detection",
        description="Detect person and car from images",
    )
    parser.add_argument(
        "--domain-id",
        dest="domain_id",
        help="Cyclonedds domain id",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--yolo-path",
        dest="yolo_path",
        help="minimum confidence level range [0,1]",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--minimum-confidence",
        dest="minimum_confidence",
        help="minimum confidence level range [0,1]",
        default=0.4,
        type=float,
    )
    parser.add_argument(
        "--iou",
        dest="iou",
        help="Intersection Over Union threshold level range [0,1]",
        default=0.5,
        type=float,
    )

    args = parser.parse_args()

    if args.minimum_confidence < 0:
        args.minimum_confidence = 0
    elif args.minimum_confidence > 1:
        args.minimum_confidence = 1

    if args.iou < 0:
        args.iou = 0
    elif args.iou > 1:
        args.iou = 1

    detection_process = DetectionProcess(args.domain_id, args.yolo_path, args.minimum_confidence, args.iou)
    try:
        detection_process.loop()
    except KeyboardInterrupt:
        print("Received keyboard interrupt")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
