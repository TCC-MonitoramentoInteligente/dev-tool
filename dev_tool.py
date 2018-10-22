import argparse
import cv2

from detector import Detector
from fainting_recognition import FaintingRecognition


def arg_parse():
    parser = argparse.ArgumentParser(description='A tool with graphical support for algorithm development')

    parser.add_argument("--video", "-v", help="Path to video file", required=True, type=str)
    parser.add_argument("--frame-loss", "-fl", dest="frame_loss",
                        help='Simulate frame loss rate caused by UDP transmission',
                        type=int, choices=range(0, 100), default=0)
    parser.add_argument("--detection-loss", "-dl", dest="non_detection",
                        help='Simulate the non-detection rate caused by noise',
                        type=int, choices=range(0, 100), default=0)

    return parser.parse_args()


def draw_box(frame, obj, color_bgr, thickness=2):
    """
    Draw a rectangle in an numpy frame
    :param frame: numpy frame
    :param obj: dict with detected object
    :param color_bgr: color
    :param thickness: thickness of rectangle
    :return: numpy frame
    """
    top_left = (obj['x'], obj['y'])
    bottom_right = (obj['x'] + obj['width'], obj['y'] + obj['height'])
    return cv2.rectangle(frame, top_left, bottom_right, color_bgr, thickness)


def main(args):
    window = "Dev Tool"

    cap = cv2.VideoCapture(args.video)
    detector = Detector()
    detector.load_model()
    algorithm = FaintingRecognition()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 600, 600)

    frame_loss_counter = 0
    non_detection_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        if args.frame_loss > 0:
            frame_loss_counter += 1
            if (1 / frame_loss_counter) <= (args.frame_loss / 100):
                frame_loss_counter = 0
                continue

        if args.non_detection > 0:
            non_detection_counter += 1
            if (1 / non_detection_counter) <= (args.non_detection / 100):
                frame_loss_counter = 0
                object_list = []
            else:
                object_list = detector.detect(frame)
        else:
            object_list = detector.detect(frame)

        person_list = algorithm.process(object_list, cap.get(cv2.CAP_PROP_POS_MSEC))

        for p in person_list:
            color = (0, 0, 0)

            if p.state == algorithm.state_normal:
                color = (0, 255, 0)  # Green
            elif p.state == algorithm.state_horizontal_warning:
                color = (255, 0, 0)  # Blue
            elif p.state == algorithm.state_vertical_warning:
                color = (0, 255, 255)  # Yellow
            elif p.state == algorithm.state_movement_alert:
                color = (0, 128, 255)  # Orange
            elif p.state == algorithm.state_fallen:
                color = (0, 0, 255)  # Red
            frame = draw_box(frame, p.object, color)

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    arguments = arg_parse()
    main(arguments)
