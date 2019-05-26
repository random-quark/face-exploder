from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import random


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


def bounding_box_naive(points):
    """returns a list containing the bottom left and the top right
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    bot_left_x = min(point[0] for point in points)
    bot_left_y = min(point[1] for point in points)
    top_right_x = max(point[0] for point in points)
    top_right_y = max(point[1] for point in points)

    return [(bot_left_x, bot_left_y), (top_right_x, top_right_y)]


def find_center(box):
    avg_x = (box[0][0] + box[1][0]) / 2
    avg_y = (box[0][1] + box[1][1]) / 2
    return [avg_x, avg_y]


def get_selection(img, point, height, width):
    marked_img = img.copy()
    # marked_img = cv2.circle(marked_img, (point[0], point[1]), 10, 0)

    v = height / 2
    h = width / 2

    y1 = point[1] - v
    y2 = point[1] + v
    x1 = point[0] - h
    x2 = point[0] + h
    crop_img = marked_img[y1:y2, x1:x2]
    return crop_img


selection_area = {
    "left_eye": [30, 30],
    "jaw": [30, 30],
    "left_eyebrow": [30, 30],
    "right_eye": [30, 30],
    "mouth": [50, 100],
    "nose": [50, 50],
    "right_eyebrow": [30, 30],
}

paste_locations = {
    "left_eye": [375, 150],
    "jaw": [100, 100],
    "left_eyebrow": [375, 100],
    "right_eye": [100, 150],
    "mouth": [250, 340],
    "nose": [250, 200],
    "right_eyebrow": [100, 100],
}


def place_on_canvas(input, pts):
    selections = []
    canvas = np.zeros((500, 500, 3), np.uint8)
    for name, point in pts.items():
        box = bounding_box_naive(point)
        center = find_center(box)
        new_image = input.copy()

        selection = get_selection(
            new_image, center, selection_area[name][0], selection_area[name][1])
        selections.append(selection)

        x, y = paste_locations[name]

        x1 = x - selection.shape[1] / 2
        x2 = x + selection.shape[1] / 2
        y1 = y - selection.shape[0] / 2
        y2 = y + selection.shape[0] / 2

        canvas[y1:y2, x1:x2] = selection
    return canvas


def get_static_image():
    return cv2.imread("eleanor.jpg")


cap = cv2.VideoCapture(0)


def get_live_image():
    ret, input = cap.read()
    return input


def create_image():
    input = get_live_image()
    resized_input = imutils.resize(input, width=500)
    gray = cv2.cvtColor(resized_input, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        # landmark_indices = face_utils.FACIAL_LANDMARKS_IDXS

        pts = {}

        for (i, name) in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
            (j, k) = face_utils.FACIAL_LANDMARKS_IDXS[name]
            pts[name] = shape[j:k]

        return place_on_canvas(input, pts)
    else:
        return input


while (True):
    output = create_image()

    # Display the resulting frame
    cv2.imshow('frame', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
