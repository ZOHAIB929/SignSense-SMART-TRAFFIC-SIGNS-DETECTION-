import cv2
import numpy as np
import imutils
import os
import math
import argparse
from neuralNet import *
from classification import training, getLabel

SIGNS = ["ERROR", "STOP", "TURN LEFT", "TURN RIGHT", "DO NOT TURN LEFT", "DO NOT TURN RIGHT", "ONE WAY", "SPEED LIMIT", "OTHER"]

def clean_images():
    for file_name in os.listdir('./'):
        if file_name.endswith('.png'):
            os.remove(file_name)

def contrast_limit(image):
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_ycrcb)
    channels[0] = cv2.equalizeHist(channels[0])
    img_ycrcb = cv2.merge(channels)
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2BGR)

def laplacian_of_gaussian(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3, scale=2)
    return cv2.convertScaleAbs(laplacian)

def binarization(image):
    return cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]

def preprocess_image(image):
    return binarization(laplacian_of_gaussian(contrast_limit(image)))

def remove_small_components(image, threshold):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    img_filtered = np.zeros(labels.shape, dtype=np.uint8)
    for i, size in enumerate(sizes):
        if size >= threshold:
            img_filtered[labels == i + 1] = 255
    return img_filtered

def find_contours(image):
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0] if imutils.is_cv2() else contours[1]

def contour_is_sign(perimeter, centroid, threshold):
    distances = [math.sqrt((p[0][0] - centroid[0]) ** 2 + (p[0][1] - centroid[1]) ** 2) for p in perimeter]
    max_distance = max(distances)
    signature = [d / max_distance for d in distances]
    mean_signature = sum((1 - s) for s in signature) / len(signature)
    return mean_signature < threshold, max_distance + 2

def crop_contour(image, center, max_distance):
    top, bottom = max(int(center[1] - max_distance), 0), min(int(center[1] + max_distance), image.shape[0] - 1)
    left, right = max(int(center[0] - max_distance), 0), min(int(center[0] + max_distance), image.shape[1] - 1)
    return image[top:bottom, left:right]

def crop_sign(image, coordinate):
    top, bottom = max(int(coordinate[0][1]), 0), min(int(coordinate[1][1]), image.shape[0] - 1)
    left, right = max(int(coordinate[0][0]), 0), min(int(coordinate[1][0]), image.shape[1] - 1)
    return image[top:bottom, left:right]

def find_largest_sign(image, contours, threshold, distance_threshold):
    max_distance = 0
    sign, coordinate = None, None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        is_sign, distance = contour_is_sign(c, [cX, cY], 1 - threshold)
        if is_sign and distance > max_distance and distance > distance_threshold:
            max_distance = distance
            coord = np.reshape(c, [-1, 2])
            left, top = np.amin(coord, axis=0)
            right, bottom = np.amax(coord, axis=0)
            coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
            sign = crop_sign(image, coordinate)
    return sign, coordinate

def remove_other_colors(img):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, np.array([100, 128, 0]), np.array([215, 255, 255]))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 128]), np.array([255, 255, 255]))
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([170, 150, 50]))
    return cv2.bitwise_or(cv2.bitwise_or(mask_blue, mask_white), mask_black)

def localization(image, min_size, threshold, model, count, current_sign_type):
    binary_image = preprocess_image(image)
    binary_image = cv2.bitwise_and(remove_small_components(binary_image, min_size), remove_other_colors(image))
    contours = find_contours(binary_image)
    sign, coordinate = find_largest_sign(image.copy(), contours, threshold, 15)
    text, sign_type = "", -1
    if sign is not None:
        sign_type = getLabel(model, sign)
        sign_type = min(sign_type, 8)
        text = SIGNS[sign_type]
        cv2.imwrite(f"{count}_{text}.png", sign)
    if sign_type > 0 and sign_type != current_sign_type:
        cv2.rectangle(image, coordinate[0], coordinate[1], (0, 255, 0), 1)
        cv2.putText(image, text, (coordinate[0][0], coordinate[0][1] - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_4)
    return coordinate, image, sign_type, text

def main(args):
    clean_images()
    model = training()
    vidcap = cv2.VideoCapture(args.file_name)
    fps, width, height = vidcap.get(cv2.CAP_PROP_FPS), vidcap.get(3), vidcap.get(4)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 480))
    termination, roiBox, roiHist = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), None, None
    count, current_sign, sign_count, coordinates = 0, None, 0, []
    with open("Output.txt", "w") as file:
        while True:
            success, frame = vidcap.read()
            if not success:
                print("Finished")
                break
            frame = cv2.resize(frame, (640, 480))
            coordinate, image, sign_type, text = localization(frame, args.min_size_components, args.similarity_contour_with_circle, model, count, current_sign)
            if coordinate:
                cv2.rectangle(image, coordinate[0], coordinate[1], (255, 255, 255), 1)
            if sign_type > 0 and (not current_sign or sign_type != current_sign):
                current_sign = sign_type
                top, left, bottom, right = coordinate[0][1], coordinate[0][0], coordinate[1][1], coordinate[1][0]
                position = [count, sign_type, left, top, right, bottom]
                cv2.rectangle(image, coordinate[0], coordinate[1], (0, 255, 0), 1)
                cv2.putText(image, text, (left, top - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_4)
                roi = frame[top:bottom, left:right]
                roiHist = cv2.normalize(cv2.calcHist([roi], [0], None, [16], [0, 180]), roiHist, 0, 255, cv2.NORM_MINMAX)
                roiBox = (left, top, right, bottom)
            elif current_sign:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
                (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
                pts = np.int0(cv2.boxPoints(r))
                tl, br = pts[np.argmin(pts.sum(axis=1))], pts[np.argmax(pts.sum(axis=1))]
                if sign_type > 0:
                    position = [count, sign_type, left, top, right, bottom]
                else:
                    position = [count, sign_type, tl[0], tl[1], br[0], br[1]]
                    cv2.rectangle(image, tuple(tl), tuple(br), (0, 255, 0), 1)
                if sign_type != current_sign:
                    current_sign, sign_count = sign_type, 0
                if sign_count > 10:
                    current_sign, sign_count = None, 0
            file.write(f"{count}, {current_sign}, {coordinate[0][0]}, {coordinate[0][1]}, {coordinate[1][0]}, {coordinate[1][1]}\n" if coordinate else "No coordinates\n")
            out.write(image)
            count += 1
    vidcap.release()
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", required=True, help="Video file path")
    parser.add_argument("--min_size_components", type=int, default=100, help="Minimum size of the components")
    parser.add_argument("--similarity_contour_with_circle", type=float, default=0.5, help="Similarity threshold")
    args = parser.parse_args()
    main(args)
