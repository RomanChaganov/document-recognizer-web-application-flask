import cv2
import numpy as np
from collections import Counter

def rotate_img(image):
    # Загрузка изображения
    # image = cv2.imread(img_path)
    padding = 50
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)[1]
    inverted_image = cv2.bitwise_not(thresholded_image)
    hor = np.array([[1, 1, 1, 1, 1, 1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    # cv2.imwrite('lines.jpg', vertical_lines_eroded_image)


    # Обнаружение прямых линий
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(vertical_lines_eroded_image, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    # line_image = np.copy(vertical_lines_eroded_image) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    if lines is not None:
        lines = list(lines)
    # else:
    #     show_result(image)
    #     exit()
    # lines.sort(key=lambda x: x[0][0]-x[0][2], reverse=False)

    # cv2.line(line_image,(coordinates[0],coordinates[1]),(coordinates[2],coordinates[3]),(255,35,0),5)

    angles = []
    if lines:
        for line in lines:
            angle = np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) * 180/np.pi
            if angle != 0:
                angles.append(angle)

    # angles = np.array(angles)
    # angle = angles.mean()
    angle = Counter(angles).most_common(1)[0][0] if angles != [] else 0

    # Применение геометрических преобразований
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    # cv2.imwrite('rotated_image.jpg', rotated_image)
    return rotated_image
#     show_result(rotated_image)
#
# def show_result(rotated_image):
#     cv2.imwrite('rotated_image.jpg', rotated_image)
#     cv2.imshow('Rotated Image', rotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()