import cv2
import numpy as np


def recognize_generator(tables_img, struct_sizes):
    for table_img in tables_img:
        cells, image_without_lines = get_cells(table_img, struct_sizes)
        rects_indexes = get_rects_indexes(cells)
        cells_imgs = get_cells_imgs(image_without_lines, rects_indexes)

        yield cells_imgs


def get_cells(table_img, struct_sizes):
    h_size, w_size = struct_sizes

    image = np.pad(table_img, pad_width=20, mode='constant', constant_values=255)
    union = _get_union(image, h_size, w_size)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    union_dilated = cv2.dilate(union, kernel, iterations=2)

    image_without_lines = cv2.subtract(cv2.bitwise_not(image), union_dilated)
    image_without_lines = cv2.bitwise_not(image_without_lines)

    # cv2.imwrite('without_lines.jpg', image_without_lines)
    # cv2.imwrite('union.jpg', union)

    contours, _ = cv2.findContours(union, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = list()
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 100:
            continue

        rect = cv2.boundingRect(contour)

        # Width and height may be small
        if rect[2] < 10 or rect[3] < 10:
            continue

        cells.append(rect)
    
    return cells, image_without_lines


def get_rects_indexes(cells):
    sorted_columns = _custom_sort(cells)
    sorted_rows = _custom_sort(cells, indexes=(1, 0))

    sorted_columns = _indexing(sorted_columns[1:])
    sorted_rows = _indexing(sorted_rows[1:], index=1)

    result = list()
    for i, rect in sorted_rows:
        for j, rect1 in sorted_columns:
            if rect == rect1:
                result.append(((i, j), rect))
    
    return result


def get_cells_imgs(table_img, rects_indexes):
    result = list()
    for indexes, rect in rects_indexes:
        x, y, w, h = rect
        cell_img = table_img[y:y+h, x:x+w]
        cv2.imwrite(f'indexes.jpg', cell_img)

        result.append((indexes, cell_img))
    
    return result


def _get_union(image, h_size, w_size):
    image_inv = cv2.bitwise_not(image)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    horz = cv2.morphologyEx(image_inv, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, w_size))
    vert = cv2.morphologyEx(image_inv, cv2.MORPH_OPEN, kernel)

    return cv2.bitwise_or(horz, vert)


def _custom_sort(rects, indexes=(0, 1), delta=10):
    previous = None
    i, j = indexes

    def sort_key(rect):
        nonlocal previous

        if previous is None or abs(rect[i] - previous) > delta:
            previous = rect[i]
            return (rect[i], rect[j])
        else:
            return (previous, rect[j])
        
    return sorted(rects, key=sort_key)


def _indexing(rects, index=0, delta=10):
    result = list()
    previous = None
    i = -1
    
    for rect in rects:
        if previous is None or abs(rect[index] - previous) > delta:
            previous = rect[index]
            i += 1

        result.append((i, rect))

    return result