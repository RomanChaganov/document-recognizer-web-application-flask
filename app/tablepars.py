import cv2
import numpy as np


def pars(binary):
    binary_inv = cv2.bitwise_not(binary)
    masks, struct_sizes = get_masks(binary_inv)
    tables = get_tables_region(masks)
    imgs = cut_tables(binary, tables)
    
    return imgs, struct_sizes


def get_masks(binary_inv):
    w_size = int(binary_inv.shape[1] / 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_size, 1))
    H = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)

    h_size = int(binary_inv.shape[0] / 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_size))
    V = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)

    union = cv2.bitwise_or(H, V)
    intersection = cv2.bitwise_and(H, V)

    return (union, intersection), (w_size, h_size)


def get_tables_region(masks):
    union, intersection = masks

    contours, _ = cv2.findContours(
        union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tables = list()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        approx = cv2.approxPolyDP(contour, 3, True)
        x, y, w, h = cv2.boundingRect(approx)

        possible_table = intersection[y:y+h, x:x+w]
        table_joints, _ = cv2.findContours(
            possible_table, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(table_joints) < 5:
            continue
            
        tables.append((x, y, w, h))
    
    return tables


def cut_tables(binary, tables):
    if tables is None:
        return binary, []

    text_img = np.copy(binary)
    
    table_imgs = list()
    for table in tables:
        x, y, w, h = table
        table_img = np.copy(text_img[y:y+h, x:x+w])
        text_img[y:y+h, x:x+w] = 255

        table_imgs.append(table_img)

    return text_img, table_imgs
