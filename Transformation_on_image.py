from math import *
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def Image_extrapolation():
    img_path = input("Enter image path: ")
    img = cv2.imread(img_path, 0)

    interpolation_factor = float(input("Enter the interpolation factor: "))

    row = len(img)
    col = len(img[0])

    # padding image
    p_img = np.zeros((row + 2, col + 2))
    p_img[:-2, :-2] = img

    sign = 0

    if interpolation_factor < 0:
        sign = 1
        interpolation_factor = abs(interpolation_factor)

    new_row = int(interpolation_factor * row)
    new_col = int(interpolation_factor * col)

    output_matrix = np.ones((new_row + 2, new_col + 2)) * -1

    # mapping the direct values
    for i in range(row):
        for j in range(col):
            output_matrix[int(i * interpolation_factor)][int(j * interpolation_factor)] = p_img[i][j]

    # filling the missing values
    for i in range(new_row):
        for j in range(new_col):
            if output_matrix[i][j] == -1:
                x = i / interpolation_factor
                y = j / interpolation_factor

                if floor(x) != x:
                    x1 = floor(x)
                    x2 = ceil(x)
                else:
                    if x == 0:
                        x1 = 0
                        x2 = 1
                    else:
                        x1 = x - 1
                        x2 = x

                if floor(y) != y:
                    y1 = floor(y)
                    y2 = ceil(y)
                else:
                    if y == 0:
                        y1 = 0
                        y2 = 1
                    else:
                        y1 = y - 1
                        y2 = y

                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                X = [
                    [x1, y1, x1 * y1, 1],
                    [x1, y2, x1 * y2, 1],
                    [x2, y1, x2 * y1, 1],
                    [x2, y2, x2 * y2, 1]
                ]

                Y = [
                    [p_img[x1][y1]],
                    [p_img[x1][y2]],
                    [p_img[x2][y1]],
                    [p_img[x2][y2]]
                ]

                if np.linalg.det(X) == 0:
                    val = 0;
                    for l in range(len(X)):
                        for m in range(len(X[0])):
                            if X[l][m] != 0:
                                val = X[l][m]
                                break

                    X = np.add(X, np.identity(4) * val / 100000)

                A = np.dot(np.linalg.inv(X), Y)
                output_matrix[i][j] = np.dot(np.array([x, y, x * y, 1]), A)

    p_img = p_img.astype(np.uint8)
    cv2.imshow("Input_Image", p_img)

    if sign == 0:
        output_matrix = output_matrix.astype(np.uint8)
        cv2.imshow("Output_Image", output_matrix)
    else:
        output_matrix2 = np.ones((new_row, new_col)) * -1

        for i in range(new_row):
            for j in range(new_col):
                output_matrix2[new_row - 1 - i][new_col - 1 - j] = output_matrix[i][j]

        output_matrix2 = output_matrix2.astype(np.uint8)
        cv2.imshow("Output_Image", output_matrix2)

    cv2.waitKey(0)


def Tranformation_on_Image():
    # taking input for the image path
    img_path = input("Enter image path: ")
    img = cv2.imread(img_path, 0)

    row = len(img)
    col = len(img[0])

    # padding the image
    p_img = np.zeros((row + 2, col + 2))
    p_img[:-2, :-2] = img

    T_matrix = np.identity(3)

    make_matrix = True

    # constructing the transformation matrix
    while make_matrix:
        choice = int(input("Enter the action you want to perform: \n1. Rotation\n2. Tranlation\n3. Scaling\n4. You are "
                           "done: "))
        if choice == 1:
            degree = float(input("Enter the angle: "))

            rot_matrix = np.ones((3, 3)) * 0
            rot_matrix[2][2] = 1

            rot_matrix[0][0], rot_matrix[1][0] = math.cos(math.radians(degree)), math.sin(math.radians(degree))
            rot_matrix[1][1] = rot_matrix[0][0]
            rot_matrix[0][1] = -1 * rot_matrix[1][0]

            T_matrix = np.dot(T_matrix, rot_matrix)
        elif choice == 2:
            X_translation = float(input("Enter the Translation on X axis: "))
            Y_translation = float(input("Enter the Translation on Y axis: "))

            trans_matrix = np.ones((3, 3)) * 0
            trans_matrix[0][0], trans_matrix[1][1], trans_matrix[2][2] = 1, 1, 1

            trans_matrix[2][0], trans_matrix[2][1] = X_translation, Y_translation

            T_matrix = np.dot(T_matrix, trans_matrix)
        elif choice == 3:
            X_scale = float(input("Enter the Scaling factor on X axis: "))
            Y_scale = float(input("Enter the Scaling factor on Y axis: "))

            scale_matrix = np.ones((3, 3)) * 0

            scale_matrix[2][2] = 1
            scale_matrix[0][0] = X_scale
            scale_matrix[1][1] = Y_scale

            T_matrix = np.dot(T_matrix, scale_matrix)
        else:
            make_matrix = False

    new_row = 8 * row
    new_col = 8 * col

    # making the output matrix
    output_matrix = np.zeros((new_row, new_col))

    if np.linalg.det(T_matrix) == 0:
        val_2 = 0
        for v in range(len(T_matrix)):
            for w in range(len(T_matrix[0])):
                if T_matrix[v][w] != 0:
                    val_2 = T_matrix[v][w]
                    break
        X = np.add(T_matrix, np.identity(4) * val_2 / 100000)

    T_inverse = np.linalg.inv(T_matrix)

    # reverse mapping the pixel and filling missing pixels using bi-linear interpolation
    for i in range(new_row):
        for j in range(new_col):
            X_map = np.dot(np.array([i - row * 4, j - col * 4, 1]), T_inverse)
            x = X_map[0]
            y = X_map[1]

            if x < 0 or x > row + 1 or y < 0 or y > col + 1:
                continue
            else:
                if floor(x) != x:
                    x1 = floor(x)
                    x2 = ceil(x)
                else:
                    if x == 0:
                        x1 = 0
                        x2 = 1
                    else:
                        x1 = x - 1
                        x2 = x

                if floor(y) != y:
                    y1 = floor(y)
                    y2 = ceil(y)
                else:
                    if y == 0:
                        y1 = 0
                        y2 = 1
                    else:
                        y1 = y - 1
                        y2 = y

                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                X = [
                    [x1, y1, x1 * y1, 1],
                    [x1, y2, x1 * y2, 1],
                    [x2, y1, x2 * y1, 1],
                    [x2, y2, x2 * y2, 1]
                ]

                Y = [
                    [p_img[x1][y1]],
                    [p_img[x1][y2]],
                    [p_img[x2][y1]],
                    [p_img[x2][y2]]
                ]

                if np.linalg.det(X) == 0:
                    val = 0
                    for l in range(len(X)):  # if det(X) = 0, adding λI where λ very small
                        for m in range(len(X[0])):
                            if X[l][m] != 0:
                                val = X[l][m]
                                break
                    X = np.add(X, np.identity(4) * X[0][0] / 100000)

                A = np.dot(np.linalg.inv(X), Y)
                output_matrix[i][j] = np.dot(np.array([x, y, x * y, 1]), A)

    # displaying the input image (padded)

    p_img = p_img.astype(np.uint8)
    cv2.imshow("Input_Image", p_img)

    # displaying the output image

    output_matrix = output_matrix.astype(np.uint8)
    cv2.imshow("Output_Image", output_matrix)

    # displaying the grids
    var1 = output_matrix.astype(np.uint8)
    plt.imshow(var1.tolist(), cmap="gray")
    plt.show()

    print(T_matrix)
    im = Image.fromarray(output_matrix)
    im.save("Output_File.jpeg")

    cv2.waitKey(0)


# We know that V = XT. Hence to find T, we need to perform (X^-1)V.
# We need 3 coordinates in X and their mapping in V.
# These are obtained from helper_code.py and the 2 images that are produced.

# 3 points from X are (7.3, 33.9) mapped to (247, 345), (63.4, 63.7) mapped to (285, 464) and (16.1, 51.9) mapped to
# (235, 380.9)

# So X = [[7.3, 33.9, 1], [63.4, 63.7, 1], [16.1, 51.9, 1]] and V = [[247, 345, 1], [285, 464, 1], [235, 380.9, 1]]
def Back_to_Original():
    X = np.array([[7.3, 33.9, 1], [63.4, 63.7, 1], [16.1, 51.9, 1]])
    V = np.array([[247, 345, 1], [285, 464, 1], [235, 380.9, 1]])

    if np.linalg.det(V) == 0:
        val_2 = 0
        for v in range(len(X)):
            for w in range(len(X)):
                if X[v][w] != 0:
                    val_2 = X[v][w]
                    break
        V = np.add(X, np.identity(3) * val_2 / 100000)

    V_inverse = np.linalg.inv(V)

    Z = np.dot(V_inverse, X)

    Z_inverse = np.linalg.inv(Z)

    img_path = input("Enter image path: ")
    img = cv2.imread(img_path, 0)

    row = len(img)
    col = len(img[0])

    # padding the image
    p_img = np.zeros((row + 2, col + 2))
    p_img[:-2, :-2] = img

    new_row = 2 * row
    new_col = 2 * col

    Z_inverse = [[1.4,-1.4,0],[1.4,1.4,0],[29,29,1]]
    # making the output matrix
    output_matrix = np.zeros((new_row, new_col))

    # reverse mapping the pixel and filling missing pixels using bi-linear interpolation
    for i in range(new_row):
        for j in range(new_col):
            X_map = np.dot(np.array([i - row , j - col , 1]), Z_inverse)
            x = X_map[0]
            y = X_map[1]

            if x < 0 or x > row + 1 or y < 0 or y > col + 1:
                continue
            else:
                if floor(x) != x:
                    x1 = floor(x)
                    x2 = ceil(x)
                else:
                    if x == 0:
                        x1 = 0
                        x2 = 1
                    else:
                        x1 = x - 1
                        x2 = x

                if floor(y) != y:
                    y1 = floor(y)
                    y2 = ceil(y)
                else:
                    if y == 0:
                        y1 = 0
                        y2 = 1
                    else:
                        y1 = y - 1
                        y2 = y

                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

                X = [
                    [x1, y1, x1 * y1, 1],
                    [x1, y2, x1 * y2, 1],
                    [x2, y1, x2 * y1, 1],
                    [x2, y2, x2 * y2, 1]
                ]

                Y = [
                    [p_img[x1][y1]],
                    [p_img[x1][y2]],
                    [p_img[x2][y1]],
                    [p_img[x2][y2]]
                ]

                if np.linalg.det(X) == 0:
                    val = 0
                    for l in range(len(X)):  # if det(X) = 0, adding λI where λ very small
                        for m in range(len(X[0])):
                            if X[l][m] != 0:
                                val = X[l][m]
                                break
                    X = np.add(X, np.identity(4) * X[0][0] / 100000)

                A = np.dot(np.linalg.inv(X), Y)
                output_matrix[i][j] = np.dot(np.array([x, y, x * y, 1]), A)

    # displaying the input image (padded)

    p_img = p_img.astype(np.uint8)
    cv2.imshow("Input_Image", p_img)

    # displaying the output image

    output_matrix = output_matrix.astype(np.uint8)
    cv2.imshow("Output_Image", output_matrix)

    print(Z_inverse)

    cv2.waitKey(0)


if __name__ == "__main__":
    while True:
        choose = int(input("Enter Ques: \n1. Image Extrapolation\n2. Tranformation on Image\n3. Change to Original "
                           "Image\n4. "
                           "Done: "))
        if choose == 1:
            Image_extrapolation()
        elif choose == 2:
            Tranformation_on_Image()
        elif choose == 3:
            Back_to_Original()
        else:
            break
