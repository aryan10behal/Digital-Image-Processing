import cv2
import numpy as np
import matplotlib.pyplot as plt


def Normalised_Histogram():
    img_path = input("Enter image path: ")
    img = cv2.imread(img_path, 0)

    # deep copy of img
    output_img = np.copy(img)

    # row and col count
    m = len(img)
    n = len(img[0])
    total_pixel = m * n

    # Normalised histogram
    h = []

    for i in range(256):
        y = np.where(img == i)
        h.append(len(y[0]) / total_pixel)

    # Finding CDF of histogram
    cdf_sum = 0

    H = []
    for i in range(256):
        cdf_sum = cdf_sum + h[i]
        H.append(cdf_sum)
    H = np.array(H)

    # Constructing equalised image and its CDF
    S = []
    for i in range(256):
        y = np.where(img == i)
        S.append(255 * H[i])
        output_img[y] = S[i]

    output_img = output_img.astype(np.uint8)
    s = []

    for i in range(256):
        y = np.where(output_img == i)
        s.append(len(y[0]) / total_pixel)

    cv2.imshow("Input_Image", img)

    cv2.imshow("Output_Image", output_img)

    f = plt.figure(1)
    plt.bar([i for i in range(256)], h)
    plt.xlabel("pixel val")
    plt.ylabel("noramlised value for pixel")
    plt.title("Histogram for Input Image")
    plt.plot()
    f.show()

    g = plt.figure(2)
    plt.bar([i for i in range(256)], s)
    plt.xlabel("pixel val")
    plt.ylabel("noramlised value for pixel")
    plt.title("Histogram for Equalised Image")
    g.show()

    plt.show()


def Finding_matched_image():
    img_path = input("Enter image path: ")
    img = cv2.imread("img.jpg", 0)

    # finding value of c for gamma function
    c = 255 / (255 ** 0.5)

    # constructing target image
    target_image = np.rint(c * np.sqrt(img))
    target_image = target_image.astype(np.uint8)

    # row and col count
    m = len(img)
    n = len(img[0])
    total_pixel = m * n

    # Computing normalised histogram values for input image and target image..
    h = []
    g = []

    for i in range(256):
        y = np.where(img == i)
        h.append(len(y[0]) / total_pixel)
        x = np.where(target_image == i)
        g.append(len(x[0]) / total_pixel)

    h = np.array(h)
    g = np.array(g)

    # Finding CDF of histogram
    cdf_sum_for_h = 0
    cdf_sum_for_g = 0

    H = []
    G = []
    for i in range(256):
        cdf_sum_for_h = cdf_sum_for_h + h[i]
        H.append(cdf_sum_for_h)
        cdf_sum_for_g = cdf_sum_for_g + g[i]
        G.append(cdf_sum_for_g)

    H = np.array(H)
    G = np.array(G)

    # finding the mapping for pixels..
    mapping = []

    for i in range(256):
        min_diff = 2
        mapped_j = -1
        for j in range(256):
            if abs(H[i] - G[j]) < min_diff:
                mapped_j = j
                min_diff = abs(H[i] - G[j])
        mapping.append(mapped_j)

    # constructing the matched image..
    matched_image = np.copy(img)

    for i in range(256):
        y = np.where(img == i)
        matched_image[y] = mapping[i]

    matched_image = matched_image.astype(np.uint8)

    # making normalised histogram for matched image
    m_h = []
    for i in range(256):
        y = np.where(matched_image == i)
        m_h.append(len(y[0]) / total_pixel)

    # displaying images
    cv2.imshow("Input_Image", img)
    cv2.imshow("Target_Image", target_image)
    cv2.imshow("Matched_Image", matched_image)

    print("\nThe pixel mappings are: \n")
    print(mapping)
    print("\n\n")

    f_show = plt.figure(1)
    plt.bar([i for i in range(256)], h)
    plt.xlabel("pixel val")
    plt.ylabel("noramlised value for pixel")
    plt.title("Normalised Histogram for Input Image")
    plt.plot()
    f_show.show()

    g_show = plt.figure(2)
    plt.bar([i for i in range(256)], g)
    plt.xlabel("pixel val")
    plt.ylabel("noramlised value for pixel")
    plt.title("Normalised Histogram for Target Image")
    g_show.show()

    m_show = plt.figure(3)
    plt.bar([i for i in range(256)], m_h)
    plt.xlabel("pixel val")
    plt.ylabel("noramlised value for pixel")
    plt.title("Normalised Histogram for Matched Image")
    m_show.show()

    plt.show()


def Convolution():
    # Taking input for the filter matrix...
    filter_matrix = []

    print("Enter the filter matrix (3x3):\n")

    for i in range(3):
        row = []
        for j in range(3):
            row.append(int(input("Enter " + str(i + 1) + "." + str(j + 1) + ": ")))
        filter_matrix.append(row)

    filter_matrix = np.array(filter_matrix)

    # rotating matrix by 180 degree.. 90 degree twice
    rotated_matrix = np.rot90(filter_matrix, 2)

    # printing original and rotated filter matrix
    print("\n-----Original filter matrix------\n")
    print(filter_matrix)
    print("\n-----Rotated filter matrix------\n")
    print(rotated_matrix)

    # taking input for image matrix
    image_matrix = []

    print("\nEnter the image matrix (3x3):\n")

    for i in range(3):
        row = []
        for j in range(3):
            row.append(int(input("Enter " + str(i + 1) + "." + str(j + 1) + ": ")))
        image_matrix.append(row)

    image_matrix = np.array(image_matrix)

    # printing image matrix..
    print("\n----Image Matrix-------\n")
    print(image_matrix)

    # making padded image matrix..
    padded_image = np.zeros((7, 7))

    for i in range(7):
        for j in range(7):
            if i < 2 or i > 4 or j < 2 or j > 4:
                padded_image[i][j] = int(0)
            else:
                padded_image[i][j] = int(image_matrix[i - 2][j - 2])

    print("\n----Padded Image Matrix:--------\n\n ", padded_image)

    # making output matrix (5x5)
    output_matrix = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            val = 0
            for x in range(3):
                for y in range(3):
                    val = val + padded_image[i + x][j + y] * rotated_matrix[x][y]
            output_matrix[i][j] = int(val)

    np.set_printoptions(suppress=True)
    output_matrix = np.array(output_matrix)
    print("\n-------Convolution Result:--------\n\n", output_matrix)


if __name__ == "__main__":
    while True:
        choose = int(input("Enter Ques: \n1. Normalised_Histogram\n2. Finding_matched_image\n3. Convolution\n4."
                           "Done: "))
        if choose == 1:
            Normalised_Histogram()
        elif choose == 2:
            Finding_matched_image()
        elif choose == 3:
            Convolution()
        else:
            break
