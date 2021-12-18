import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def Equalize(I_equalizer):
    # deep copy of img
    final_equalised_I = np.copy(I_equalizer)

    # row and col count
    m = len(I_equalizer)
    n = len(I_equalizer[0])
    total_pixel = m * n

    # Normalised histogram
    h = []

    for i in range(256):
        y = np.where(I_equalizer == i)
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
        y = np.where(I_equalizer == i)
        S.append(255 * H[i])
        final_equalised_I[y] = S[i]
    s = []

    for i in range(256):
        y = np.where(final_equalised_I == i)
        s.append(len(y[0]) / total_pixel)

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
    return final_equalised_I


# img_path = input("path: ")
rgb_matrix = cv2.imread("./img2.tif", 1)
cv2.imshow("Input Image", rgb_matrix)

rgb_matrix = np.float32(rgb_matrix)/255

H = np.zeros((512, 512))
S = np.zeros((512, 512))
I = np.zeros((512, 512))

for i in range(512):
    for j in range(512):
        r = (rgb_matrix[i][j][0])
        g = (rgb_matrix[i][j][1])
        b = (rgb_matrix[i][j][2])
        val1 = r+b+g
        if val1 == 0:
            val1 = val1 + 0.00001
        val2 = (r - g) * (r - g) + (r - b) * (g - b)
        if val2 == 0:
            val2 = val2 + 0.000001
        val = ((r-g) + (r-b))/2*math.sqrt(val2)
        theta = math.acos(val)
        if b <= g:
            H[i][j] = theta
        else:
            H[i][j] = 2*math.pi - theta

        S[i][j] = 1 - (3 * min(r, min(g, b)))/(val1)

        I[i][j] = (r + g + b)/3

Final_I = Equalize(np.uint8(np.multiply(I, 255)))
Final_I = np.float32(Final_I)/255

HSI_image = cv2.merge((H, S, I))
cv2.imshow('HSI image', HSI_image)

# HSI_image_equalized = cv2.merge((H, S, Final_I))
# cv2.imshow('HSI image equalized', HSI_image_equalized)

R = np.zeros((512, 512))
G = np.zeros((512, 512))
B = np.zeros((512, 512))

Final_Image = np.zeros((512, 512, 3))

for i in range(512):
    for j in range(512):
        if 0 <= H[i][j] < 2*math.pi/3:
            B[i][j] = Final_I[i][j] * (1 - S[i][j])
            R[i][j] = Final_I[i][j] * (
                    1 + (S[i][j] * math.cos(H[i][j]) / math.cos(math.pi/3 - H[i][j])))
            G[i][j] = 3 * Final_I[i][j] - (R[i][j] + B[i][j])
        if 2*math.pi/3 <= H[i][j] < 4*math.pi/3:
            H[i][j] = H[i][j] - 2*math.pi/3
            R[i][j] = Final_I[i][j] * (1 - S[i][j])
            G[i][j] = Final_I[i][j] * (
                    1 + (S[i][j] * math.cos(H[i][j]) / math.cos(math.pi/3 - H[i][j])))
            B[i][j] = 3 * Final_I[i][j] - (R[i][j] + G[i][j])
        elif 4*math.pi/3 <= H[i][j] <= 2*math.pi:
            H[i][j] = H[i][j] - 4*math.pi/3
            G[i][j] = Final_I[i][j] * (1 - S[i][j])
            B[i][j] = Final_I[i][j] * (
                    1 + (S[i][j] * math.cos(H[i][j]) / math.cos(math.pi/3 - H[i][j])))
            R[i][j] = 3 * Final_I[i][j] - (G[i][j] + B[i][j])

        Final_Image[i][j][0] = R[i][j]
        Final_Image[i][j][1] = G[i][j]
        Final_Image[i][j][2] = B[i][j]

cv2.imshow('Final Image', Final_Image)
cv2.waitKey(0)
