import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def make_filter(G, D0, n):
    shape = 2 * n
    H = np.zeros((shape, shape))
    center = np.array(H.shape) / 2.0
    order = 2
    exp = 2 * order

    # filter in fourier domain...
    for iy in range(shape):
        for ix in range(shape):
            D_x_y = ((iy - center[0]) ** 2 + (ix - center[1]) ** 2) ** 0.5
            D_final = (D_x_y / D0) ** exp + 1
            H[iy, ix] = 1 / D_final

    title = "Filter " + str(D0)
    filter = (H * 50).astype(np.uint8)
    cv2.imshow(title, filter)

    filter_spectrum = np.zeros((shape, shape))

    for iy in range(shape):
        for ix in range(shape):
            filter_spectrum[iy][ix] = 20 * np.log(abs(H[iy][ix]))

    filter_spectrum = filter_spectrum.astype(np.uint8)
    title = "Filter Spectrum " + str(D0)
    cv2.imshow(title, filter_spectrum)

    Gx = np.multiply(G, H)

    Gx = np.fft.ifft2(Gx)
    Gx = Gx.real
    for iy in range(shape):
        for ix in range(shape):
            Gx[iy][ix] = Gx[iy][ix] * ((-1) ** (ix + iy))
    return Gx


def butterworth_filter():
    # resizing image
    img = Image.open("Cameraman.jpg")
    new_size = (256, 256)
    img = img.resize(new_size)
    img.save("resized_img.jpg")

    # reopening image
    img_path = input("Enter image path: ")
    img = cv2.imread(img_path, 0)

    # display input image
    input_image = img.astype(np.uint8)
    cv2.imshow("Input Image", input_image)

    m, n = len(img), len(img[0])

    # padded image
    padded_img = np.zeros((2 * m, 2 * n))
    padded_img[:-256, :-256] = img

    Image_padded = padded_img.astype(np.uint8)
    cv2.imshow("Padded Image", Image_padded)

    shifted_padded_img = np.zeros((2 * m, 2 * n))

    shape = 2 * n

    # shifting centre
    for iy in range(shape):
        for ix in range(shape):
            shifted_padded_img[iy][ix] = padded_img[iy][ix] * ((-1) ** (ix + iy))

    # unshifted dft
    dft = np.fft.fft2(padded_img)

    magnitude_spectrum_unshifted = np.zeros((shape, shape))

    for iy in range(shape):
        for ix in range(shape):
            magnitude_spectrum_unshifted[iy][ix] = 20 * np.log(abs(dft[iy][ix]))

    spectrum_unshifted = magnitude_spectrum_unshifted.astype(np.uint8)
    cv2.imshow("Unshifted Spectrum", spectrum_unshifted)

    # centred dft
    G = np.fft.fft2(shifted_padded_img)

    magnitude_spectrum = np.zeros((shape, shape))
    for iy in range(shape):
        for ix in range(shape):
            magnitude_spectrum[iy][ix] = 20 * np.log(abs(G[iy][ix]))

    spectrum = magnitude_spectrum.astype(np.uint8)
    cv2.imshow("Spectrum", spectrum)

    G1 = make_filter(G, 10, n)

    cropped_image1 = np.zeros((m, n))
    cropped_image1 = G1[:-256, :-256]
    p_img1 = cropped_image1.astype(np.uint8)
    cv2.imshow("D0_10", p_img1)

    G2 = make_filter(G, 30, n)

    cropped_image2 = np.zeros((m, n))
    cropped_image2 = G2[:-256, :-256]
    p_img2 = cropped_image2.astype(np.uint8)
    cv2.imshow("D0_30", p_img2)

    G3 = make_filter(G, 60, n)

    cropped_image3 = np.zeros((m, n))
    cropped_image3 = G3[:-256, :-256]
    p_img3 = cropped_image3.astype(np.uint8)
    cv2.imshow("D0_60", p_img3)

    cv2.waitKey(0)


def manual_convolution():
    # opening image
    # img_path = input("Enter image path: ")
    img = Image.open("Cameraman.jpg")

    new_size = (256, 256)
    img = img.resize(new_size)
    img.save("resized_img.jpg")

    # opening image
    # img_path = input("Enter image path: ")
    img = cv2.imread("resized_img.jpg", 0)

    input_image = img.astype(np.uint8)
    cv2.imshow("Input Image", input_image)

    m, n = len(img), len(img[0])

    print(m, n)

    p, q = 9, 9

    # creating box filter (9x9)
    box_filter = (np.ones((9, 9))) / 81

    inbuilt_conv = cv2.filter2D(img, -1, box_filter)

    p_img1 = inbuilt_conv.astype(np.uint8)
    cv2.imshow("Inbuilt Convolution result", p_img1)

    padded_img = np.zeros((m + p - 1, n + q - 1))
    padded_img[:-p + 1, :-q + 1] = img

    padded_box_filter = np.zeros((m + p - 1, n + q - 1))
    padded_box_filter[:-m + 1, :-n + 1] = box_filter

    pad_img = padded_img.astype(np.uint8)
    cv2.imshow("Padded_Image", pad_img)

    G = np.fft.fft2(padded_img)
    H = np.fft.fft2(padded_box_filter)

    G = np.multiply(H, G)

    G = np.fft.ifft2(G)
    G = G.real

    cropped_image = np.zeros((m, n))
    cropped_image = G[:-9, :-9]
    fft_img = cropped_image.astype(np.uint8)
    cv2.imshow("FFT Convolution", fft_img)
    cv2.waitKey(0)


def denoise_barbara():
    # img_path = input("Enter image path: ")
    img = cv2.imread("noiseIm.jpg", 0)

    input_image = img.astype(np.uint8)
    cv2.imshow("Input Image", input_image)

    m, n = len(img), len(img[0])

    # padded image
    padded_img = np.zeros((2 * m, 2 * n))
    padded_img[:m, :n] = img

    shifted_padded_img = np.zeros((2 * m, 2 * n))

    # shifting centre
    for iy in range(2 * m):
        for ix in range(2 * n):
            shifted_padded_img[iy][ix] = padded_img[iy][ix] * ((-1) ** (ix + iy))

    G = np.fft.fft2(shifted_padded_img)

    magnitude_spectrum = np.log(1 + np.abs(G))

    noise_zone = []

    var1 = magnitude_spectrum.astype(np.uint8)
    plt.imshow(var1.tolist(), cmap="gray")
    plt.show()

    while True:
        x_coordinate = int(input("Enter the x coordinate: "))
        y_coordinate = int(input("Enter the y coordinate: "))
        noise_zone.append([x_coordinate, y_coordinate])
        exit_loop = input("Are you done? (y/n) ")
        if exit_loop == "y":
            break

    filter = np.ones((2 * m, 2 * n))

    print(noise_zone)

    for i in range(len(noise_zone)):
        for j in range(-15, 15):
            for k in range(-15, 15):
                filter[int(noise_zone[i][0]) + j][int(noise_zone[i][1]) + k] = 0

    G = np.fft.fft2(shifted_padded_img)
    G = np.multiply(G, filter)
    G = np.fft.ifft2(G)
    G = G.real

    # shifting centre
    for iy in range(2 * m):
        for ix in range(2 * n):
            G[iy][ix] = G[iy][ix] * ((-1) ** (ix + iy))

    cropped_image = np.zeros((m, n))
    cropped_image = G[:m, :n]

    fft_img = cropped_image.astype(np.uint8)
    cv2.imshow("Noise free", fft_img)
    cv2.waitKey(0)


choice = int(input("Enter Choice: "))

while 1 <= choice <= 3:
    if choice == 1:
        butterworth_filter()
    elif choice == 2:
        manual_convolution()
    elif choice == 3:
        denoise_barbara()
    choice = int(input("Enter Choice: "))
