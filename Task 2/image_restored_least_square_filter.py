import cv2
import numpy as np
import math

org_img = cv2.imread("./img.jpg", 0)

# img_path = input("path: ")
img = cv2.imread("./noiseIm.jpg", 0)

# display input image
input_image = img.astype(np.uint8)
cv2.imshow("Input Image", input_image)

# padded image
padded_img = np.zeros((512, 512))
padded_img[:256, :256] = img

Image_padded = padded_img.astype(np.uint8)
cv2.imshow("Padded Image", Image_padded)

# making box filter...
box_filter = np.ones((11, 11))/121
padded_box_filter = np.zeros((512, 512))
padded_box_filter[:11, :11] = box_filter

box_filter_image = padded_box_filter.astype(np.uint8)
cv2.imshow("Box filter padded", box_filter_image)

# making laplacian filter...
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
padded_laplacian = np.zeros((512, 512))
padded_laplacian[:3, :3] = laplacian


F_noise_img = np.fft.fft2(padded_img)
F_box = np.fft.fft2(padded_box_filter)
F_laplacian = np.fft.fft2(padded_laplacian)

box_filter_image_fourier = F_box.real.astype(np.uint8)
cv2.imshow("Box filter fourier", box_filter_image_fourier)

F_box_conjugate = np.conjugate(F_box)
F_box_abs_sq = np.square(np.abs(F_box))

F_laplacian_abs_sq = np.square(np.abs(F_laplacian))

lambdas = np.array([0, 0.25, 0.5, 0.75, 1])
lambda_optimal = math.inf

least_mse = math.inf
best_image = np.zeros((256, 256))


for l in lambdas:
    cls = np.divide(F_box_conjugate, (F_box_abs_sq + l*F_laplacian_abs_sq))
    cls_print = cls.real.astype(np.uint8)
    x = "Cls with lambda: "
    x = x + str(l)
    cv2.imshow(x, cls_print)
    F_restored = np.multiply(cls, F_noise_img)
    f_restored = np.fft.ifft2(F_restored).real
    f_restored_cropped = np.zeros((256, 256))
    f_restored_cropped = f_restored[:256, :256]

    x = "fft of images with lambda: "
    x = x + str(l)
    cur_img = F_restored.real.astype(np.uint8)
    cv2.imshow(x, cur_img)

    cur_img = f_restored_cropped.astype(np.uint8)
    x = "Image with lambda: "
    x = x + str(l)
    cv2.imshow(x, cur_img)
    # mean square error
    mse = np.mean(np.square(org_img - f_restored_cropped))
    print(mse)
    if mse < least_mse:
        least_mse = mse
        lambda_optimal = l
        best_image = f_restored_cropped

# displaying image
restored_img = best_image.astype(np.uint8)
cv2.imshow("Best Restored Image", restored_img)

print("lambda optimal: " + str(lambda_optimal))
print("least MSE: " + str(least_mse))

PSNR = 10*math.log10((255*255)/least_mse)

print("PSNR: " + str(PSNR))

cv2.waitKey(0)
