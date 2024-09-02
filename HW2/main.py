import numpy as np
import matplotlib.pyplot as plt
import cv2


class LocalProcessor:
    def __init__(self, image_path):
        self.image = np.array(cv2.imread(
            image_path, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
        self.window_size = 8
        self.clip_limit = 50
        self.AHE = None
        self.CLAHE = None

    def image_histogram_equalization(self, image_grid, number_bins=256):
        image_histogram, bins = np.histogram(
            image_grid.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum()
        cdf = (number_bins-1) * cdf / cdf[-1]
        cdf = cv2.equalizeHist(image_histogram)
        image_equalized = np.interp(image_grid.flatten(), bins[:-1], cdf)
        return image_equalized.reshape(image_grid.shape)

    def clipped_image_histogram_equalization(self, image_grid, number_bins=256):
        image_histogram, bins = np.histogram(
            image_grid.flatten(), number_bins, density=True)
        excess = np.clip(image_histogram - self.clip_limit,
                         0, None).sum() // 256
        image_histogram = np.clip(image_histogram, 0, self.clip_limit)
        image_histogram += excess
        cdf = image_histogram.cumsum()
        cdf = (number_bins-1) * cdf / cdf[-1]
        image_equalized = np.interp(image_grid.flatten(), bins[:-1], cdf)
        return image_equalized.reshape(image_grid.shape)

    def apply_AHE(self):
        self.AHE = np.zeros(self.image.shape, dtype=np.uint8)
        for row in range(0, self.image.shape[0], self.window_size):
            for col in range(0, self.image.shape[1], self.window_size):
                row_end = min(row + self.window_size, self.image.shape[0])
                col_end = min(col + self.window_size, self.image.shape[1])
                block = self.image[row:row_end, col:col_end]
                self.AHE[row:row_end, col:col_end] = self.image_histogram_equalization(
                    block)

    def apply_CLAHE(self):
        self.CLAHE = np.zeros(self.image.shape, dtype=np.uint8)
        for row in range(0, self.image.shape[0], self.window_size):
            for col in range(0, self.image.shape[1], self.window_size):
                row_end = min(row + self.window_size, self.image.shape[0])
                col_end = min(col + self.window_size, self.image.shape[1])
                block = self.image[row:row_end, col:col_end]
                self.CLAHE[row:row_end,
                           col:col_end] = self.clipped_image_histogram_equalization(block)

    def visualize(self):
        plt.figure(figsize=(15, 6))
        plt.subplot(131), plt.imshow(
            self.image, cmap='gray'), plt.title('Original Image')
        plt.subplot(132), plt.imshow(
            self.AHE, cmap='gray'), plt.title('Output of AHE')
        plt.subplot(133), plt.imshow(
            self.CLAHE, cmap='gray'), plt.title('Output of CLAHE')
        plt.show()


localProcessor = LocalProcessor("./image1.png")
localProcessor.apply_AHE()
localProcessor.apply_CLAHE()
localProcessor.visualize()


#################################################################################
def remove_noise(image_path):
    image = cv2.imread(image_path, 0)
    f = np.fft.fft2(image)
    cf = np.fft.fftshift(f)

    cf[1:125, 110:130] = 0
    cf[190:320, 110:130] = 0
    cf[120:image.shape[0], 1:100] = 0
    cf[1:100, 120:image.shape[1]] = 0

    ifourier = np.fft.ifftshift(cf)
    denoised_image = np.fft.ifft2(ifourier)

    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(
        image, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(np.abs(denoised_image),
                                 cmap='gray'), plt.title('Denoised Image')
    plt.show()


remove_noise("./image2.jpg")
