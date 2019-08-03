# encoding: utf-8
import cv2
import numpy as np
import scipy.misc
import skimage
from skimage import measure

scale = 2

def calculate_metrics(hr_y_list, sr_y_list, bnd=2):
    class BaseMetric:
        def __init__(self):
            self.name = 'base'

        def image_preprocess(self, image):
            image_copy = image.copy()
            image_copy[image_copy < 0] = 0
            image_copy[image_copy > 255] = 255
            image_copy = np.around(image_copy).astype(np.double)
            return image_copy

        def evaluate(self, gt, pr):
            pass

        def evaluate_list(self, gtlst, prlst):
            resultlist = list(map(lambda gt, pr: self.evaluate(gt, pr), gtlst, prlst))
            return sum(resultlist) / len(resultlist)


    class PSNRMetric(BaseMetric):
        def __init__(self):
            self.name = 'psnr'

        def evaluate(self, gt, pr):
            gt = self.image_preprocess(gt)
            pr = self.image_preprocess(pr)
            return skimage.measure.compare_psnr(gt, pr, data_range=255)


    class SSIMMetric(BaseMetric):
        def __init__(self):
            self.name = 'ssim'

        def evaluate(self, gt, pr):
            def ssim(img1, img2):
                C1 = (0.01 * 255) ** 2
                C2 = (0.03 * 255) ** 2

                img1 = img1.astype(np.float64)
                img2 = img2.astype(np.float64)
                kernel = cv2.getGaussianKernel(11, 1.5)
                window = np.outer(kernel, kernel.transpose())

                mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
                mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
                mu1_sq = mu1 ** 2
                mu2_sq = mu2 ** 2
                mu1_mu2 = mu1 * mu2
                sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
                sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
                sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

                ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
                return ssim_map.mean()

            def esrgan_ssim(img1, img2):
                if not img1.shape == img2.shape:
                    raise ValueError('Input images must have the same dimensions.')
                if img1.ndim == 2:
                    return ssim(img1, img2)
                elif img1.ndim == 3:
                    if img1.shape[2] == 3:
                        ssims = []
                        for i in range(3):
                            ssims.append(ssim(img1, img2))
                        return np.array(ssims).mean()
                    elif img1.shape[2] == 1:
                        return ssim(np.squeeze(img1), np.squeeze(img2))
                else:
                    raise ValueError('Wrong input image dimensions.')

            gt = self.image_preprocess(gt)
            pr = self.image_preprocess(pr)
            return esrgan_ssim(gt[..., 0], pr[..., 0])

    y_mean_psnr = 0
    y_mean_ssim = 0
    assert len(hr_y_list) == len(sr_y_list)
    for i in range(len(hr_y_list)):
        hr_y, sr_y = hr_y_list[i], sr_y_list[i]
        hr_y = hr_y[bnd:-bnd, bnd:-bnd, :]
        sr_y = sr_y[bnd:-bnd, bnd:-bnd, :]
        y_mean_psnr += PSNRMetric().evaluate(sr_y, hr_y) / len(sr_y_list)
        y_mean_ssim += SSIMMetric().evaluate(sr_y, hr_y) / len(sr_y_list)
    return y_mean_psnr, y_mean_ssim
