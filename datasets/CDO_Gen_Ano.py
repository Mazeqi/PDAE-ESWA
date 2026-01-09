
import numpy as np
import cv2

SKIP_BACKGROUND = {
    'mvtec': {
        'carpet': False,
        'grid': False,
        'leather': False,
        'tile': False,
        'wood': False,
        'bottle': True,
        'cable': False,
        'capsule': False,
        'hazelnut': True,
        'metal_nut': True,
        'pill': True,
        'screw': True,
        'toothbrush': True,
        'transistor': False,
        'zipper': True
    }
}
class CDO_GEN_ANO():
    def __init__(self, image_size = 256, dataset_name = "mvtec", category = "pill"):
        super().__init__()
        self.h = image_size
        self.w = image_size
        self.skip_bkg = SKIP_BACKGROUND[dataset_name][category]

    def __call__(self, images):
        images = np.array(images)
        _, augmented_image, anomaly_mask, _ = self.transform_image(images)
        return augmented_image, anomaly_mask
    
    def estimate_background(self, image, thr_low=30, thr_high=220):

        gray_image = np.mean(image * 255, axis=2)

        bkg_msk_high = np.where(gray_image > thr_high, np.ones_like(gray_image), np.zeros_like(gray_image))
        bkg_msk_low = np.where(gray_image < thr_low, np.ones_like(gray_image), np.zeros_like(gray_image))

        bkg_msk = np.bitwise_or(bkg_msk_low.astype(np.uint8), bkg_msk_high.astype(np.uint8))
        bkg_msk = cv2.medianBlur(bkg_msk, 7)
        kernel = np.ones((19, 19), np.uint8)
        bkg_msk = cv2.dilate(bkg_msk, kernel)

        bkg_msk = bkg_msk.astype(np.float32)
        return bkg_msk

    def augment_image(self, image):

        # generate noise image
        noise_image = np.random.randint(0, 255, size=image.shape).astype(np.float32) / 255.0
        patch_mask = np.zeros(image.shape[:2], dtype=np.float32)

        # get bkg mask
        bkg_msk = self.estimate_background(image)

        # generate random mask
        patch_number = np.random.randint(0, 5)
        augmented_image = image

        MAX_TRY_NUMBER = 200
        for i in range(patch_number):
            try_count = 0
            while try_count < MAX_TRY_NUMBER:
                try_count += 1

                patch_dim1 = np.random.randint(self.h // 20, self.h // 2)
                patch_dim2 = np.random.randint(self.w // 20, self.w // 2)

                center_dim1 = np.random.randint(patch_dim1, image.shape[0] - patch_dim1)
                center_dim2 = np.random.randint(patch_dim2, image.shape[1] - patch_dim2)

                if self.skip_bkg:
                    if bkg_msk[center_dim1, center_dim2] > 0:
                        continue

                coor_min_dim1 = np.clip(center_dim1 - patch_dim1, 0, image.shape[0])
                coor_min_dim2 = np.clip(center_dim2 - patch_dim2, 0, image.shape[1])

                coor_max_dim1 = np.clip(center_dim1 + patch_dim1, 0, image.shape[0])
                coor_max_dim2 = np.clip(center_dim2 + patch_dim2, 0, image.shape[1])

                break

            patch_mask[coor_min_dim1:coor_max_dim1, coor_min_dim2:coor_max_dim2] = 1.0

        augmented_image[patch_mask > 0] = noise_image[patch_mask > 0]

        patch_mask = patch_mask[:, :, np.newaxis]

        if patch_mask.max() > 0:
            has_anomaly = 1.0
        else:
            has_anomaly = 0.0

        return augmented_image, patch_mask, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image):
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image)

        image = (image * 255.0).astype(np.uint8)
        augmented_image = (augmented_image * 255.0).astype(np.uint8)
        anomaly_mask = (anomaly_mask[:, :, 0] * 255.0).astype(np.uint8)

        return image, augmented_image, anomaly_mask, has_anomaly