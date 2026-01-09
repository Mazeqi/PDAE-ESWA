import random
import numpy as np
from torchvision import transforms
from PIL import Image
class CutPaste(object):

    def __init__(self, transform = True, type = 'binary', cutpaste_area_ratio_list = []):

        '''
        This class creates to different augmentation CutPaste and CutPaste-Scar. Moreover, it returns augmented images
        for binary and 3 way classification

        :arg
        :transform[binary]: - if True use Color Jitter augmentations for patches
        :type[str]: options ['binary' or '3way'] - classification type
        '''
        self.type = type
        self.cutpaste_area_ratio_list = cutpaste_area_ratio_list
        if transform:
            self.transform = transforms.ColorJitter(brightness = 0.1,
                                                      contrast = 0.1,
                                                      saturation = 0.1,
                                                      hue = 0.1)
        else:
            self.transform = None

    def crop_and_paste_patch(self, image, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """

        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if transform:
            patch= transform(patch)

        new_w_left_de = int(org_w/2 * self.cutpaste_area_ratio_list[0])
        new_w_left_add = int(org_w/2 * self.cutpaste_area_ratio_list[1])
        new_h_top_de = int(org_h/2 * self.cutpaste_area_ratio_list[2])
        new_h_top_add = int(org_h/2 * self.cutpaste_area_ratio_list[3])
        
        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]
            #2 3 6 4
            #new_w_left_de = int(org_w/2)
            #new_w_left_add = int(org_w/2)
            #new_h_top_de =  int(org_h/4)
            #new_h_top_add = int(org_h/4)

        # new location
        paste_left, paste_top = random.randint(np.abs(org_w/2 - new_w_left_de), np.abs(org_w/2 + new_w_left_add)), random.randint(np.abs(org_h/2 - new_h_top_de), np.abs(org_h/2 + new_h_top_add))
        aug_image = image.copy()
        
        patch_rand = (np.random.rand(patch.size[1], patch.size[0], 3)*255).astype(np.uint8)
        patch_rand = Image.fromarray(patch_rand, mode='RGB')
        #patch = Image.new('RGB', patch.size, "red")
        aug_image.paste(patch_rand, (paste_left, paste_top), mask=mask)

        aug_mask   = Image.new('1', aug_image.size, 0)
        patch_mask = Image.new('1', patch.size, 1)
        aug_mask.paste(patch_mask, (paste_left, paste_top), mask=mask)
        #aug_mask = np.zeros(image.size, dtype=np.float32)
        #aug_mask[paste_top: paste_top + patch_h, paste_left:paste_left + patch_w] = 1.0
            
        return aug_image, aug_mask

    def cutpaste(self, image, area_ratio = (0.02, 0.15), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        '''
        CutPaste augmentation

        :image: [PIL] - original image
        :area_ratio: [tuple] - range for area ratio for patch
        :aspect_ratio: [tuple] -  range for aspect ratio

        :return: PIL image after CutPaste transformation
        '''

        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        cutpaste, aug_mask = self.crop_and_paste_patch(image, patch_w, patch_h, self.transform, rotation = (-45, 45))
        return cutpaste, aug_mask

    def cutpaste_scar(self, image, width = [2, 8], length = [2, 8], rotation = (-45, 45)):
        '''

        :image: [PIL] - original image
        :width: [list] - range for width of patch
        :length: [list] - range for length of patch
        :rotation: [tuple] - range for rotation

        :return: PIL image after CutPaste-Scare transformation
        '''
        patch_w, patch_h = random.randint(*width), random.randint(*length)
        cutpaste_scar, aug_mask = self.crop_and_paste_patch(image, patch_w, patch_h, self.transform, rotation = rotation)
        return cutpaste_scar, aug_mask

    def __call__(self, image):
        '''

        :image: [PIL] - original image
        :return: if type == 'binary' returns original image and randomly chosen transformation, else it returns
                original image, an image after CutPaste transformation and an image after CutPaste-Scar transformation
        '''
        if self.type == 'binary':
            aug = random.choice([self.cutpaste])
            return aug(image)

        elif self.type == '3way':
            cutpaste = self.cutpaste(image)
            scar = self.cutpaste_scar(image)
            return cutpaste, scar
