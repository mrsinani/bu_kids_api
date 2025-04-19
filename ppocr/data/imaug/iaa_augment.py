# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Simplified version of IaaAugment that does not require albumentations.
This is a placeholder for Railway deployment.
"""

import numpy as np
import cv2
import random

class IaaAugment:
    def __init__(self, augmenter_args=None, **kwargs):
        self.augmenter_args = augmenter_args if augmenter_args else []
        
        # Parse augmentation parameters
        self.use_resize = False
        self.scale_range = (0.5, 3.0)
        self.use_flip = False
        self.flip_prob = 0.5
        self.use_rotate = False
        self.rotate_range = (-10, 10)
        
        # Process the augmenter args
        for aug in self.augmenter_args:
            if isinstance(aug, dict):
                aug_type = aug.get('type', '')
                aug_args = aug.get('args', {})
                
                if aug_type == 'Resize' and 'size' in aug_args:
                    self.use_resize = True
                    self.scale_range = tuple(aug_args['size'])
                elif aug_type == 'Fliplr' and 'p' in aug_args:
                    self.use_flip = True
                    self.flip_prob = aug_args['p']
                elif aug_type == 'Affine' and 'rotate' in aug_args:
                    self.use_rotate = True
                    rotate = aug_args['rotate']
                    if isinstance(rotate, list) and len(rotate) == 2:
                        self.rotate_range = tuple(rotate)

    def __call__(self, data):
        image = data["image"]
        polys = data["polys"]
        
        # Apply transformations
        h, w = image.shape[:2]
        
        # 1. Scale/resize transform
        if self.use_resize:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Scale polygons
            if polys is not None and len(polys) > 0:
                new_polys = []
                for poly in polys:
                    new_poly = poly * scale
                    new_polys.append(new_poly)
                polys = np.array(new_polys)
        
        # 2. Horizontal flip
        if self.use_flip and random.random() < self.flip_prob:
            image = cv2.flip(image, 1)  # 1 for horizontal flip
            
            # Flip polygons
            if polys is not None and len(polys) > 0:
                new_polys = []
                for poly in polys:
                    flipped_poly = poly.copy()
                    flipped_poly[:, 0] = w - poly[:, 0]
                    new_polys.append(flipped_poly)
                polys = np.array(new_polys)
        
        # 3. Rotation
        if self.use_rotate:
            angle = random.uniform(self.rotate_range[0], self.rotate_range[1])
            
            # Get rotation matrix
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to image
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            
            # Apply rotation to polygons
            if polys is not None and len(polys) > 0:
                new_polys = []
                for poly in polys:
                    new_poly = np.ones((len(poly), 3))
                    new_poly[:, :2] = poly
                    # Apply transformation
                    new_poly = np.dot(M, new_poly.T).T
                    new_polys.append(new_poly)
                polys = np.array(new_polys)
        
        # Update data
        data["image"] = image
        data["polys"] = polys
        
        return data
