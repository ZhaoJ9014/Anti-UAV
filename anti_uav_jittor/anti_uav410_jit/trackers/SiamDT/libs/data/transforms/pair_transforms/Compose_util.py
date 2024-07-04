import jittor as jt
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img

class RandomResize:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        size = random.randint(self.min_size, self.max_size)
        return jt.img_resize(img, (size, size))

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        new_h, new_w = self.size, self.size
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return img[top: top + new_h, left: left + new_w]

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        new_h, new_w = self.size, self.size
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        return img[top: top + new_h, left: left + new_w]

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return jt.img_resize(img, self.size)

class ToTensor:
    def __call__(self, img):
        return jt.array(img)

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (img - self.mean) / self.std


transform = Compose([
            RandomResize(max_scale=1.05),
            CenterCrop(instance_sz - shift),
            RandomCrop(instance_sz - 2 * shift),
            ToTensor()])