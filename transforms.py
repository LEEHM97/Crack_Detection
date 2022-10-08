import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def make_transform(args):
    test_transform = A.compose([ToTensorV2()])

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRain(
                brightness_coefficient=0.9,
                drop_length=10,
                drop_width=1,
                blur_value=3,
                p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, p=1
            ),
            A.RandomCrop(args.crop_image_size, args.crop_image_size),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5,
            ),
            A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.5),
            ToTensorV2(),
        ]
    )

    return train_transform, test_transform
