from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAffined,
    RandFlipd,
    RandAdjustContrastd,
)


def get_train_transform() -> Compose:
    return Compose([
        # [H, W] → [1, H, W] so spatial transforms can work
        EnsureChannelFirstd(keys=['img', 'mask'], channel_dim='no_channel'),
        RandAffined(
            keys=['img', 'mask'],
            mode=['bilinear', 'nearest'],
            prob=0.5,
            rotate_range=(0.26,),            # ±15°
            translate_range=(10, 10),        # ±10 px
            scale_range=((-0.1, 0.1), (-0.1, 0.1)),   # 0.9–1.1
            padding_mode='zeros',
        ),
        RandFlipd(keys=['img', 'mask'], spatial_axis=0, prob=0.3),
        RandFlipd(keys=['img', 'mask'], spatial_axis=1, prob=0.3),
        RandAdjustContrastd(keys=['img'], prob=0.5, gamma=(0.7, 1.3)),
    ])
