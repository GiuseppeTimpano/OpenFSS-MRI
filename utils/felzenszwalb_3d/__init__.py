"""
3D Felzenszwalb supervoxel segmentation (Cython).

Faithful port of the original Q-Net / SSL-ALPNet kernel. Build the extension
once on the target machine before use:

    cd utils/felzenszwalb_3d
    pip install cython
    python setup.py build_ext --inplace

Then `from utils.felzenszwalb_3d import felzenszwalb_3d`.
"""
import numpy as np

try:
    from ._felz3d import felzenszwalb_cython_3d
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "felzenszwalb_3d C extension not built. Run:\n"
        "    cd utils/felzenszwalb_3d && pip install cython && "
        "python setup.py build_ext --inplace"
    ) from exc


def felzenszwalb_3d(image, scale=1, sigma=0.8, min_size=20,
                    multichannel=True, spacing=(1, 1, 1)):
    """
    3D Felzenszwalb segmentation. Same signature as the original wrapper.

    image:    (D, H, W) ndarray
    min_size: minimum component size (the original passes n_sv here)
    spacing:  voxel spacing (z, x, y) for anisotropic edge weighting
    returns:  (D, H, W) int label volume
    """
    image = np.atleast_3d(image)
    return felzenszwalb_cython_3d(
        image, scale=scale, sigma=sigma, min_size=min_size, spacing=spacing)
