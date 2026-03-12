from typing import Dict

import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

from vismatch.utils import to_normalized_coords, to_px_coords, to_numpy, _load_image, to_tensor_image


# Union of all accepted matcher inputs: a raw image in any supported form, or
# a pre-extracted feature set (Dict).
MatchInput = torch.Tensor | np.ndarray | str | Path | Image.Image | Dict


class BaseMatcher(torch.nn.Module):
    """
    This serves as a base class for all matchers. It provides a simple interface
    for its sub-classes to implement, namely each matcher must specify its own
    ``__init__`` and ``_forward`` methods. It also provides a common image loader
    and homography estimator.

    Sub-classes must implement at least one of:

    * ``_forward(img0, img1) -> (mkpts0, mkpts1, kpts0, kpts1, desc0, desc1)``:
      the traditional image-pair path (required when callers supply raw images).
        * ``extract_features(img) -> (kpts, desc)`` **and**
            ``match_features(kpts0, desc0, kpts1, desc1) -> (mkpts0, mkpts1)``:
      the decoupled extraction/matching path that enables ``(image, features)``
      and ``(features, features)`` inputs.

    The :meth:`forward` method accepts any mix of images and feature set (Dict)
    objects and automatically dispatches to the appropriate path:

    * ``(image, image)``       — calls ``_forward`` (original behaviour).
        * ``(image, features)``    — extracts one side via ``extract_features``, then matches
            via ``match_features``.
        * ``(features, features)`` — calls ``match_features`` directly.
    """

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__()
        self.device: str = device

        self.skip_ransac: bool = False

        # OpenCV default ransac params
        self.ransac_iters: int = kwargs.get("ransac_iters", 2000)
        self.ransac_conf: float = kwargs.get("ransac_conf", 0.95)
        self.ransac_reproj_thresh: float = kwargs.get("ransac_reproj_thresh", 3)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def load_image(path: str | Path, resize: int | tuple = None, rot_angle: float = 0) -> torch.Tensor:
        """load image from filesystem and return as tensor. Optionally rotate and resize.

        Args:
            path (str | Path): path to image on filesystem
            resize (int | tuple, optional): size to resize img, either single value for square resize or tuple of (H, W). Defaults to None.
            rot_angle (float, optional): CCW rotation angle in degrees. Defaults to 0.

        Returns:
            torch.Tensor: image as tensor (C x H x W)
        """
        return _load_image(path=path, resize=resize, rot_angle=rot_angle)

    def rescale_coords(
        self,
        pts: np.ndarray | torch.Tensor,
        h_orig: int,
        w_orig: int,
        h_new: int,
        w_new: int,
    ) -> np.ndarray:
        """Rescale kpts coordinates from one img size to another

        Args:
            pts (np.ndarray | torch.Tensor): (N,2) array of kpts
            h_orig (int): height of original img
            w_orig (int): width of original img
            h_new (int): height of new img
            w_new (int): width of new img

        Returns:
            np.ndarray: (N,2) array of kpts in original img coordinates
        """
        return to_px_coords(to_normalized_coords(pts, h_new, w_new), h_orig, w_orig)

    def compute_ransac(
        self, matched_kpts0: np.ndarray, matched_kpts1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process matches into inliers and the respective Homography using RANSAC.

        Args:
            matched_kpts0 (np.ndarray): matching kpts from img0
            matched_kpts1 (np.ndarray): matching kpts from img1

        Returns:
            H (np.ndarray): (3 x 3) homography matrix from img0 to img1. Can be None if no homography is found
            inlier_kpts0 (np.ndarray): inlier kpts in img0
            inlier_kpts1 (np.ndarray): inlier kpts in img1
        """
        if len(matched_kpts0) < 4 or self.skip_ransac:  # Sperical matchers like sphereglue skip RANSAC
            return None, np.empty([0, 2]), np.empty([0, 2])

        H, inliers_mask = cv2.findHomography(
            matched_kpts0,
            matched_kpts1,
            cv2.USAC_MAGSAC,
            self.ransac_reproj_thresh,
            self.ransac_conf,
            self.ransac_iters,
        )
        inliers_mask = inliers_mask[:, 0].astype(bool)
        inlier_kpts0 = matched_kpts0[inliers_mask]
        inlier_kpts1 = matched_kpts1[inliers_mask]

        return H, inlier_kpts0, inlier_kpts1

    def extract_features(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract keypoints and descriptors from a single image tensor.

        The default implementation calls ``_forward(img, img)`` and returns the
        ``all_kpts`` / ``all_desc`` outputs for that image.  Sub-classes that
        own a dedicated feature extractor (e.g. SuperPoint, SIFT) should
        override this method for efficiency so the network is not run twice.

        Args:
            img (torch.Tensor): image tensor ``(3, H, W)`` in ``[0, 1]`` range,
                already on the correct device.

        Returns:
            dict: result dict with keys:
                - all_kpts0 (torch.Tensor): (N, 2) detected keypoints
                - all_desc0 (torch.Tensor): (N, D) descriptors
        """
        _, _, all_kpts, _, all_desc, _ = self._forward(img, img)
        kpts = to_numpy(all_kpts) if all_kpts is not None else np.empty([0, 2])
        desc = to_numpy(all_desc) if all_desc is not None else np.empty([0, 0])
        return {"all_kpts0": kpts, "all_desc0": desc}

    def match_features(
        self,
        fset0: Dict[str, torch.Tensor],
        fset1: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Match pre-extracted features from two images.

        Sub-classes that support ``(image, features)`` or
        ``(features, features)`` inputs must override this method.
        The default implementation raises :class:`NotImplementedError`.

        Args:
            fset0 (Dict[str, torch.Tensor]): Feature set from the first image, must contain keys 'all_kpts0' and 'all_desc0'.
            fset1 (Dict[str, torch.Tensor]): Feature set from the second image, must contain keys 'all_kpts0' and 'all_desc0'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Matched keypoints from both images as ``(M, 2)`` arrays.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support feature-input matching. "
            "Override match_features() to enable (image, features) or "
            "(features, features) inputs."
        )

    @torch.inference_mode()
    def forward(
        self,
        input0: MatchInput,
        input1: MatchInput,
    ) -> dict:
        """Run the matching pipeline on two inputs.

        Each input may be a raw image (``torch.Tensor``, ``np.ndarray``,
        file path ``str``/``Path``, or ``PIL.Image``) *or* a pre-extracted
        feature set ``Dict``.  Three usage patterns are supported:

        * ``(image, image)``       — full pipeline via ``_forward``.
                * ``(image, features)``    — extract one side, match via
                    ``match_features``.
                * ``(features, features)`` — match directly via
                    ``match_features``.

        Matchers that do not decouple extraction from matching (e.g. LoFTR)
        only support the ``(image, image)`` mode.

        Args:
            input0 (MatchInput): first image or feature set ``Dict``.
            input1 (MatchInput): second image or feature set ``Dict``.

        Returns:
            dict: result dict with keys:
                - num_inliers (int): number of inliers after RANSAC, i.e. len(inlier_kpts0)
                - H (np.ndarray): (3 x 3) homography matrix to map matched_kpts0 to matched_kpts1
                - all_kpts0 (np.ndarray): (N0 x 2) all detected keypoints from img0
                - all_kpts1 (np.ndarray): (N1 x 2) all detected keypoints from img1
                - all_desc0 (np.ndarray): (N0 x D) all descriptors from img0
                - all_desc1 (np.ndarray): (N1 x D) all descriptors from img1
                - matched_kpts0 (np.ndarray): (N2 x 2) keypoints from img0 that match matched_kpts1 (pre-RANSAC)
                - matched_kpts1 (np.ndarray): (N2 x 2) keypoints from img1 that match matched_kpts0 (pre-RANSAC)
                - inlier_kpts0 (np.ndarray): (N3 x 2) filtered matched_kpts0 that fit the H model (post-RANSAC)
                - inlier_kpts1 (np.ndarray): (N3 x 2) filtered matched_kpts1 that fit the H model (post-RANSAC)
        """

        is_features0 = isinstance(input0, dict)
        is_features1 = isinstance(input1, dict)

        if not is_features0 and not is_features1:
            # (image, image) — original code path through _forward
            img0 = to_tensor_image(input0).to(self.device)
            img1 = to_tensor_image(input1).to(self.device)

            # self._forward() is implemented by the children modules
            matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1 = self._forward(img0, img1)
        else:
            # (image, features) or (features, features) — use extract_features + match_features
            if is_features0:
                all_kpts0 = input0["all_kpts0"]
                all_desc0 = input0["all_desc0"]
                fset0 = {"all_kpts0": all_kpts0, "all_desc0": all_desc0}
            else:
                img0 = to_tensor_image(input0).to(self.device)
                fset0 = self.extract_features(img0)
                all_kpts0, all_desc0 = fset0["all_kpts0"], fset0["all_desc0"]

            if is_features1:
                all_kpts1 = input1["all_kpts0"]
                all_desc1 = input1["all_desc0"]
                fset1 = {"all_kpts0": all_kpts1, "all_desc0": all_desc1}
            else:
                img1 = to_tensor_image(input1).to(self.device)
                fset1 = self.extract_features(img1)
                all_kpts1, all_desc1 = fset1["all_kpts0"], fset1["all_desc0"]

            matched_kpts0, matched_kpts1 = self.match_features(fset0, fset1)

        # Check that returned objects are of accepted types (nd.array, torch.tensor or None)
        self.check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1)

        # Convert torch tensors to numpy. None objects stay None
        matched_kpts0, matched_kpts1 = to_numpy(matched_kpts0), to_numpy(matched_kpts1)
        all_kpts0, all_kpts1 = to_numpy(all_kpts0), to_numpy(all_kpts1)
        all_desc0, all_desc1 = to_numpy(all_desc0), to_numpy(all_desc1)

        # Some models might return kpts=None if no kpts are found. In this case, set an empty array with dim (0, 2)
        matched_kpts0 = self.get_empty_array_if_none(matched_kpts0)
        matched_kpts1 = self.get_empty_array_if_none(matched_kpts1)
        all_kpts0 = self.get_empty_array_if_none(all_kpts0)
        all_kpts1 = self.get_empty_array_if_none(all_kpts1)
        # Same for descriptors: if it is empty set as descriptor an array with dim (0, 2)
        all_desc0 = self.get_empty_array_if_none(all_desc0)
        all_desc1 = self.get_empty_array_if_none(all_desc1)

        # Check that shapes are correct and consistent
        self.check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1)

        # Compute RANSAC to obtain the inliers and homography matrix
        H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "all_kpts0": all_kpts0,
            "all_kpts1": all_kpts1,
            "all_desc0": all_desc0,
            "all_desc1": all_desc1,
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
        }

    def extract(self, img: torch.Tensor | np.ndarray | str | Path | Image.Image) -> dict[str, np.ndarray]:
        """Extract keypoints and descriptors from a single image.

        Convenience wrapper around :meth:`forward`.  To extract features and
        keep them for later use as :class:`FeatureSet` inputs, prefer calling
        :meth:`extract_features` directly after moving the image tensor to the device.

        Args:
            img (torch.Tensor | np.ndarray | str | Path | Image.Image): image
                as ``(3, H, W)`` array in ``[0, 1]`` range, a file path, or a
                PIL Image.

        Returns:
            dict: result dict with keys:
                - all_kpts0 (np.ndarray): (N, 2) detected keypoints
                - all_desc0 (np.ndarray): (N, D) descriptors
        """
        result = self.forward(img, img)
        kpts = result["matched_kpts0"] if isinstance(self, EnsembleMatcher) else result["all_kpts0"]
        return {"all_kpts0": kpts, "all_desc0": result["all_desc0"]}

    @staticmethod
    def get_empty_array_if_none(array: np.ndarray | None) -> np.ndarray:
        if array is None or array.size == 0:
            return np.empty([0, 2])
        return array

    @staticmethod
    def check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1):
        """Check that objects are of accepted types (nd.array, torch.tensor or None)"""

        def is_array_or_tensor_or_none(data) -> bool:
            return data is None or isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)

        assert is_array_or_tensor_or_none(matched_kpts0)
        assert is_array_or_tensor_or_none(matched_kpts1)
        assert is_array_or_tensor_or_none(all_kpts0)
        assert is_array_or_tensor_or_none(all_kpts1)
        assert is_array_or_tensor_or_none(all_desc0)
        assert is_array_or_tensor_or_none(all_desc1)

    @staticmethod
    def check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1):
        """Check that objects have appropriate shapes, e.g. keypoints should have shape (N, 2)"""

        def check_kpts_shape(np_array) -> bool:
            """Keypoint arrays should be in the form of N x 2"""
            return np_array.ndim == 2 and np_array.shape[1] == 2

        assert check_kpts_shape(matched_kpts0), f"matched_kpts0 shape should be (N x 2) but it is {matched_kpts0.shape}"
        assert check_kpts_shape(matched_kpts1), f"matched_kpts1 shape should be (N x 2) but it is {matched_kpts1.shape}"
        assert check_kpts_shape(all_kpts0), f"all_kpts0 shape should be (N x 2) but it is {all_kpts0.shape}"
        assert check_kpts_shape(all_kpts1), f"all_kpts1 shape should be (N x 2) but it is {all_kpts1.shape}"
        # Number of matched_kpts should be equal from both images
        assert matched_kpts0.shape == matched_kpts1.shape, f"{matched_kpts0.shape} != {matched_kpts1.shape}"
        # Descriptors should have shape (N x D)
        assert all_desc0.ndim == 2, str(all_desc0.shape)
        assert all_desc1.ndim == 2, str(all_desc1.shape)
        # Some models return no descriptors. If there are descriptors, there should be as many keypoints as descriptors.
        if all_desc0.shape[0] != 0:
            assert all_desc0.shape[0] == all_kpts0.shape[0], f"{all_desc0.shape[0]} != {all_kpts0.shape[0]}"
        if all_desc1.shape[0] != 0:
            assert all_desc1.shape[0] == all_kpts1.shape[0], f"{all_desc1.shape[0]} != {all_kpts1.shape[0]}"


class EnsembleMatcher(BaseMatcher):
    def __init__(self, matcher_names: list[str] = [], device: str = "cpu", **kwargs):
        from vismatch import get_matcher

        super().__init__(device, **kwargs)
        self.matchers = [get_matcher(name, device=device, **kwargs) for name in matcher_names]

    def _forward(self, img0: torch.Tensor, img1: torch.Tensor) -> tuple[np.ndarray, np.ndarray, None, None, None, None]:
        all_matched_kpts0, all_matched_kpts1 = [], []
        for matcher in self.matchers:
            matched_kpts0, matched_kpts1, _, _, _, _ = matcher._forward(img0, img1)
            all_matched_kpts0.append(to_numpy(matched_kpts0))
            all_matched_kpts1.append(to_numpy(matched_kpts1))
        all_matched_kpts0, all_matched_kpts1 = np.concatenate(all_matched_kpts0), np.concatenate(all_matched_kpts1)
        return all_matched_kpts0, all_matched_kpts1, None, None, None, None
