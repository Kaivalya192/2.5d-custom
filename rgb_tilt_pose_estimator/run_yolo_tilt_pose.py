"""
Compatibility wrapper for the refactored tilt-pose apps.

Preferred entrypoints:
- rgb_tilt_pose_estimator/apps/run_tilt_pose_rs.py
- rgb_tilt_pose_estimator/apps/run_tilt_pose_oak.py
"""

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Re-export legacy symbols for existing imports in o3d/stl scripts.
from rgb_tilt_pose_estimator.common.masks import make_mask_from_polygon, overlap_ratio, suppress_overlaps
from rgb_tilt_pose_estimator.common.roi import load_roi_yaml
from rgb_tilt_pose_estimator.common.tracker import (
    PoseTracker,
    TrackState,
    normal_to_pose,
    random_color,
)


def _extract_source(argv):
    src = "rs"
    for i, a in enumerate(argv):
        if a == "--source" and (i + 1) < len(argv):
            src = str(argv[i + 1]).strip().lower()
            break
    return src


def main() -> None:
    src = _extract_source(sys.argv[1:])
    if src == "oak":
        from rgb_tilt_pose_estimator.apps.run_tilt_pose_oak import main as app_main
    else:
        from rgb_tilt_pose_estimator.apps.run_tilt_pose_rs import main as app_main
    app_main()


if __name__ == "__main__":
    main()

