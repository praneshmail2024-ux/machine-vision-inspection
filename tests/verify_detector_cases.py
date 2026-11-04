import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ring_detector import RingDetection, RingDetector


def has_detection_near(
    detections: list, center: tuple[int, int], tolerance: float = 12.0
) -> bool:
    return any(
        float(
            np.hypot(
                detection.center[0] - center[0],
                detection.center[1] - center[1],
            )
        )
        <= tolerance
        for detection in detections
    )


def synthetic_ring_detection(center: tuple[int, int]) -> RingDetection:
    contour = cv2.ellipse2Poly(
        center, (32, 27), 0, 0, 360, 5
    ).reshape(-1, 1, 2)
    return RingDetection(
        center,
        contour,
        contour,
        (
            (float(center[0]), float(center[1])),
            (64.0, 54.0),
            0.0,
        ),
        source="primary",
        confidence=0.95,
    )


def direct_exit_path_count(x_positions: tuple[int, ...]) -> int:
    detector = RingDetector(count_mode="exit")
    detector.exit_system_armed = True
    for center_y in range(450, 29, -6):
        detector._update_tracks(
            [
                synthetic_ring_detection((center_x, center_y))
                for center_x in x_positions
            ],
            gate_y=240,
            diagonal=800.0,
            frame_height=480,
        )
    detector.exit_motion_grace_frames = 5
    detector._update_tracks([], 240, 800.0, 480)
    detector._update_tracks([], 240, 800.0, 480)
    return detector.track_crossing_count


def direct_paused_top_count() -> int:
    detector = RingDetector(count_mode="exit")
    detector.exit_system_armed = True
    for center_y in range(450, 49, -6):
        detector._update_tracks(
            [synthetic_ring_detection((320, center_y))],
            gate_y=240,
            diagonal=800.0,
            frame_height=480,
        )
    # The ring has completed a real directed line crossing. A later stationary
    # top-edge contour flicker must not add another count.
    detector.exit_motion_grace_frames = 5
    detector._update_tracks([], 240, 800.0, 480)
    detector._update_tracks([], 240, 800.0, 480)
    return detector.track_crossing_count


def touching_rings_frame() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    for center in ((270, 190), (350, 190), (270, 270), (350, 270)):
        cv2.ellipse(frame, center, (48, 42), 0, 0, 360, cardboard, -1)
        cv2.ellipse(frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1)
    return frame


def seam_frame(y: int = 235) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (130, y), (510, y + 12), (52, 52, 52), -1)
    cv2.line(frame, (130, y + 3), (510, y + 3), (78, 78, 78), 2)
    return frame


def neutral_belt_circle_frame() -> np.ndarray:
    """A grey circular belt mark: geometrically ring-like but not cardboard."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(frame, (300, 260), (48, 40), 0, 0, 360, (76, 76, 76), 12)
    return frame


def shadow_background_and_frame(
    center: tuple[int, int] = (300, 260),
) -> tuple[np.ndarray, np.ndarray]:
    """A soft dark oval caused only by illumination, not cardboard."""
    background = np.full((480, 640, 3), 70, dtype=np.uint8)
    frame = background.copy()
    cv2.ellipse(frame, center, (50, 42), 0, 0, 360, (48, 48, 48), -1)
    frame = cv2.GaussianBlur(frame, (17, 17), 0)
    return background, frame


def annular_rings_frame(centers: tuple[tuple[int, int], ...]) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    for center in centers:
        cv2.ellipse(frame, center, (48, 42), 0, 0, 360, cardboard, -1)
        cv2.ellipse(frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1)
    return frame


def flat_ring_in_cluster_frame() -> np.ndarray:
    """A true side-profile ring surrounded by four normal rings."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    rings = (
        ((320, 240), (42, 24), (30, 12)),
        ((320, 174), (48, 42), (33, 27)),
        ((320, 306), (48, 42), (33, 27)),
        ((252, 240), (48, 42), (33, 27)),
        ((388, 240), (48, 42), (33, 27)),
    )
    for center, outer_axes, inner_axes in rings:
        cv2.ellipse(frame, center, outer_axes, 0, 0, 360, cardboard, -1)
        cv2.ellipse(frame, center, inner_axes, 0, 0, 360, (12, 12, 12), -1)
    return frame


def solid_side_profile_frame() -> np.ndarray:
    """A ring with no visible hole, touching an ordinary ring at one end."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    for center in ((220, 245), (220, 335)):
        cv2.ellipse(frame, center, (48, 42), 0, 0, 360, cardboard, -1)
        cv2.ellipse(frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1)
    box = cv2.boxPoints(((310, 290), (82, 28), -8))
    cv2.fillConvexPoly(frame, np.int32(np.round(box)), cardboard)
    return frame


def mixed_orientation_frame() -> np.ndarray:
    """Ten rings: four open ovals plus six solid side-profile bands."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    open_rings = (
        ((150, 135), (47, 39), (31, 24), -12),
        ((300, 135), (49, 38), (33, 23), 14),
        ((455, 135), (46, 40), (30, 26), -7),
        ((155, 340), (49, 37), (33, 22), 21),
    )
    for center, outer_axes, inner_axes, angle in open_rings:
        cv2.ellipse(frame, center, outer_axes, angle, 0, 360, cardboard, -1)
        cv2.ellipse(frame, center, inner_axes, angle, 0, 360, (12, 12, 12), -1)

    side_profiles = (
        ((300, 245), (84, 28), 0),
        ((455, 245), (82, 27), 18),
        ((300, 345), (86, 26), -26),
        ((430, 355), (82, 27), 56),
        ((535, 335), (78, 25), 88),
        ((115, 245), (80, 27), -62),
    )
    for center, size, angle in side_profiles:
        box = cv2.boxPoints((center, size, angle))
        cv2.fillConvexPoly(frame, np.int32(np.round(box)), cardboard)
    return frame


def dense_mixed_orientation_frame() -> np.ndarray:
    """Screenshot-like pile with nearby open, tilted and fully side-on rings."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    open_rings = (
        ((215, 100), (47, 36), (31, 22), 20),
        ((180, 300), (40, 28), (25, 15), -10),
        ((390, 270), (52, 44), (35, 28), 0),
        ((240, 390), (48, 34), (31, 20), 5),
    )
    for center, outer_axes, inner_axes, angle in open_rings:
        cv2.ellipse(frame, center, outer_axes, angle, 0, 360, cardboard, -1)
        cv2.ellipse(frame, center, inner_axes, angle, 0, 360, (12, 12, 12), -1)

    side_profiles = (
        ((205, 165), (80, 26), -30),
        ((285, 195), (82, 26), 55),
        ((365, 135), (82, 28), 88),
        ((445, 170), (82, 28), 88),
        ((210, 235), (84, 26), 0),
        ((315, 350), (84, 27), -42),
    )
    for center, size, angle in side_profiles:
        box = cv2.boxPoints((center, size, angle))
        cv2.fillConvexPoly(frame, np.int32(np.round(box)), cardboard)
    return frame


def side_wall_pair_frame(connected: bool) -> np.ndarray:
    """One open U-shaped side view, or two independent parallel side bands."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    center = np.array((320.0, 250.0), dtype=np.float32)
    angle = 34.0
    radians = np.deg2rad(angle)
    direction = np.array((np.cos(radians), np.sin(radians)), dtype=np.float32)
    normal = np.array((-direction[1], direction[0]), dtype=np.float32)
    for offset in (-25.0, 25.0):
        wall_center = tuple((center + normal * offset).tolist())
        wall = cv2.boxPoints((wall_center, (92, 18), angle))
        cv2.fillConvexPoly(frame, np.int32(np.round(wall)), cardboard)
    if connected:
        bridge_center = tuple((center - direction * 39.0).tolist())
        bridge = cv2.boxPoints((bridge_center, (20, 68), angle))
        cv2.fillConvexPoly(frame, np.int32(np.round(bridge)), cardboard)
    return frame


def bright_occluded_hole_frame() -> np.ndarray:
    """One open ring whose hole is crossed by a bright side-on second ring."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    cv2.ellipse(frame, (315, 240), (52, 38), 12, 0, 360, cardboard, -1)
    cv2.ellipse(frame, (315, 240), (34, 22), 12, 0, 360, (12, 12, 12), -1)
    crossing_wall = cv2.boxPoints(((300, 240), (72, 16), -30))
    cv2.fillConvexPoly(
        frame, np.int32(np.round(crossing_wall)), cardboard
    )
    return frame


def hidden_crossing_side_profile_frame() -> np.ndarray:
    """Six open rings plus a side-on seventh band crossing one ring rim."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    for center in (
        (120, 100),
        (300, 100),
        (480, 100),
        (120, 340),
        (300, 340),
        (480, 340),
    ):
        cv2.ellipse(frame, center, (48, 42), 0, 0, 360, cardboard, -1)
        cv2.ellipse(frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1)
    crossing_band = cv2.boxPoints(((350, 100), (84, 26), 0))
    cv2.fillConvexPoly(
        frame, np.int32(np.round(crossing_band)), cardboard
    )
    return frame


def illuminated_rim_fragment_frame(center_y: int = 240) -> np.ndarray:
    """One ring whose bright tangent wall must not become a second flat ring."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(
        frame, (320, center_y), (43, 51), 11, 0, 360, (65, 65, 65), -1
    )
    cv2.ellipse(
        frame, (320, center_y), (25, 36), 11, 0, 360, (12, 12, 12), -1
    )
    bright_rim = cv2.boxPoints(((291, center_y - 14), (63, 19), -63))
    cv2.fillConvexPoly(
        frame, np.int32(np.round(bright_rim)), (125, 175, 215)
    )
    return frame


def interstitial_gap_frame(real_center: bool = False) -> np.ndarray:
    """Four rings forming a false pocket, optionally with a true centre ring."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    centers = ((262, 240), (378, 240), (320, 178), (320, 302))
    for center in centers:
        cv2.ellipse(frame, center, (48, 42), 0, 0, 360, cardboard, -1)
    for center in centers:
        cv2.ellipse(frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1)
    if real_center:
        cv2.ellipse(frame, (320, 240), (48, 42), 0, 0, 360, cardboard, -1)
        cv2.ellipse(frame, (320, 240), (33, 27), 0, 0, 360, (12, 12, 12), -1)
    return frame


def dense_touching_columns_frame(shift_y: int = 0) -> np.ndarray:
    """Ten touching rings in two columns, without four false bridge objects."""
    centers = tuple(
        (x, y + shift_y)
        for y in (550, 625, 700, 775, 850)
        for x in (285, 355)
    )
    return annular_rings_frame(centers)


def dense_close_twenty_three_frame(shift_y: int = 0) -> np.ndarray:
    """Twenty-three close/partly covered RGB rings in a compact moving batch."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    horizontal_spacing = 62
    vertical_spacing = 58
    row_sizes = (5, 5, 5, 4, 4)
    for row, row_size in enumerate(row_sizes):
        row_offset = (5 - row_size) * horizontal_spacing / 2
        for column in range(row_size):
            center = (
                int(160 + row_offset + column * horizontal_spacing),
                int(560 + row * vertical_spacing + shift_y),
            )
            # Draw in camera-depth order. A later ring may cover part of an
            # earlier rim/hole, matching the partial occlusion seen on the belt.
            cv2.ellipse(
                frame, center, (48, 42), 0, 0, 360, cardboard, -1
            )
            cv2.ellipse(
                frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1
            )
    return frame


def hidden_dense_crossing_frame(center_y: int) -> np.ndarray:
    """Four proven rings; two become hidden while their group crosses the line."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    for index, center_x in enumerate((190, 280, 370, 460)):
        # The middle rings remain physically in the moving batch but become
        # completely occluded after they have formed stable, armed identities.
        if index in (1, 2) and center_y <= 270:
            continue
        center = (center_x, center_y)
        cv2.ellipse(frame, center, (48, 42), 0, 0, 360, cardboard, -1)
        cv2.ellipse(
            frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1
        )
    return frame


def temporarily_hidden_exit_frame(center_y: int) -> np.ndarray:
    """Four entering rings; two merge out of view, then separate before exit."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    for index, center_x in enumerate((190, 280, 370, 460)):
        if index in (1, 2) and 180 < center_y <= 270:
            continue
        center = (center_x, center_y)
        cv2.ellipse(frame, center, (48, 42), 0, 0, 360, cardboard, -1)
        cv2.ellipse(
            frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1
        )
    return frame


def main() -> None:
    empty = np.zeros((480, 640, 3), dtype=np.uint8)

    visible_stability_detector = RingDetector()
    oscillating_visible_values = [
        visible_stability_detector._stabilize_visible_count(value, True)
        for value in (15, 12, 14, 13, 12, 14, 13, 12)
    ]
    empty_visible_value = (
        visible_stability_detector._stabilize_visible_count(0, False)
    )

    overlay_stability_detector = RingDetector()
    smoothed_overlay_geometry = []
    for frame_index, (offset_x, offset_y, axis_delta, angle_delta) in enumerate(
        (
            (0, 0, 0, 0),
            (2, -1, 3, 4),
            (-1, 2, -2, -3),
            (1, -2, 4, 3),
            (-2, 1, -3, -4),
            (2, 0, 2, 2),
            (0, -1, 0, -2),
        ),
        start=1,
    ):
        center = (320 + offset_x, 240 + offset_y)
        axes = (41 + axis_delta, 48 + axis_delta)
        contour = cv2.ellipse2Poly(center, axes, 11 + angle_delta, 0, 360, 5)
        contour = contour.reshape((-1, 1, 2))
        detection = RingDetection(
            center,
            contour,
            contour,
            (
                (float(center[0]), float(center[1])),
                (float(axes[0] * 2), float(axes[1] * 2)),
                float(11 + angle_delta),
            ),
        )
        overlay_stability_detector._tracking_frame_index = frame_index
        smoothed_overlay_geometry.append(
            overlay_stability_detector._smoothed_overlay_ellipse(1, detection)
        )
    overlay_center_span_x = max(
        geometry[0][0] for geometry in smoothed_overlay_geometry
    ) - min(geometry[0][0] for geometry in smoothed_overlay_geometry)
    overlay_center_span_y = max(
        geometry[0][1] for geometry in smoothed_overlay_geometry
    ) - min(geometry[0][1] for geometry in smoothed_overlay_geometry)
    overlay_angle_span = max(
        geometry[2] for geometry in smoothed_overlay_geometry
    ) - min(geometry[2] for geometry in smoothed_overlay_geometry)

    overlay_identity_detector = RingDetector()
    identity_detections = []
    for center in ((220, 240), (320, 240)):
        contour = cv2.ellipse2Poly(center, (42, 48), 0, 0, 360, 5)
        contour = contour.reshape((-1, 1, 2))
        identity_detections.append(
            RingDetection(
                center,
                contour,
                contour,
                (
                    (float(center[0]), float(center[1])),
                    (84.0, 96.0),
                    0.0,
                ),
            )
        )
    overlay_identity_detector._tracking_frame_index = 1
    first_overlay_assignments = overlay_identity_detector._assign_overlay_ids(
        identity_detections
    )
    for overlay_id, detection in first_overlay_assignments:
        overlay_identity_detector._smoothed_overlay_ellipse(
            overlay_id, detection
        )
    overlay_identity_detector._tracking_frame_index = 2
    reversed_identity_detections = [
        RingDetection(
            (322, 240),
            identity_detections[1].inner_contour,
            identity_detections[1].outer_contour,
            ((322.0, 240.0), (84.0, 96.0), 0.0),
        ),
        RingDetection(
            (218, 240),
            identity_detections[0].inner_contour,
            identity_detections[0].outer_contour,
            ((218.0, 240.0), (84.0, 96.0), 0.0),
        ),
    ]
    reversed_overlay_ids = [
        overlay_id
        for overlay_id, _ in overlay_identity_detector._assign_overlay_ids(
            reversed_identity_detections
        )
    ]

    split_hole_detections = []
    for center, size, angle in (
        ((252, 349), (40.3, 64.9), 1.5),
        ((278, 373), (23.5, 66.1), 20.9),
    ):
        contour = cv2.ellipse2Poly(
            center,
            (round(size[0] / 2), round(size[1] / 2)),
            round(angle),
            0,
            360,
            5,
        ).reshape((-1, 1, 2))
        split_hole_detections.append(
            RingDetection(
                center,
                contour,
                None,
                (
                    (float(center[0]), float(center[1])),
                    size,
                    angle,
                ),
            )
        )
    dark_hole_image = np.full((480, 640), 52, dtype=np.uint8)
    split_hole_kept = RingDetector._suppress_split_hole_duplicates(
        split_hole_detections, dark_hole_image
    )
    cardboard_divided_image = dark_hole_image.copy()
    cv2.line(
        cardboard_divided_image,
        split_hole_detections[0].center,
        split_hole_detections[1].center,
        180,
        9,
        cv2.LINE_AA,
    )
    touching_pair_kept = RingDetector._suppress_split_hole_duplicates(
        split_hole_detections, cardboard_divided_image
    )

    composite_members = [
        synthetic_ring_detection((270, 190)),
        synthetic_ring_detection((270, 270)),
    ]
    composite_contour = cv2.ellipse2Poly(
        (270, 230), (39, 85), 0, 0, 360, 5
    ).reshape((-1, 1, 2))
    multi_ring_composite = RingDetection(
        (270, 230),
        composite_contour,
        composite_contour,
        ((270.0, 230.0), (78.0, 170.0), 0.0),
        fallback=True,
        source="flat",
        confidence=0.90,
    )
    multi_ring_composite_rejected = (
        RingDetector._is_multi_ring_composite(
            multi_ring_composite,
            [*composite_members, multi_ring_composite],
        )
    )
    isolated_profile_rejected = RingDetector._is_multi_ring_composite(
        multi_ring_composite,
        [composite_members[0], multi_ring_composite],
    )

    rings_detector = RingDetector()
    rings_detector.calibrate_background(empty)
    rings, _ = rings_detector.detect(touching_rings_frame())

    green_only_detector = RingDetector()
    green_only_detector.calibrate_background(empty)
    green_only_frame = None
    for _ in range(6):
        green_only_frame, _ = green_only_detector.process(
            touching_rings_frame()
        )
    assert green_only_frame is not None
    green_outline_pixels = int(
        np.count_nonzero(
            np.all(green_only_frame == np.array((0, 255, 0)), axis=2)
        )
    )
    yellow_outline_pixels = int(
        np.count_nonzero(
            np.all(green_only_frame == np.array((0, 215, 255)), axis=2)
        )
    )
    cyan_marker_pixels = int(
        np.count_nonzero(
            np.all(green_only_frame == np.array((255, 255, 0)), axis=2)
        )
    )

    transient_display_detector = RingDetector()
    transient_display_detector.calibrate_background(empty)
    transient_display_values = [
        transient_display_detector.process(
            annular_rings_frame(((320, 240),))
        )[1]["visible"]
        for _ in range(3)
    ]

    moving_eleven_detector = RingDetector()
    moving_eleven_detector.calibrate_background(empty)
    moving_eleven_status = None
    moving_eleven_centers = (
        (145, 130),
        (265, 130),
        (385, 130),
        (505, 130),
        (145, 250),
        (265, 250),
        (385, 250),
        (505, 250),
        (205, 370),
        (325, 370),
        (445, 370),
    )
    for shift_y in range(0, -61, -6):
        moving_eleven_status = moving_eleven_detector.process(
            annular_rings_frame(
                tuple((x, y + shift_y) for x, y in moving_eleven_centers)
            )
        )[1]

    seam_detector = RingDetector()
    seam_detector.calibrate_background(empty)
    seam, _ = seam_detector.detect(seam_frame())

    neutral_belt_circle_detector = RingDetector()
    neutral_belt_circle_detector.calibrate_background(empty)
    neutral_belt_circle, _ = neutral_belt_circle_detector.detect(
        neutral_belt_circle_frame()
    )

    shadow_background, shadow_frame = shadow_background_and_frame()
    shadow_detector = RingDetector()
    shadow_detector.calibrate_background(shadow_background)
    shadow_detections, _ = shadow_detector.detect(shadow_frame)

    moving_shadow_detector = RingDetector(count_mode="exit")
    moving_shadow_detector.calibrate_background(shadow_background)
    moving_shadow_detector.arm_counting()
    moving_shadow_status = None
    for center_y in range(450, 149, -6):
        _, moving_shadow_frame = shadow_background_and_frame((300, center_y))
        _, moving_shadow_status = moving_shadow_detector.process(
            moving_shadow_frame
        )

    flat_detector = RingDetector()
    flat_detector.calibrate_background(empty)
    flat_cluster, _ = flat_detector.detect(flat_ring_in_cluster_frame())

    solid_detector = RingDetector()
    solid_detector.calibrate_background(empty)
    solid_profile, _ = solid_detector.detect(solid_side_profile_frame())

    mixed_detector = RingDetector()
    mixed_detector.calibrate_background(empty)
    mixed_orientation, _ = mixed_detector.detect(mixed_orientation_frame())

    dense_mixed_detector = RingDetector()
    dense_mixed_detector.calibrate_background(empty)
    dense_mixed_orientation, _ = dense_mixed_detector.detect(
        dense_mixed_orientation_frame()
    )
    dense_mixed_status = None
    for _ in range(18):
        _, dense_mixed_status = dense_mixed_detector.process(
            dense_mixed_orientation_frame()
        )

    connected_side_detector = RingDetector()
    connected_side_detector.calibrate_background(empty)
    connected_side_walls, _ = connected_side_detector.detect(
        side_wall_pair_frame(connected=True)
    )

    independent_side_detector = RingDetector()
    independent_side_detector.calibrate_background(empty)
    independent_side_walls, _ = independent_side_detector.detect(
        side_wall_pair_frame(connected=False)
    )

    occluded_hole_detector = RingDetector()
    occluded_hole_detector.calibrate_background(empty)
    bright_occluded_hole, _ = occluded_hole_detector.detect(
        bright_occluded_hole_frame()
    )

    hidden_flat_detector = RingDetector()
    hidden_flat_detector.calibrate_background(empty)
    hidden_crossing_side_profile, _ = hidden_flat_detector.detect(
        hidden_crossing_side_profile_frame()
    )

    illuminated_rim_detector = RingDetector()
    illuminated_rim_detector.calibrate_background(empty)
    illuminated_rim_fragment, _ = illuminated_rim_detector.detect(
        illuminated_rim_fragment_frame()
    )

    moving_illuminated_rim_detector = RingDetector()
    moving_illuminated_rim_detector.calibrate_background(empty)
    moving_illuminated_rim_status = None
    for y in range(330, 149, -6):
        _, moving_illuminated_rim_status = moving_illuminated_rim_detector.process(
            illuminated_rim_fragment_frame(y)
        )

    interstitial_gap_detector = RingDetector()
    interstitial_gap_detector.calibrate_background(empty)
    interstitial_gap, _ = interstitial_gap_detector.detect(
        interstitial_gap_frame()
    )

    real_center_detector = RingDetector()
    real_center_detector.calibrate_background(empty)
    surrounded_real_center, _ = real_center_detector.detect(
        interstitial_gap_frame(real_center=True)
    )

    dense_columns_detector = RingDetector()
    dense_columns_detector.calibrate_background(empty)
    dense_touching_columns, _ = dense_columns_detector.detect(
        dense_touching_columns_frame(shift_y=-470)
    )

    moving_dense_columns_detector = RingDetector()
    moving_dense_columns_detector.calibrate_background(empty)
    moving_dense_columns_status = None
    for shift_y in range(0, -1001, -10):
        _, moving_dense_columns_status = moving_dense_columns_detector.process(
            dense_touching_columns_frame(shift_y)
        )

    dense_twenty_three_detector = RingDetector()
    dense_twenty_three_detector.calibrate_background(empty)
    dense_twenty_three, _ = dense_twenty_three_detector.detect(
        dense_close_twenty_three_frame(shift_y=-470)
    )

    moving_twenty_three_detector = RingDetector(count_mode="exit")
    moving_twenty_three_detector.calibrate_background(empty)
    moving_twenty_three_status = None
    for shift_y in range(0, -1001, -8):
        _, moving_twenty_three_status = moving_twenty_three_detector.process(
            dense_close_twenty_three_frame(shift_y)
        )

    hidden_crossing_detector = RingDetector()
    hidden_crossing_detector.calibrate_background(empty)
    hidden_crossing_status = None
    for center_y in range(350, 149, -6):
        _, hidden_crossing_status = hidden_crossing_detector.process(
            hidden_dense_crossing_frame(center_y)
        )

    exit_occlusion_detector = RingDetector(count_mode="exit")
    exit_occlusion_detector.calibrate_background(empty)
    exit_occlusion_status = None
    for center_y in range(510, -101, -6):
        _, exit_occlusion_status = exit_occlusion_detector.process(
            temporarily_hidden_exit_frame(center_y)
        )

    exit_midfield_detector = RingDetector(count_mode="exit")
    exit_midfield_detector.calibrate_background(empty)
    exit_midfield_status = None
    for center_y in range(330, -101, -6):
        _, exit_midfield_status = exit_midfield_detector.process(
            annular_rings_frame(((320, center_y),))
        )

    exit_line_only_detector = RingDetector(count_mode="exit")
    exit_line_only_detector.calibrate_background(empty)
    exit_line_only_status = None
    for center_y in range(450, 149, -6):
        _, exit_line_only_status = exit_line_only_detector.process(
            annular_rings_frame(((320, center_y),))
        )

    # Once an empty FOV has armed production counting, a single ring should
    # increment at the red line instead of waiting to disappear at the far
    # edge. This is the live conveyor case that previously remained at zero.
    armed_single_line_detector = RingDetector(count_mode="exit")
    armed_single_line_detector.calibrate_background(empty)
    for _ in range(5):
        armed_single_line_detector.process(empty)
    armed_single_line_status = None
    for center_y in range(450, 149, -6):
        _, armed_single_line_status = armed_single_line_detector.process(
            annular_rings_frame(((320, center_y),))
        )

    saved_background_start_detector = RingDetector(count_mode="exit")
    saved_background_start_detector.calibrate_background(empty)
    saved_background_start_detector.arm_counting()
    saved_background_start_status = None
    for center_y in range(450, 149, -6):
        _, saved_background_start_status = saved_background_start_detector.process(
            annular_rings_frame(((320, center_y),))
        )

    uncleared_entry_detector = RingDetector(count_mode="exit")
    uncleared_entry_detector.calibrate_background(empty)
    uncleared_entry_status = None
    for center_y in range(450, -101, -6):
        _, uncleared_entry_status = uncleared_entry_detector.process(
            annular_rings_frame(((320, center_y),))
        )

    reset_present_detector = RingDetector(count_mode="exit")
    reset_present_detector.calibrate_background(empty)
    for _ in range(5):
        reset_present_detector.process(empty)
    for center_y in range(450, 419, -6):
        reset_present_detector.process(
            annular_rings_frame(((320, center_y),))
        )
    reset_present_detector.reset()
    reset_present_detector.calibrate_background(empty)
    reset_present_status = None
    for center_y in range(414, -101, -6):
        _, reset_present_status = reset_present_detector.process(
            annular_rings_frame(((320, center_y),))
        )

    late_recovery_detector = RingDetector(count_mode="exit")
    late_recovery_detector.calibrate_background(empty)
    for _ in range(5):
        late_recovery_detector.process(empty)
    late_recovery_status = None
    for center_y in range(300, -101, -6):
        _, late_recovery_status = late_recovery_detector.process(
            annular_rings_frame(((320, center_y),))
        )

    # A ring that was already counted at the line can briefly lose its
    # tracking identity in a touching train. If it is reacquired above the
    # line, that new identity is not a fresh conveyor entry and must not add a
    # second count when it later leaves the top of the FOV.
    recreated_above_line_detector = RingDetector(count_mode="exit")
    recreated_above_line_detector.calibrate_background(empty)
    for _ in range(5):
        recreated_above_line_detector.process(empty)
    recreated_above_line_status = None
    for center_y in range(330, 179, -6):
        _, recreated_above_line_status = (
            recreated_above_line_detector.process(
                annular_rings_frame(((320, center_y),))
            )
        )
    for _ in range(27):
        _, recreated_above_line_status = (
            recreated_above_line_detector.process(empty)
        )
    for center_y in range(180, -101, -6):
        _, recreated_above_line_status = (
            recreated_above_line_detector.process(
                annular_rings_frame(((320, center_y),))
            )
        )

    duplicate_exit_total = direct_exit_path_count((320, 338))
    close_pair_exit_total = direct_exit_path_count((300, 338))
    paused_top_total = direct_paused_top_count()

    vertical_train_detector = RingDetector(count_mode="exit")
    vertical_train_detector.calibrate_background(empty)
    vertical_train_status = None
    for center_y in range(520, -181, -8):
        _, vertical_train_status = vertical_train_detector.process(
            annular_rings_frame(
                ((320, center_y), (320, center_y + 70))
            )
        )

    recorded_interstitial_frame = cv2.imread(
        str(
            PROJECT_ROOT
            / "data"
            / "samples"
            / "interstitial_false_eight_rings.jpg"
        )
    )
    recorded_interstitial_background = cv2.imread(
        str(
            PROJECT_ROOT
            / "data"
            / "samples"
            / "interstitial_false_empty_belt.jpg"
        )
    )
    if (
        recorded_interstitial_frame is None
        or recorded_interstitial_background is None
    ):
        raise SystemExit("Interstitial-gap regression images are missing.")
    recorded_interstitial_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    recorded_interstitial_detector.calibrate_background(
        recorded_interstitial_background
    )
    recorded_interstitial, _ = recorded_interstitial_detector.detect(
        recorded_interstitial_frame
    )
    recorded_interstitial_status = None
    for _ in range(18):
        _, recorded_interstitial_status = recorded_interstitial_detector.process(
            recorded_interstitial_frame
        )

    tangent_fragments_frame = cv2.imread(
        str(
            PROJECT_ROOT
            / "data"
            / "samples"
            / "tangent_fragments_eight_rings.jpg"
        )
    )
    if tangent_fragments_frame is None:
        raise SystemExit("Tangent-fragment regression image is missing.")
    tangent_fragments_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    tangent_fragments_detector.calibrate_background(
        recorded_interstitial_background
    )
    tangent_fragments, _ = tangent_fragments_detector.detect(
        tangent_fragments_frame
    )

    tangent_fragments_ten_frame = cv2.imread(
        str(
            PROJECT_ROOT
            / "data"
            / "samples"
            / "tangent_fragments_ten_rings.jpg"
        )
    )
    tangent_fragments_background = cv2.imread(
        str(
            PROJECT_ROOT
            / "data"
            / "samples"
            / "tangent_fragments_empty_belt.jpg"
        )
    )
    if (
        tangent_fragments_ten_frame is None
        or tangent_fragments_background is None
    ):
        raise SystemExit("Ten-ring tangent-fragment regression images are missing.")
    tangent_fragments_ten_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    tangent_fragments_ten_detector.calibrate_background(
        tangent_fragments_background
    )
    tangent_fragments_ten, _ = tangent_fragments_ten_detector.detect(
        tangent_fragments_ten_frame
    )

    edge_pocket_frame = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "edge_pocket_ten_rings.jpg")
    )
    edge_pocket_background = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "edge_pocket_empty_belt.jpg")
    )
    if edge_pocket_frame is None or edge_pocket_background is None:
        raise SystemExit("Edge-pocket regression images are missing.")
    edge_pocket_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    edge_pocket_detector.calibrate_background(edge_pocket_background)
    edge_pocket_rings, _ = edge_pocket_detector.detect(edge_pocket_frame)

    boundary_hough_frame = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "boundary_hough_ten_rings.jpg")
    )
    boundary_hough_background = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "boundary_hough_empty_belt.jpg")
    )
    if boundary_hough_frame is None or boundary_hough_background is None:
        raise SystemExit("Boundary-Hough regression images are missing.")
    boundary_hough_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    boundary_hough_detector.calibrate_background(boundary_hough_background)
    boundary_hough_rings, _ = boundary_hough_detector.detect(
        boundary_hough_frame
    )

    dense_identity_ten_frame = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "dense_identity_ten_rings.jpg")
    )
    dense_identity_eight_frame = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "dense_identity_eight_rings.jpg")
    )
    if dense_identity_ten_frame is None or dense_identity_eight_frame is None:
        raise SystemExit("Dense identity regression images are missing.")
    dense_identity_ten_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    dense_identity_ten_detector.calibrate_background(boundary_hough_background)
    dense_identity_ten, _ = dense_identity_ten_detector.detect(
        dense_identity_ten_frame
    )
    dense_identity_eight_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    dense_identity_eight_detector.calibrate_background(
        boundary_hough_background
    )
    dense_identity_eight, _ = dense_identity_eight_detector.detect(
        dense_identity_eight_frame
    )

    live_dense_fragment_directory = (
        PROJECT_ROOT / "data" / "samples" / "live_dense_fragment_case"
    )
    live_dense_fragment_background = cv2.imread(
        str(live_dense_fragment_directory / "empty_belt.jpg")
    )
    live_dense_fragment_frames = [
        cv2.imread(str(live_dense_fragment_directory / f"frame_{index}.jpg"))
        for index in range(8)
    ]
    live_dense_pocket_edge_frame = cv2.imread(
        str(live_dense_fragment_directory / "frame_pocket_edge.jpg")
    )
    if live_dense_fragment_background is None or any(
        frame is None for frame in live_dense_fragment_frames
    ) or live_dense_pocket_edge_frame is None:
        raise SystemExit("Live dense-fragment regression images are missing.")
    live_dense_fragment_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    live_dense_fragment_detector.calibrate_background(
        live_dense_fragment_background
    )
    live_dense_fragment_raw_counts: list[int] = []
    live_dense_fragment_status = None
    for frame in live_dense_fragment_frames:
        frame_detections, _ = live_dense_fragment_detector.detect(frame)
        live_dense_fragment_raw_counts.append(len(frame_detections))
        _, live_dense_fragment_status = live_dense_fragment_detector.process(frame)
    live_dense_pocket_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    live_dense_pocket_detector.calibrate_background(
        live_dense_fragment_background
    )
    live_dense_pocket_detections, _ = live_dense_pocket_detector.detect(
        live_dense_pocket_edge_frame
    )

    belt_background = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "belt_join_negative.jpg")
    )
    if belt_background is None:
        raise SystemExit("Belt-joint regression images are missing.")
    # Keep the stable recorded empty frame as the reference, then add the same
    # real belt-joint texture at a second position. This regression must not
    # depend on the operator-replaceable live calibration file.
    belt_join_frame = belt_background.copy()
    seam_patch = belt_background[270:322, 90:397]
    belt_join_frame[190:242, 90:397] = seam_patch
    belt_join_detector = RingDetector(roi_left=0.14, roi_top=0.03, roi_right=0.62, roi_bottom=0.97)
    belt_join_detector.calibrate_background(belt_background)
    belt_join, _ = belt_join_detector.detect(belt_join_frame)

    overlap_detector = RingDetector()
    overlap_detector.calibrate_background(empty)
    partial_overlap, _ = overlap_detector.detect(
        annular_rings_frame(((302, 240), (338, 240)))
    )

    chain_detector = RingDetector()
    chain_detector.calibrate_background(empty)
    touching_chain, _ = chain_detector.detect(
        annular_rings_frame(((272, 240), (320, 240), (368, 240)))
    )

    coincident_detector = RingDetector()
    coincident_detector.calibrate_background(empty)
    fully_coincident, _ = coincident_detector.detect(
        annular_rings_frame(((320, 240), (320, 240)))
    )

    # A static object on the line and an object travelling in the wrong
    # direction must never increment the production counter.
    static_detector = RingDetector()
    static_detector.calibrate_background(empty)
    static_statuses = [
        static_detector.process(annular_rings_frame(((320, 240),)))[1]
        for _ in range(4)
    ]

    jitter_detector = RingDetector()
    jitter_detector.calibrate_background(empty)
    jitter_status = None
    for y in (252, 248, 243, 238, 245, 236, 242, 239, 246, 237):
        _, jitter_status = jitter_detector.process(
            annular_rings_frame(((320, y),))
        )

    display_dropout_detector = RingDetector()
    display_dropout_detector.calibrate_background(empty)
    for _ in range(6):
        display_dropout_detector.process(annular_rings_frame(((320, 240),)))
    display_dropout_statuses = [
        display_dropout_detector.process(empty)[1] for _ in range(3)
    ]

    reverse_detector = RingDetector()
    reverse_detector.calibrate_background(empty)
    reverse_status = None
    for y in range(150, 331, 6):
        _, reverse_status = reverse_detector.process(annular_rings_frame(((320, y),)))

    moving_overlap_detector = RingDetector()
    moving_overlap_detector.calibrate_background(empty)
    moving_overlap_status = None
    for y in range(330, 149, -6):
        _, moving_overlap_status = moving_overlap_detector.process(
            annular_rings_frame(((302, y), (338, y)))
        )

    slow_motion_detector = RingDetector()
    slow_motion_detector.calibrate_background(empty)
    slow_motion_status = None
    for y in range(275, 204, -1):
        _, slow_motion_status = slow_motion_detector.process(
            annular_rings_frame(((320, y),))
        )

    repeated_belt_statuses = [
        belt_join_detector.process(belt_join_frame)[1] for _ in range(5)
    ]

    synthetic_seam_detector = RingDetector()
    synthetic_seam_detector.calibrate_background(empty)
    synthetic_seam_status = None
    for y in (310, 280, 250, 235, 220, 190):
        _, synthetic_seam_status = synthetic_seam_detector.process(seam_frame(y))

    real_sample_expectations = {
        "sample_round.png": 1,
        "sample_collapsed.png": 1,
        "sample_irregular.png": 1,
        "sample_overlap.png": 2,
    }
    real_sample_counts: dict[str, int] = {}
    for filename, expected in real_sample_expectations.items():
        sample = cv2.imread(str(PROJECT_ROOT / "data" / "samples" / filename))
        if sample is None:
            raise SystemExit(f"Missing detector sample: {filename}")
        sample_detector = RingDetector()
        sample_detections, _ = sample_detector.detect(sample)
        real_sample_counts[filename] = len(sample_detections)

    performance_frame = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "performance_ten_rings.jpg")
    )
    performance_background = cv2.imread(
        str(PROJECT_ROOT / "data" / "samples" / "performance_empty_belt.jpg")
    )
    if performance_frame is None or performance_background is None:
        raise SystemExit("Live ten-ring performance regression images are missing.")
    performance_detector = RingDetector(
        roi_left=0.14,
        roi_top=0.03,
        roi_right=0.62,
        roi_bottom=0.97,
    )
    performance_detector.calibrate_background(performance_background)
    performance_detections, _ = performance_detector.detect(performance_frame)
    performance_status = None
    # The dashboard intentionally waits for persistent evidence before changing
    # an already-published non-zero visible count.
    for _ in range(18):
        _, performance_status = performance_detector.process(performance_frame)

    print({
        "oscillating_visible_values": oscillating_visible_values,
        "empty_visible_value": empty_visible_value,
        "overlay_center_span": (
            round(overlay_center_span_x, 2),
            round(overlay_center_span_y, 2),
        ),
        "overlay_angle_span": round(overlay_angle_span, 2),
        "reversed_overlay_ids": reversed_overlay_ids,
        "split_hole_kept": len(split_hole_kept),
        "touching_pair_kept": len(touching_pair_kept),
        "multi_ring_composite_rejected": multi_ring_composite_rejected,
        "isolated_profile_rejected": isolated_profile_rejected,
        "green_outline_pixels": green_outline_pixels,
        "yellow_outline_pixels": yellow_outline_pixels,
        "cyan_marker_pixels": cyan_marker_pixels,
        "transient_display_values": transient_display_values,
        "moving_eleven_visible": (
            moving_eleven_status["visible"]
            if moving_eleven_status
            else -1
        ),
        "touching_rings": len(rings),
        "flat_ring_cluster": len(flat_cluster),
        "solid_side_profile": len(solid_profile),
        "mixed_orientation": len(mixed_orientation),
        "dense_mixed_orientation": len(dense_mixed_orientation),
        "dense_mixed_visible": (
            dense_mixed_status["visible"] if dense_mixed_status else -1
        ),
        "connected_side_walls": len(connected_side_walls),
        "independent_side_walls": len(independent_side_walls),
        "bright_occluded_hole": len(bright_occluded_hole),
        "hidden_crossing_side_profile": len(hidden_crossing_side_profile),
        "illuminated_rim_fragment": len(illuminated_rim_fragment),
        "moving_illuminated_rim_total": (
            moving_illuminated_rim_status["total"]
            if moving_illuminated_rim_status
            else -1
        ),
        "interstitial_gap": len(interstitial_gap),
        "surrounded_real_center": len(surrounded_real_center),
        "dense_touching_columns": len(dense_touching_columns),
        "moving_dense_columns_total": (
            moving_dense_columns_status["total"]
            if moving_dense_columns_status
            else -1
        ),
        "dense_close_twenty_three": len(dense_twenty_three),
        "moving_twenty_three_total": (
            moving_twenty_three_status["total"]
            if moving_twenty_three_status
            else -1
        ),
        "hidden_crossing_total": (
            hidden_crossing_status["total"]
            if hidden_crossing_status
            else -1
        ),
        "exit_occlusion_total": (
            exit_occlusion_status["total"]
            if exit_occlusion_status
            else -1
        ),
        "exit_midfield_total": (
            exit_midfield_status["total"]
            if exit_midfield_status
            else -1
        ),
        "exit_line_only_total": (
            exit_line_only_status["total"]
            if exit_line_only_status
            else -1
        ),
        "armed_single_line_total": (
            armed_single_line_status["total"]
            if armed_single_line_status
            else -1
        ),
        "saved_background_start_total": (
            saved_background_start_status["total"]
            if saved_background_start_status
            else -1
        ),
        "uncleared_entry_total": (
            uncleared_entry_status["total"]
            if uncleared_entry_status
            else -1
        ),
        "reset_present_total": (
            reset_present_status["total"]
            if reset_present_status
            else -1
        ),
        "late_recovery_total": (
            late_recovery_status["total"]
            if late_recovery_status
            else -1
        ),
        "recreated_above_line_total": (
            recreated_above_line_status["total"]
            if recreated_above_line_status
            else -1
        ),
        "duplicate_exit_total": duplicate_exit_total,
        "close_pair_exit_total": close_pair_exit_total,
        "paused_top_total": paused_top_total,
        "vertical_train_total": (
            vertical_train_status["total"]
            if vertical_train_status
            else -1
        ),
        "recorded_interstitial": len(recorded_interstitial),
        "recorded_interstitial_visible": (
            recorded_interstitial_status["visible"]
            if recorded_interstitial_status
            else -1
        ),
        "tangent_fragments": len(tangent_fragments),
        "tangent_fragments_ten": len(tangent_fragments_ten),
        "edge_pocket_ten": len(edge_pocket_rings),
        "boundary_hough_ten": len(boundary_hough_rings),
        "dense_identity_ten": len(dense_identity_ten),
        "dense_identity_eight": len(dense_identity_eight),
        "live_dense_fragment_raw": live_dense_fragment_raw_counts,
        "live_dense_fragment_visible": (
            live_dense_fragment_status["visible"]
            if live_dense_fragment_status
            else -1
        ),
        "live_dense_pocket_edge": len(live_dense_pocket_detections),
        "belt_join": len(belt_join),
        "belt_seam": len(seam),
        "neutral_belt_circle": len(neutral_belt_circle),
        "shadow_oval": len(shadow_detections),
        "moving_shadow_total": (
            moving_shadow_status["total"]
            if moving_shadow_status
            else -1
        ),
        "partial_overlap": len(partial_overlap),
        "touching_chain": len(touching_chain),
        "fully_coincident_rgb_limit": len(fully_coincident),
        "static_total": static_statuses[-1]["total"],
        "line_jitter_total": jitter_status["total"] if jitter_status else -1,
        "dropout_visible": [
            status["visible"] for status in display_dropout_statuses
        ],
        "reverse_total": reverse_status["total"] if reverse_status else -1,
        "moving_overlap_total": (
            moving_overlap_status["total"] if moving_overlap_status else -1
        ),
        "slow_motion_total": (
            slow_motion_status["total"] if slow_motion_status else -1
        ),
        "live_ten_ring_detections": len(performance_detections),
        "live_ten_ring_visible": (
            performance_status["visible"] if performance_status else -1
        ),
        "real_samples": real_sample_counts,
    })
    print([
        {"center": item.center, "size": tuple(round(value, 1) for value in item.ellipse[1]), "fallback": item.fallback}
        for item in rings
    ])
    if (
        oscillating_visible_values != [15] * 8
        or empty_visible_value != 0
        or overlay_center_span_x > 2.5
        or overlay_center_span_y > 2.5
        or overlay_angle_span > 3.0
        or reversed_overlay_ids != [2, 1]
        or len(split_hole_kept) != 1
        or len(touching_pair_kept) != 2
        or not multi_ring_composite_rejected
        or isolated_profile_rejected
        or green_outline_pixels <= 0
        or yellow_outline_pixels != 0
        or cyan_marker_pixels != 0
        or transient_display_values != [0, 0, 0]
        or moving_eleven_status is None
        or moving_eleven_status["visible"] != 11
        or len(rings) != 4
        or len(flat_cluster) != 5
        or len(solid_profile) != 3
        or len(mixed_orientation) != 10
        or len(dense_mixed_orientation) != 10
        or dense_mixed_status is None
        or dense_mixed_status["visible"] != 10
        or dense_mixed_status["total"] != 0
        or len(connected_side_walls) != 1
        or len(independent_side_walls) != 2
        or len(bright_occluded_hole) != 2
        or len(hidden_crossing_side_profile) != 7
        or len(illuminated_rim_fragment) != 1
        or moving_illuminated_rim_status is None
        or moving_illuminated_rim_status["visible"] != 1
        or moving_illuminated_rim_status["total"] != 1
        or len(interstitial_gap) != 4
        or len(surrounded_real_center) != 5
        or len(dense_touching_columns) != 10
        or moving_dense_columns_status is None
        or moving_dense_columns_status["total"] != 10
        or len(dense_twenty_three) != 23
        or moving_twenty_three_status is None
        or moving_twenty_three_status["total"] != 23
        or hidden_crossing_status is None
        or hidden_crossing_status["total"] != 4
        or exit_occlusion_status is None
        or exit_occlusion_status["total"] != 4
        or exit_midfield_status is None
        or exit_midfield_status["total"] != 0
        or exit_line_only_status is None
        or exit_line_only_status["total"] != 0
        or armed_single_line_status is None
        or armed_single_line_status["total"] != 1
        or saved_background_start_status is None
        or saved_background_start_status["total"] != 1
        or uncleared_entry_status is None
        or uncleared_entry_status["total"] != 0
        or reset_present_status is None
        or reset_present_status["total"] != 0
        or late_recovery_status is None
        or late_recovery_status["total"] != 1
        or recreated_above_line_status is None
        or recreated_above_line_status["total"] != 1
        or duplicate_exit_total != 1
        or close_pair_exit_total != 2
        or paused_top_total != 1
        or vertical_train_status is None
        or vertical_train_status["total"] != 2
        or len(recorded_interstitial) != 8
        or recorded_interstitial_status is None
        or recorded_interstitial_status["visible"] != 8
        or len(tangent_fragments) != 8
        or len(tangent_fragments_ten) != 10
        or len(edge_pocket_rings) != 10
        or len(boundary_hough_rings) != 10
        or has_detection_near(boundary_hough_rings, (124, 124))
        or not has_detection_near(boundary_hough_rings, (189, 336))
        or len(dense_identity_ten) != 10
        or has_detection_near(dense_identity_ten, (142, 117))
        or not has_detection_near(dense_identity_ten, (264, 229))
        or len(dense_identity_eight) != 8
        or has_detection_near(dense_identity_eight, (159, 197))
        or has_detection_near(dense_identity_eight, (163, 229))
        or not has_detection_near(dense_identity_eight, (266, 196))
        or not has_detection_near(dense_identity_eight, (262, 258))
        or live_dense_fragment_raw_counts != [7] * 8
        or live_dense_fragment_status is None
        or live_dense_fragment_status["visible"] != 7
        or len(live_dense_pocket_detections) != 7
        or has_detection_near(live_dense_pocket_detections, (200, 139))
        or len(belt_join) != 0
        or len(seam) != 0
        or len(neutral_belt_circle) != 0
        or len(shadow_detections) != 0
        or moving_shadow_status is None
        or moving_shadow_status["total"] != 0
        or len(partial_overlap) != 2
        or len(touching_chain) != 3
        or len(fully_coincident) != 1
        or any(status["total"] != 0 for status in static_statuses)
        or jitter_status is None
        or jitter_status["total"] != 0
        or [status["visible"] for status in display_dropout_statuses] != [0, 0, 0]
        or reverse_status is None
        or reverse_status["total"] != 0
        or moving_overlap_status is None
        or moving_overlap_status["total"] != 2
        or slow_motion_status is None
        or slow_motion_status["total"] != 1
        or len(performance_detections) != 10
        or performance_status is None
        or performance_status["visible"] != 10
        or any(status["visible"] != 0 or status["total"] != 0 for status in repeated_belt_statuses)
        or synthetic_seam_status is None
        or synthetic_seam_status["total"] != 0
        or real_sample_counts != real_sample_expectations
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
