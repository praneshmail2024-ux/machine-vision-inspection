"""Contour-hierarchy ring detector and line-crossing counter for conveyor images."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np


@dataclass
class RingDetection:
    center: tuple[int, int]
    inner_contour: np.ndarray
    outer_contour: np.ndarray | None
    ellipse: tuple[tuple[float, float], tuple[float, float], float]
    fallback: bool = False
    source: str = "primary"
    confidence: float = 0.95


class RingDetector:
    """Detect cardboard rings by finding an elliptical hole inside a beige contour."""

    def __init__(
        self,
        roi_left: float = 0.0,
        roi_top: float = 0.0,
        roi_right: float = 1.0,
        roi_bottom: float = 1.0,
        count_mode: str = "line",
    ) -> None:
        if count_mode not in {"line", "exit"}:
            raise ValueError("count_mode must be 'line' or 'exit'.")
        self.next_id = 1
        # id -> (x, y, counted, missing_frames, confirmed_frames, velocity_x, velocity_y)
        self.tracks: dict[int, tuple[int, int, bool, int, int, float, float]] = {}
        self.total_count = 0
        self.last_visible = 0
        self._visible_count_history: list[int] = []
        self._visible_lower_frames = 0
        self._visible_change_candidate: int | None = None
        self._visible_change_frames = 0
        # visual id -> (center_x, center_y, width, height, angle, last_frame)
        # Detection remains frame-accurate; only the operator-facing green
        # overlay is filtered so contour noise cannot make a stationary ring shake.
        self._overlay_geometry: dict[
            int, tuple[float, float, float, float, float, int]
        ] = {}
        self._next_overlay_id = 1
        self.processing_fps = 0.0
        self.background: np.ndarray | None = None
        self._foreground_mask: np.ndarray | None = None
        self._positive_foreground_mask: np.ndarray | None = None
        self.roi_left = roi_left
        self.roi_top = roi_top
        self.roi_right = roi_right
        self.roi_bottom = roi_bottom
        self.gate_ratio = 0.50
        # Exit counting needs to preserve an identity through a longer dense
        # merge because the object may not be visible again until well above the
        # decorative guide line.
        self.max_missing_frames = 24 if count_mode == "exit" else 12
        self.gate_occupied = False
        self.gate_clear_frames = 4
        self.gate_rearm_frames = 4
        self.track_crossing_count = 0
        self.track_armed: set[int] = set()
        # Exit mode counts a proven object lifecycle instead of a red-line
        # crossing: enter through the bottom FOV, stay identified, leave at top.
        self.count_mode = count_mode
        self.exit_entry_tracks: set[int] = set()
        self.exit_confirmed_tracks: set[int] = set()
        self.exit_track_start_x: dict[int, int] = {}
        self.exit_track_start_y: dict[int, int] = {}
        self.exit_recent_events: list[tuple[int, int, int]] = []
        self._tracking_frame_index = 0
        # Do not treat objects already present during startup/reset as fresh
        # entries. Exit counting becomes eligible only after the FOV has been
        # visibly clear for several consecutive frames.
        self.exit_system_armed = False
        self.exit_clear_frames = 0
        self.exit_arm_clear_frames = 5
        self._motion_frames: list[np.ndarray] = []
        self.scene_velocity_y = 0.0
        self.direction_motion_frames = 0
        self.exit_motion_grace_frames = 0
        self._display_history: list[list[RingDetection]] = []
        # The supplied recording shows objects travelling from bottom to top.
        self.motion_direction = -1
        self._debug_material_mask: np.ndarray | None = None
        self._debug_foreground_mask: np.ndarray | None = None
        self._debug_cleaned_mask: np.ndarray | None = None
        self._debug_watershed_view: np.ndarray | None = None
        self._frame_hsv: np.ndarray | None = None
        self._frame_gray: np.ndarray | None = None
        self._frame_blurred_gray: np.ndarray | None = None
        self._diagnostic_frame_index = 0

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()
        self.total_count = 0
        self.last_visible = 0
        self._visible_count_history.clear()
        self._visible_lower_frames = 0
        self._visible_change_candidate = None
        self._visible_change_frames = 0
        self._overlay_geometry.clear()
        self._next_overlay_id = 1
        self.processing_fps = 0.0
        self.gate_occupied = False
        self.gate_clear_frames = self.gate_rearm_frames
        self.track_crossing_count = 0
        self.track_armed.clear()
        self.exit_entry_tracks.clear()
        self.exit_confirmed_tracks.clear()
        self.exit_track_start_x.clear()
        self.exit_track_start_y.clear()
        self.exit_recent_events.clear()
        self._tracking_frame_index = 0
        self.exit_system_armed = False
        self.exit_clear_frames = 0
        self._motion_frames.clear()
        self.scene_velocity_y = 0.0
        self.direction_motion_frames = 0
        self.exit_motion_grace_frames = 0
        self._display_history.clear()
        self._diagnostic_frame_index = 0

    def calibrate_background(self, frame: np.ndarray) -> None:
        """Store an empty conveyor frame so non-circular/collapsed objects remain visible."""
        self.background = cv2.GaussianBlur(frame, (5, 5), 0)

    def arm_counting(self) -> None:
        """Allow verified conveyor tracks to count without another empty-FOV wait."""
        if self.count_mode == "exit":
            self.exit_system_armed = True
            self.exit_clear_frames = self.exit_arm_clear_frames

    def settings(self) -> dict[str, float]:
        return {
            "left": round(self.roi_left * 100, 1),
            "top": round(self.roi_top * 100, 1),
            "right": round(self.roi_right * 100, 1),
            "bottom": round(self.roi_bottom * 100, 1),
            "counting_line": round(self.gate_ratio * 100, 1),
        }

    def update_settings(
        self, left: float, top: float, right: float, bottom: float, counting_line: float
    ) -> None:
        values = (left, top, right, bottom, counting_line)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Field-of-view values must be between 0 and 100 percent.")
        if right - left < 0.10 or bottom - top < 0.10:
            raise ValueError("The detection area must be at least 10 percent wide and high.")
        if not top <= counting_line <= bottom:
            raise ValueError("The counting line must be inside the detection area.")
        self.roi_left, self.roi_top = left, top
        self.roi_right, self.roi_bottom = right, bottom
        self.gate_ratio = counting_line
        self.tracks.clear()
        self.track_armed.clear()
        self.exit_entry_tracks.clear()
        self.exit_confirmed_tracks.clear()
        self.exit_track_start_x.clear()
        self.exit_track_start_y.clear()
        self.exit_recent_events.clear()
        self._tracking_frame_index = 0
        self.exit_system_armed = False
        self.exit_clear_frames = 0
        self._motion_frames.clear()
        self.scene_velocity_y = 0.0
        self.direction_motion_frames = 0
        self.exit_motion_grace_frames = 0
        self._display_history.clear()
        self.last_visible = 0
        self._visible_count_history.clear()
        self._visible_lower_frames = 0
        self._visible_change_candidate = None
        self._visible_change_frames = 0
        self._overlay_geometry.clear()
        self._next_overlay_id = 1

    def _roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        roi = np.zeros((height, width), dtype=np.uint8)
        x1, x2 = int(width * self.roi_left), int(width * self.roi_right)
        y1, y2 = int(height * self.roi_top), int(height * self.roi_bottom)
        roi[y1:y2, x1:x2] = 255
        return roi

    def status(self) -> dict[str, int | float]:
        return {
            "visible": self.last_visible,
            "total": self.total_count,
            "processing_fps": round(self.processing_fps, 1),
            "background_calibrated": self.background is not None,
        }

    def background_change_ratio(self) -> float:
        """Return the foreground share inside the configured inspection area."""
        if self._foreground_mask is None:
            return 0.0
        roi = self._roi_mask(self._foreground_mask.shape)
        roi_pixels = cv2.countNonZero(roi)
        if roi_pixels <= 0:
            return 0.0
        changed_pixels = cv2.countNonZero(
            cv2.bitwise_and(self._foreground_mask, roi)
        )
        return changed_pixels / roi_pixels

    def _clean_primary_evidence_is_complete(
        self,
        frame: np.ndarray,
        detections: list[RingDetection],
    ) -> bool:
        """Decide whether expensive collapsed/overlap recovery can add an object.

        A clean hierarchy hole is already the strongest possible ring evidence.
        Full Hough, watershed and directional-profile recovery is still required
        when cardboard remains outside all proven rings, crosses an inner hole,
        or a primary contour is elongated enough to represent a merged group.
        """
        if (
            not detections
            or self._foreground_mask is None
            or self._frame_hsv is None
            or any(
                detection.source != "primary"
                or detection.confidence < 0.85
                or self._is_merged_hole(detection)
                for detection in detections
            )
        ):
            return False

        height, width = frame.shape[:2]
        image_area = height * width
        roi = self._roi_mask((height, width))
        hsv = self._frame_hsv
        beige_material = cv2.inRange(
            hsv, np.array((4, 8, 75)), np.array((42, 235, 255))
        )
        neutral_bright = cv2.inRange(
            hsv, np.array((0, 0, 100)), np.array((179, 72, 255))
        )
        foreground_support = cv2.dilate(
            self._foreground_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        material = cv2.bitwise_and(
            cv2.bitwise_or(beige_material, neutral_bright),
            cv2.bitwise_and(foreground_support, roi),
        )

        explained = np.zeros((height, width), dtype=np.uint8)
        obstruction_pixels = max(70, round(image_area * 0.00023))
        for detection in detections:
            center = (
                int(round(detection.ellipse[0][0])),
                int(round(detection.ellipse[0][1])),
            )
            axes = (
                max(1, int(round(detection.ellipse[1][0] * 0.85 / 2))),
                max(1, int(round(detection.ellipse[1][1] * 0.85 / 2))),
            )
            hole = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                hole,
                center,
                axes,
                detection.ellipse[2],
                0,
                360,
                255,
                cv2.FILLED,
            )
            # A bright band through a dark hole is independent evidence for a
            # side-on ring crossing the already proven open ring.
            if cv2.countNonZero(cv2.bitwise_and(material, hole)) >= obstruction_pixels:
                return False

            cover_axes = (
                max(1, int(round(detection.ellipse[1][0] * 1.72 / 2))),
                max(1, int(round(detection.ellipse[1][1] * 1.72 / 2))),
            )
            cv2.ellipse(
                explained,
                center,
                cover_axes,
                detection.ellipse[2],
                0,
                360,
                255,
                cv2.FILLED,
            )

        unexplained = cv2.bitwise_and(material, cv2.bitwise_not(explained))
        unexplained = cv2.morphologyEx(
            unexplained,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            unexplained
        )
        largest_unexplained = (
            int(np.max(stats[1:, cv2.CC_STAT_AREA]))
            if component_count > 1
            else 0
        )
        return largest_unexplained < max(95, round(image_area * 0.00031))

    def _ring_mask(self, frame: np.ndarray) -> np.ndarray:
        """Keep cardboard-coloured pixels while rejecting the black belt and metal rail."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._frame_hsv = hsv
        self._frame_gray = gray
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        blurred_gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        self._frame_blurred_gray = blurred_gray
        # Beige cardboard under normal factory lighting. This is one signal, not a
        # hard requirement: the Astra often renders the same rim almost grey.
        cardboard_colour = cv2.inRange(hsv, np.array((5, 10, 75)), np.array((38, 225, 255)))
        detection_region = self._roi_mask(cardboard_colour.shape)
        # Compare each pixel with its local neighbourhood rather than one global
        # threshold. The recording contains a very bright rail on the right; global
        # Otsu therefore rejected the thinner grey rims between brighter rings.
        local_bright = cv2.adaptiveThreshold(
            blurred_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            51,
            -8,
        )
        # Reject low-level belt texture while retaining grey cardboard.
        visible_material = cv2.inRange(gray, 48, 255)
        local_bright = cv2.bitwise_and(local_bright, visible_material)
        mask = cv2.bitwise_or(cardboard_colour, local_bright)
        self._debug_material_mask = cv2.bitwise_and(mask, detection_region)
        self._foreground_mask = np.zeros(mask.shape, dtype=np.uint8)
        self._positive_foreground_mask = np.zeros(mask.shape, dtype=np.uint8)
        if self.background is not None and self.background.shape == frame.shape:
            # Preserve the absolute-change mask for stable segmentation of thin
            # and side-on rings. Keep a second signed/material mask solely for
            # the final shadow veto: cast shadows darken the belt but cardboard
            # either brightens it or contributes the expected material colour.
            difference = cv2.absdiff(blurred_frame, self.background)
            absolute_change = cv2.inRange(
                cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY), 18, 255
            )
            background_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
            brightness_gain = cv2.subtract(blurred_gray, background_gray)
            brighter_foreground = cv2.inRange(brightness_gain, 18, 255)
            cardboard_change = cv2.bitwise_and(
                absolute_change, cardboard_colour
            )
            positive_foreground = cv2.bitwise_or(
                brighter_foreground, cardboard_change
            )
            foreground = absolute_change
            material = cv2.inRange(gray, 42, 255)
            self._foreground_mask = cv2.bitwise_and(
                foreground, cv2.bitwise_and(detection_region, material)
            )
            self._positive_foreground_mask = cv2.bitwise_and(
                positive_foreground,
                cv2.bitwise_and(detection_region, material),
            )
            self._debug_foreground_mask = self._foreground_mask.copy()
            # In calibrated mode, static bright structures (the white spring and
            # rails in the empty recording) must not be OR-ed back into the mask.
            # Dilated change support retains the full rim around a moving object.
            foreground_support = cv2.dilate(
                foreground,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            )
            colour_on_foreground = cv2.bitwise_and(mask, foreground_support)
            mask = cv2.bitwise_or(colour_on_foreground, self._foreground_mask)
        mask = cv2.bitwise_and(mask, detection_region)
        # A large closing kernel joins the rims of two rings that just touch. Keep
        # cleanup deliberately small and let the local-peak watershed split groups.
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        self._debug_cleaned_mask = mask.copy()
        return mask

    def detect(self, frame: np.ndarray) -> tuple[list[RingDetection], np.ndarray]:
        """Return one detection per ring, using its interior hole to avoid double contours."""
        mask = self._ring_mask(frame)
        image_area = frame.shape[0] * frame.shape[1]
        # A real ring or side profile occupies far more pixels than this. Skip
        # all ellipse, Hough and directional-morphology stages on a genuinely
        # empty calibrated belt; this preserves CPU for camera acquisition.
        if cv2.countNonZero(mask) < max(80, round(image_area * 0.00025)):
            self._debug_watershed_view = np.zeros_like(frame)
            return [], mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        min_hole_area = image_area * 0.0008
        max_hole_area = image_area * 0.35
        detections: list[RingDetection] = []

        for index, contour in enumerate(contours if hierarchy is not None else []):
            parent = hierarchy[0][index][3]
            # A ring is recognised by a dark inner hole bounded by cardboard: child contour only.
            if parent < 0 or len(contour) < 5:
                continue

            area = cv2.contourArea(contour)
            if not min_hole_area <= area <= max_hole_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Perspective and imperfect cardboard make a circular ring appear oval or uneven.
            if circularity < 0.28:
                continue

            ellipse = cv2.fitEllipse(contour)
            major, minor = sorted(ellipse[1], reverse=True)
            if minor < max(22, min(frame.shape[:2]) * 0.045) or minor / major < 0.22:
                continue
            if self.background is not None and major < 42.0:
                # Tiny belt texture pockets can become hierarchy children after
                # background subtraction. A real ring hole in the calibrated
                # production view preserves at least one substantial axis;
                # collapsed/side-on rings are handled by independent material
                # and edge paths below.
                continue
            ellipse_area = np.pi * major * minor / 4
            if not 0.55 <= area / ellipse_area <= 1.25:
                continue

            outer = contours[parent]
            # Touching rings may share one large outer contour. In that case the fitted ellipse
            # is used for the outer guide; otherwise draw the true outer contour.
            outer_area = cv2.contourArea(outer)
            outer_contour = outer if outer_area / max(area, 1) < 2.8 else None
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            detections.append(RingDetection(center, contour, outer_contour, ellipse))

        # The mask can contain a second small child contour inside the same dark
        # hole (dust, highlights, or an annotation in a diagnostic recording).
        # Keep the largest ellipse first, then suppress offset duplicates.
        primary_candidates = detections
        if len(primary_candidates) >= 3:
            candidate_areas = sorted(
                (item.ellipse[1][0] * item.ellipse[1][1] for item in primary_candidates),
                reverse=True,
            )
            reference_count = max(2, int(np.ceil(len(candidate_areas) * 0.55)))
            reference_area = float(np.median(candidate_areas[:reference_count]))
            # Small diamond/triangular voids formed between four touching rings
            # are hierarchy children too, but are far smaller than true holes.
            primary_candidates = [
                item
                for item in primary_candidates
                if (
                    item.ellipse[1][0] * item.ellipse[1][1]
                    >= reference_area * 0.34
                    or (
                        max(item.ellipse[1]) >= 58.0
                        and min(item.ellipse[1]) / max(item.ellipse[1]) <= 0.52
                    )
                )
            ]
        detections = []
        for candidate in sorted(
            primary_candidates,
            key=lambda item: item.ellipse[1][0] * item.ellipse[1][1],
            reverse=True,
        ):
            candidate_area = candidate.ellipse[1][0] * candidate.ellipse[1][1]
            if any(
                self._point_inside_ellipse(
                    candidate.center, existing.ellipse, scale=1.25
                )
                and candidate_area
                < existing.ellipse[1][0] * existing.ellipse[1][1] * 0.70
                for existing in detections
            ):
                # Dust/highlights can cut a second, smaller child contour inside
                # one true hole. Containment plus a clear area difference is
                # safer than globally merging nearby rings that genuinely touch.
                continue
            if not any(self._same_ring(candidate, existing) for existing in detections):
                detections.append(candidate)

        # Recover a collapsed/open oval from its complete material outline before
        # considering weak edge fragments.  Previously, a small fallback arc could
        # occupy the same contour first and veto the correct material candidate.
        for candidate in self._material_ovals(mask, frame.shape[:2]):
            contains_known_ring = any(
                cv2.pointPolygonTest(candidate.inner_contour, detected.center, False) >= 0
                for detected in detections
            )
            if not contains_known_ring and not any(
                self._same_ring(candidate, detected) for detected in detections
            ):
                detections.append(candidate)

        preliminary_strong_primary_count = sum(
            detection.source == "primary"
            and detection.confidence >= 0.85
            for detection in detections
        )
        preliminary_primary_complete = (
            len(detections) >= 10
            and preliminary_strong_primary_count >= 9
            and not any(
                detection.source == "primary"
                and self._is_merged_hole(detection)
                for detection in detections
            )
        ) or self._clean_primary_evidence_is_complete(frame, detections)

        # A collapsed or partly hidden ring can lose its enclosed hole. Add
        # ellipse candidates from its visible edges only when the primary holes
        # leave material unexplained. A fully proven clean scene otherwise paid
        # for Canny/ellipse scoring that could not add another physical object.
        fallback_candidates = (
            []
            if preliminary_primary_complete
            else self._fallback_ellipses(frame, mask, detections)
        )
        for candidate in fallback_candidates:
            contained_primary_centres = sum(
                self._point_inside_ellipse(primary.center, candidate.ellipse, scale=1.02)
                for primary in detections
                if primary.source == "primary"
            )
            # One large open contour around two already-proven holes is the
            # touching group boundary, not an additional ring.
            if contained_primary_centres >= 2:
                continue
            candidate_area = candidate.ellipse[1][0] * candidate.ellipse[1][1]
            is_fragment_of_stronger_detection = any(
                detected.source in {"primary", "material", "overlap", "flat"}
                and self._point_inside_ellipse(candidate.center, detected.ellipse, scale=1.10)
                and candidate_area
                < detected.ellipse[1][0] * detected.ellipse[1][1] * 0.60
                for detected in detections
            )
            if is_fragment_of_stronger_detection:
                continue
            if not any(self._same_ring(candidate, detected) for detected in detections):
                detections.append(candidate)

        strong_primary_count = sum(
            detection.source == "primary"
            and detection.confidence >= 0.85
            for detection in detections
        )
        primary_complete_fast_path = preliminary_primary_complete or (
            len(detections) >= 10
            and strong_primary_count >= 9
            and not any(
                detection.source == "primary"
                and self._is_merged_hole(detection)
                for detection in detections
            )
        ) or self._clean_primary_evidence_is_complete(frame, detections)

        # Independent circular recovery. Touching rims can merge their inner
        # holes into one long contour while their individual radial edge patterns
        # remain visible. Two or more strong Hough proposals can therefore replace
        # one elongated primary group, enabling evidence-based overlap splitting.
        # When at least ten proposals are already present and nine are strong,
        # independent hierarchy holes, this recovery is redundant. The
        # conservative fast path preserves the full pipeline for smaller,
        # collapsed, irregular, or merged-hole scenes.
        hough_candidates = (
            []
            if primary_complete_fast_path
            else self._hough_rings(frame, mask)
        )
        merged_groups: list[RingDetection] = []
        for merged in detections:
            if merged.source != "primary" or not self._is_merged_hole(merged):
                continue
            merged_minor = min(merged.ellipse[1])
            members = [
                candidate
                for candidate in hough_candidates
                # Two independently validated partial circles inside the same
                # elongated hole are stronger evidence than either proposal on
                # its own.  Keeping the old 0.70 cut-off discarded both members
                # whenever close rims hid roughly half of each circumference.
                if candidate.confidence >= 0.58
                if self._point_inside_ellipse(candidate.center, merged.ellipse, scale=1.22)
                and 35 <= max(candidate.ellipse[1]) <= max(115, merged_minor * 1.9)
            ]
            if len(members) < 2:
                continue
            member_diameter = float(np.median([max(item.ellipse[1]) for item in members]))
            maximum_separation = max(
                np.hypot(
                    first.center[0] - second.center[0],
                    first.center[1] - second.center[1],
                )
                for index, first in enumerate(members)
                for second in members[index + 1 :]
            )
            if maximum_separation < max(member_diameter * 0.42, max(merged.ellipse[1]) * 0.35):
                continue
            strong_member_count = sum(
                member.confidence >= 0.70 for member in members
            )
            # Moderate-only splits need two separated circles and a clearly
            # elongated parent. This prevents a lone interstitial pocket from
            # becoming an object while allowing two mutually supporting,
            # partially occluded rims to separate.
            if (
                strong_member_count == 0
                and (
                    len(members) < 2
                    or max(merged.ellipse[1]) / max(merged_minor, 1.0) < 1.80
                )
            ):
                continue
            for member in members:
                member.source = "overlap"
                member.confidence = 0.94 if member.confidence >= 0.70 else 0.86
            merged_groups.append(merged)

        if merged_groups:
            merged_ids = {id(candidate) for candidate in merged_groups}
            detections = [candidate for candidate in detections if id(candidate) not in merged_ids]

        for candidate in hough_candidates:
            if candidate.source != "overlap" and candidate.confidence < 0.70:
                continue
            # A single circle inside an elongated primary is not enough evidence
            # to split it.  Only the validated multi-circle branch above may turn
            # those contained proposals into independent overlap detections.
            if candidate.source != "overlap" and any(
                detected.source == "primary"
                and self._point_inside_ellipse(candidate.center, detected.ellipse, scale=1.22)
                for detected in detections
            ):
                continue
            if not any(self._same_ring(candidate, detected) for detected in detections):
                detections.append(candidate)

        # A bright highlight on one wall can create a small hierarchy child
        # inside a full fallback/Hough ring. Keep the complete ring and discard
        # only a substantially smaller contained primary fragment.
        detections = [
            candidate
            for candidate in detections
            if candidate.source != "primary"
            or not any(
                other is not candidate
                and other.source in {"fallback", "hough", "material", "overlap"}
                and self._point_inside_ellipse(
                    candidate.center, other.ellipse, scale=1.06
                )
                and candidate.ellipse[1][0] * candidate.ellipse[1][1]
                < other.ellipse[1][0] * other.ellipse[1][1] * 0.55
                for other in detections
            )
        ]

        # Re-check fallback group boundaries after overlap promotion.  At the
        # earlier fallback stage a group may still have been represented by one
        # merged primary, so its surrounding ellipse contained only one centre.
        trusted_centres = [
            detected.center
            for detected in detections
            if detected.source in {"primary", "overlap", "material", "flat"}
        ]
        detections = [
            candidate
            for candidate in detections
            if candidate.source != "fallback"
            or sum(
                self._point_inside_ellipse(center, candidate.ellipse, scale=1.02)
                for center in trusted_centres
            )
            < 2
        ]

        # Seeded watershed separates joined material regions. Inner-hole detections are strong
        # markers; distance-transform peaks recover collapsed rings without an opening.
        self._diagnostic_frame_index += 1
        update_diagnostic_watershed = (
            not primary_complete_fast_path
            or self._debug_watershed_view is None
            or self._diagnostic_frame_index % 30 == 0
        )
        watershed_candidates = (
            self._watershed_ovals(frame, mask, detections)
            if update_diagnostic_watershed
            else []
        )
        if self._debug_watershed_view is None or self._diagnostic_frame_index % 30 == 0:
            # Show only watershed-accepted regions. Using the RGB image as the
            # background made an ordinary belt seam look like a detected object.
            watershed_view = np.zeros_like(frame)
            palette = ((255, 0, 255), (0, 190, 255), (70, 255, 120), (255, 170, 40))
            for index, candidate in enumerate(watershed_candidates, start=1):
                colour = palette[(index - 1) % len(palette)]
                cv2.drawContours(watershed_view, [candidate.inner_contour], -1, colour, cv2.FILLED, cv2.LINE_AA)
                cv2.drawContours(watershed_view, [candidate.inner_contour], -1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(watershed_view, candidate.center, 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    watershed_view,
                    f"W{index}",
                    (candidate.center[0] + 7, candidate.center[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            self._debug_watershed_view = watershed_view
        # Watershed remains a diagnostic segmentation view, but it is not an
        # independent source of object count. Blob geometry alone previously
        # promoted belt joints and the outside of touching groups as fake rings.

        # A ring resting fully on its side has no dark opening at all.  Fuse the
        # independently validated solid-band proposals without allowing one band
        # to erase several proven holes in a touching cluster.
        flat_candidates = (
            []
            if primary_complete_fast_path
            else self._flat_material_profiles(frame)
        )
        for flat_candidate in flat_candidates:
            flat_major = max(flat_candidate.ellipse[1])
            directional_probe = flat_candidate.confidence == 0.88
            if self._is_undersized_dense_flat_fragment(
                flat_candidate, detections
            ):
                # In a dense pile, short pieces of the rim from a ring that is
                # already leaving the ROI can survive the solid-band erosion.
                # Learn the projected ring diameter from the proven holes in the
                # same frame and require a side-on object to preserve most of it.
                continue
            if self._is_duplicate_open_ring_wall(flat_candidate, detections):
                # The isotropic material-core pass can describe the bright outer
                # wall of an ordinary open ring as a second broad oval.  A true
                # crossing side profile is much thinner; comparable, near-
                # concentric axes are duplicate evidence for the same ring.
                continue
            if directional_probe and self._is_oversized_interior_flat_bridge(
                flat_candidate, detections
            ):
                # Directional morphology can connect the facing rims of two
                # dense ring columns into one long central strip. A physical
                # side-on ring stays close to the diameter learned from nearby
                # proven holes; this multi-ring bridge is substantially longer.
                continue
            if flat_candidate.confidence == 0.88 and any(
                detected.source
                in {"primary", "overlap", "hough", "material"}
                and self._point_inside_ellipse(
                    flat_candidate.center, detected.ellipse, scale=1.45
                )
                for detected in detections
            ):
                # Directional morphology is a supplementary recovery signal.
                # A straight section of an already proven oval rim can also
                # survive that opening, so never promote it through the middle
                # of a stronger complete-ring proposal. Multi-scale core flats
                # (0.90) remain eligible when a real side-on ring crosses one.
                continue
            if flat_candidate.confidence >= 0.90 and any(
                self._is_tangent_rim_fragment(flat_candidate, detected)
                for detected in detections
            ):
                # A brightly lit section of an ordinary ring wall can survive
                # the erosion-core test as a short solid band.  Reject it only
                # when it lies in the proven ring's annulus, follows the local
                # ellipse tangent, and is shorter/thinner than that ring's hole.
                # Genuine side-on rings crossing a hole or lying beside it do
                # not satisfy this combined geometry check.
                continue
            is_short_wall_of_open_ring = any(
                detected.source
                in (
                    {"primary", "overlap", "hough", "material"}
                    if directional_probe
                    else {
                        "primary",
                        "overlap",
                        "fallback",
                        "hough",
                        "material",
                    }
                )
                and self._point_inside_ellipse(
                    flat_candidate.center, detected.ellipse, scale=1.80
                )
                and flat_major < max(detected.ellipse[1]) * 0.78
                for detected in detections
            )
            if is_short_wall_of_open_ring:
                continue
            # A parallel-edge-only proposal on the broad near wall of an already
            # proven open ring is supporting evidence for that ring, not another
            # object.  Strong multi-scale solid cores remain eligible when two
            # physical pieces genuinely cross.
            if flat_candidate.confidence < 0.85 and any(
                detected.source
                in {"primary", "overlap", "fallback", "hough", "material"}
                and self._point_inside_ellipse(
                    flat_candidate.center, detected.ellipse, scale=1.35
                )
                for detected in detections
            ):
                continue
            geometric_duplicates = [
                detected
                for detected in detections
                if self._rotated_rect_iou(detected.ellipse, flat_candidate.ellipse) >= 0.42
            ]
            flat_guard_duplicates = [
                detected
                for detected in detections
                if (
                    detected.source in {"fallback", "hough", "material"}
                    or (
                        detected.source == "primary"
                        and detected.ellipse[1][0] * detected.ellipse[1][1]
                        < flat_candidate.ellipse[1][0]
                        * flat_candidate.ellipse[1][1]
                        * 0.55
                    )
                )
                and self._center_overlaps_flat_profile(detected, flat_candidate)
                and np.hypot(
                    detected.center[0] - flat_candidate.center[0],
                    detected.center[1] - flat_candidate.center[1],
                )
                < max(24.0, max(flat_candidate.ellipse[1]) * 0.46)
            ]
            geometric_duplicate_ids = {
                id(detected) for detected in geometric_duplicates
            }
            geometric_duplicates.extend(
                detected
                for detected in flat_guard_duplicates
                if id(detected) not in geometric_duplicate_ids
            )
            if any(
                detected.source in {"primary", "overlap"}
                and detected.confidence >= flat_candidate.confidence
                for detected in geometric_duplicates
            ):
                continue
            # Replace only weaker geometric proposals for the same band.  Nearby
            # or crossing primary holes belong to other physical rings and stay.
            duplicate_ids = {
                id(detected)
                for detected in geometric_duplicates
                if detected.source in {"fallback", "material", "hough"}
                or detected.confidence < flat_candidate.confidence
            }
            detections = [
                detected for detected in detections if id(detected) not in duplicate_ids
            ]
            if not any(
                self._rotated_rect_iou(detected.ellipse, flat_candidate.ellipse) >= 0.42
                for detected in detections
            ):
                detections.append(flat_candidate)

        flat_candidate_count = sum(
            detection.source == "flat" for detection in detections
        )
        if (
            flat_candidate_count >= 2
            and flat_candidate_count * 2 >= len(detections)
            and len(watershed_candidates) > 0
            and len(watershed_candidates) * 2 <= len(detections)
        ):
            # A newly lit/moving belt can turn its joint and specular streaks into
            # several independent flat proposals. In the reported single-ring
            # sequence this produced 3–6 raw candidates while watershed remained
            # at one physical object. When flat recovery dominates the frame,
            # promote the marker-controlled watershed result as the authoritative
            # object set. A one-frame watershed over-split still fails temporal
            # confirmation and cannot flash in the final view.
            detections = list(watershed_candidates)

        # If a bright, partially occluded ring still has no hole/arc proposal,
        # recover at most a few substantial cardboard regions left unexplained
        # after masking every proven object. This is material evidence, not an
        # area-only count estimate.
        residual_candidates = (
            []
            if primary_complete_fast_path
            else self._residual_material_candidates(frame, detections)
        )
        for residual_candidate in residual_candidates:
            if not any(
                self._same_ring(residual_candidate, detected)
                or self._point_inside_ellipse(
                    residual_candidate.center, detected.ellipse, scale=1.16
                )
                for detected in detections
            ):
                detections.append(residual_candidate)

        # Strong highlights can split one dark ring opening into two displaced
        # hierarchy children. They are not two touching rings when the short
        # bridge between their centres is continuously dark (a genuine pair has
        # a bright cardboard wall across that bridge).
        detections = self._suppress_split_hole_duplicates(
            detections,
            self._frame_gray
            if self._frame_gray is not None
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        )

        # In a fast dense column, morphology can occasionally join the facing
        # walls of two consecutive rings into one long oval. The two real holes
        # are still detected independently, so retaining that enclosing oval
        # creates an eleventh track for ten physical rings. Reject only a large,
        # aligned proposal that geometrically spans two strong ring centres.
        detections = [
            candidate
            for candidate in detections
            if not self._is_multi_ring_composite(candidate, detections)
        ]

        # When three or four rings touch, the tiny dark diamond between them can
        # resemble an additional oval to the edge/fallback stages. It is not a
        # ring hole: it is substantially smaller and geometrically surrounded by
        # real holes. Reject only that specific configuration so isolated small,
        # irregular, or collapsed rings remain eligible.
        detections = [
            candidate
            for candidate in detections
            if not self._is_interstitial_gap(candidate, detections)
        ]
        if self.background is not None:
            detections = [
                candidate
                for candidate in detections
                if self._has_nonshadow_foreground(candidate)
            ]
        if (
            self.background is not None
            and len(detections) == 1
            and not self._has_isolated_cardboard_support(
                detections[0], frame
            )
        ):
            # A moving grey belt mark can form a clean hierarchy child and look
            # circular even though it has no cardboard rim. In an isolated scene
            # require either beige pixels or a genuinely bright projected rim.
            # Dense piles are handled by the interstitial/duplicate geometry
            # guards above, while flat/material side profiles use their own much
            # stricter foreground-fill validation and remain eligible here.
            detections = []
        if self.background is not None:
            detections = [
                candidate
                for candidate in detections
                if self._detection_inside_roi(candidate, frame.shape[:2])
            ]

        return detections, mask

    def _has_nonshadow_foreground(
        self, candidate: RingDetection
    ) -> bool:
        """Require signed/cardboard evidence inside a detected object region."""
        if self._positive_foreground_mask is None:
            return True
        support_region = np.zeros(
            self._positive_foreground_mask.shape, dtype=np.uint8
        )
        center, size, angle = candidate.ellipse
        if candidate.source in {"flat", "material", "residual"}:
            cv2.ellipse(
                support_region,
                (center, (float(size[0]) * 1.10, float(size[1]) * 1.10), angle),
                255,
                cv2.FILLED,
            )
        else:
            inner = np.zeros(support_region.shape, dtype=np.uint8)
            cv2.ellipse(
                inner,
                (center, (float(size[0]) * 0.88, float(size[1]) * 0.88), angle),
                255,
                cv2.FILLED,
            )
            cv2.ellipse(
                support_region,
                (center, (float(size[0]) * 1.45, float(size[1]) * 1.45), angle),
                255,
                cv2.FILLED,
            )
            support_region = cv2.bitwise_and(
                support_region, cv2.bitwise_not(inner)
            )
        region_area = cv2.countNonZero(support_region)
        if region_area < 80:
            return False
        positive_area = cv2.countNonZero(
            cv2.bitwise_and(
                self._positive_foreground_mask, support_region
            )
        )
        if candidate.source in {"flat", "material", "residual"}:
            # A side-on/collapsed ring can expose only a thin bright cardboard
            # strip. The recorded dense case has 2.27% signed support, while a
            # soft belt shadow has none. Keep this allowance local to projected
            # profiles; ordinary hole detections retain the stricter threshold.
            return positive_area >= max(50, round(region_area * 0.018))
        return positive_area >= max(40, round(region_area * 0.025))

    def _has_isolated_cardboard_support(
        self, candidate: RingDetection, frame: np.ndarray
    ) -> bool:
        """Reject an isolated neutral belt loop without hiding side profiles."""
        if candidate.source in {"flat", "material", "residual"}:
            return True

        gray = (
            self._frame_gray
            if self._frame_gray is not None
            and self._frame_gray.shape == frame.shape[:2]
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        hsv = (
            self._frame_hsv
            if self._frame_hsv is not None
            and self._frame_hsv.shape[:2] == frame.shape[:2]
            else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        )
        center, size, angle = candidate.ellipse
        inner = np.zeros(gray.shape, dtype=np.uint8)
        outer = np.zeros(gray.shape, dtype=np.uint8)
        cv2.ellipse(
            inner,
            (center, (float(size[0]) * 0.96, float(size[1]) * 0.96), angle),
            255,
            cv2.FILLED,
        )
        cv2.ellipse(
            outer,
            (center, (float(size[0]) * 1.38, float(size[1]) * 1.38), angle),
            255,
            cv2.FILLED,
        )
        annulus = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        annulus_area = cv2.countNonZero(annulus)
        if annulus_area < 80:
            return False

        cardboard = cv2.inRange(
            hsv, np.array((4, 8, 70)), np.array((42, 240, 255))
        )
        cardboard_fraction = (
            cv2.countNonZero(cv2.bitwise_and(cardboard, annulus))
            / annulus_area
        )
        rim_brightness = cv2.mean(gray, mask=annulus)[0]
        return cardboard_fraction >= 0.040 or rim_brightness >= 100.0

    def _residual_material_candidates(
        self, frame: np.ndarray, detections: list[RingDetection]
    ) -> list[RingDetection]:
        """Recover large foreground-cardboard regions not explained by a ring."""
        if self.background is None or self._foreground_mask is None:
            return []
        height, width = frame.shape[:2]
        hsv = (
            self._frame_hsv
            if self._frame_hsv is not None
            and self._frame_hsv.shape[:2] == frame.shape[:2]
            else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        )
        cardboard = cv2.inRange(
            hsv, np.array((4, 8, 70)), np.array((42, 240, 255))
        )
        foreground_support = cv2.dilate(
            self._foreground_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        cardboard = cv2.bitwise_and(cardboard, foreground_support)
        cardboard = cv2.bitwise_and(cardboard, self._roi_mask(cardboard.shape))

        explained = np.zeros(cardboard.shape, dtype=np.uint8)
        for detection in detections:
            center, size, angle = detection.ellipse
            if detection.source == "flat":
                expanded = (
                    center,
                    (float(size[0]) * 1.25, float(size[1]) * 1.25),
                    angle,
                )
                cv2.fillConvexPoly(
                    explained,
                    np.int32(np.round(cv2.boxPoints(expanded))),
                    255,
                )
            else:
                expanded = (
                    center,
                    (float(size[0]) * 1.65, float(size[1]) * 1.65),
                    angle,
                )
                cv2.ellipse(explained, expanded, 255, cv2.FILLED)

        residual = cv2.bitwise_and(cardboard, cv2.bitwise_not(explained))
        residual = cv2.morphologyEx(
            residual,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            residual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = height * width
        minimum_area = max(700.0, image_area * 0.0020)
        maximum_area = image_area * 0.035
        candidates: list[RingDetection] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = float(cv2.contourArea(contour))
            if not minimum_area <= area <= maximum_area:
                continue
            rect = cv2.minAreaRect(contour)
            major, minor = sorted(rect[1], reverse=True)
            if not 45.0 <= major <= 155.0 or not 14.0 <= minor <= 105.0:
                continue
            aspect = major / max(minor, 1.0)
            if not 1.20 <= aspect <= 5.8:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(hull_area, 1.0)
            if solidity < 0.35:
                continue
            candidate = RingDetection(
                (int(round(rect[0][0])), int(round(rect[0][1]))),
                contour,
                contour,
                (rect[0], rect[1], rect[2]),
                fallback=True,
                source="residual",
                confidence=0.62,
            )
            if not self._detection_inside_roi(candidate, frame.shape[:2]):
                continue
            if not any(self._same_ring(candidate, existing) for existing in candidates):
                candidates.append(candidate)
            if len(candidates) >= 3:
                break
        return candidates

    def _detection_inside_roi(
        self, candidate: RingDetection, frame_shape: tuple[int, int]
    ) -> bool:
        """Keep complete objects and reject clipped rail/edge fragments."""
        height, width = frame_shape
        clearance = max(10.0, min(height, width) * 0.02)
        box = cv2.boxPoints(candidate.ellipse)
        roi_x1, roi_x2 = width * self.roi_left, width * self.roi_right
        roi_y1, roi_y2 = height * self.roi_top, height * self.roi_bottom
        if candidate.source not in {"flat", "residual"}:
            center_x, center_y = candidate.ellipse[0]
            if not (
                roi_x1 + 2.0 < center_x < roi_x2 - 2.0
                and roi_y1 + 2.0 < center_y < roi_y2 - 2.0
            ):
                return False
            roi_rectangle = (
                ((roi_x1 + roi_x2) * 0.5, (roi_y1 + roi_y2) * 0.5),
                (roi_x2 - roi_x1, roi_y2 - roi_y1),
                0.0,
            )
            _, intersection = cv2.rotatedRectangleIntersection(
                candidate.ellipse, roi_rectangle
            )
            intersection_area = (
                abs(float(cv2.contourArea(intersection)))
                if intersection is not None
                else 0.0
            )
            candidate_area = max(
                float(candidate.ellipse[1][0] * candidate.ellipse[1][1]), 1.0
            )
            boundary_clearance = max(4.0, min(height, width) * 0.012)
            near_roi_boundary = (
                float(np.min(box[:, 0])) <= roi_x1 + boundary_clearance
                or float(np.max(box[:, 0])) >= roi_x2 - boundary_clearance
                or float(np.min(box[:, 1])) <= roi_y1 + boundary_clearance
                or float(np.max(box[:, 1])) >= roi_y2 - boundary_clearance
            )
            if candidate.source == "primary" and near_roi_boundary:
                contour_area = abs(float(cv2.contourArea(candidate.inner_contour)))
                ellipse_area = np.pi * candidate_area / 4.0
                if contour_area / max(ellipse_area, 1.0) < 0.72:
                    # A dark pocket between the last ring and the conveyor rail
                    # can close against the ROI edge and resemble a large hole.
                    # Wait until an entering real ring forms a regular ellipse.
                    return False
            # Let a ring be detected while its rim is entering the field of
            # view. Its centre and most of its fitted shape must already be
            # inside, so the adjacent metal rail cannot become a candidate.
            return intersection_area / candidate_area >= 0.72
        return (
            float(np.min(box[:, 0])) > roi_x1 + clearance
            and float(np.max(box[:, 0])) < roi_x2 - clearance
            and float(np.min(box[:, 1])) > roi_y1 + clearance
            and float(np.max(box[:, 1])) < roi_y2 - clearance
        )

    @staticmethod
    def _same_ring(first: RingDetection, second: RingDetection) -> bool:
        first_size = max(first.ellipse[1])
        second_size = max(second.ellipse[1])
        distance = np.hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])
        if distance < 7:
            return True

        first_merged = RingDetector._is_merged_hole(first)
        second_merged = RingDetector._is_merged_hole(second)
        # A Hough centre inside an elongated merged hole is deliberately a split
        # proposal, not a duplicate of the group. The old 70%-of-largest-axis
        # rule collapsed these genuine overlapping centres back into one object.
        if (
            (first_merged and second.source == "overlap")
            or (second_merged and first.source == "overlap")
        ):
            return False

        smaller_size = min(first_size, second_size)
        if first.source == "primary" and second.source == "primary":
            threshold = max(12.0, 0.22 * smaller_size)
        elif first.source == "primary" or second.source == "primary":
            threshold = max(18.0, 0.48 * smaller_size)
        else:
            threshold = max(18.0, 0.34 * smaller_size)
        return distance < threshold

    @staticmethod
    def _split_hole_duplicate(
        first: RingDetection,
        second: RingDetection,
        gray: np.ndarray,
    ) -> bool:
        """Identify two aligned ellipse fragments cut from one dark opening."""
        if first.source != "primary" or second.source != "primary":
            return False

        first_major, first_minor = sorted(first.ellipse[1], reverse=True)
        second_major, second_minor = sorted(second.ellipse[1], reverse=True)
        major_similarity = min(first_major, second_major) / max(
            first_major, second_major, 1.0
        )
        if major_similarity < 0.78 or min(first_minor, second_minor) > 30.0:
            return False

        distance = float(
            np.hypot(
                first.center[0] - second.center[0],
                first.center[1] - second.center[1],
            )
        )
        if distance < 7.0 or distance > min(
            48.0, max(first_major, second_major) * 0.64
        ):
            return False

        def major_axis_angle(detection: RingDetection) -> float:
            angle = float(detection.ellipse[2])
            if detection.ellipse[1][0] > detection.ellipse[1][1]:
                angle += 90.0
            return angle % 180.0

        angle_difference = abs(
            (
                (
                    major_axis_angle(first)
                    - major_axis_angle(second)
                    + 90.0
                )
                % 180.0
            )
            - 90.0
        )
        if angle_difference > 30.0:
            return False

        first_center = np.asarray(first.center, dtype=np.float32)
        second_center = np.asarray(second.center, dtype=np.float32)
        bridge_start = tuple(
            np.rint(first_center * 0.72 + second_center * 0.28).astype(int)
        )
        bridge_end = tuple(
            np.rint(first_center * 0.28 + second_center * 0.72).astype(int)
        )
        bridge_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(bridge_mask, bridge_start, bridge_end, 255, 7, cv2.LINE_AA)
        bridge_values = gray[bridge_mask > 0]
        if bridge_values.size == 0:
            return False

        dark_fraction = float(np.mean(bridge_values < 90))
        cardboard_fraction = float(np.mean(bridge_values > 120))
        return dark_fraction >= 0.78 and cardboard_fraction <= 0.10

    @classmethod
    def _suppress_split_hole_duplicates(
        cls,
        detections: list[RingDetection],
        gray: np.ndarray,
    ) -> list[RingDetection]:
        """Keep the strongest fragment from each proven single-hole group."""
        ranked = sorted(
            enumerate(detections),
            key=lambda item: (
                item[1].confidence,
                item[1].ellipse[1][0] * item[1].ellipse[1][1],
            ),
            reverse=True,
        )
        kept_indices: list[int] = []
        kept_detections: list[RingDetection] = []
        for index, candidate in ranked:
            if any(
                cls._split_hole_duplicate(candidate, kept, gray)
                for kept in kept_detections
            ):
                continue
            kept_indices.append(index)
            kept_detections.append(candidate)

        accepted = set(kept_indices)
        return [
            detection
            for index, detection in enumerate(detections)
            if index in accepted
        ]

    @staticmethod
    def _is_merged_hole(candidate: RingDetection) -> bool:
        major, minor = sorted(candidate.ellipse[1], reverse=True)
        return major >= 100 and major / max(minor, 1.0) >= 1.65

    @staticmethod
    def _point_inside_ellipse(
        point: tuple[int, int],
        ellipse: tuple[tuple[float, float], tuple[float, float], float],
        scale: float = 1.0,
    ) -> bool:
        center, size, angle = ellipse
        radius_x = max(2.0, float(size[0]) * 0.5 * scale)
        radius_y = max(2.0, float(size[1]) * 0.5 * scale)
        delta_x = float(point[0]) - float(center[0])
        delta_y = float(point[1]) - float(center[1])
        radians = np.deg2rad(float(angle))
        cosine, sine = np.cos(radians), np.sin(radians)
        local_x = delta_x * cosine + delta_y * sine
        local_y = -delta_x * sine + delta_y * cosine
        normalized_radius = (
            (local_x / radius_x) ** 2
            + (local_y / radius_y) ** 2
        )
        return normalized_radius <= 1.0 + 1e-6

    @classmethod
    def _is_multi_ring_composite(
        cls,
        candidate: RingDetection,
        detections: list[RingDetection],
    ) -> bool:
        """Reject one elongated proposal spanning two proven ring holes."""
        candidate_major, candidate_minor = sorted(
            candidate.ellipse[1], reverse=True
        )
        if (
            candidate_major < 100.0
            or candidate_major / max(candidate_minor, 1.0) < 1.45
        ):
            return False

        candidate_area = (
            float(candidate.ellipse[1][0])
            * float(candidate.ellipse[1][1])
        )
        contained = [
            detection
            for detection in detections
            if detection is not candidate
            and detection.source in {"primary", "overlap"}
            and detection.confidence >= 0.85
            and cls._point_inside_ellipse(
                detection.center, candidate.ellipse, scale=1.02
            )
        ]
        if len(contained) < 2:
            return False

        width, height = candidate.ellipse[1]
        major_angle = float(candidate.ellipse[2])
        if height > width:
            major_angle += 90.0
        major_angle %= 180.0

        for first_index, first in enumerate(contained):
            first_major = max(first.ellipse[1])
            first_area = (
                float(first.ellipse[1][0])
                * float(first.ellipse[1][1])
            )
            for second in contained[first_index + 1 :]:
                second_major = max(second.ellipse[1])
                second_area = (
                    float(second.ellipse[1][0])
                    * float(second.ellipse[1][1])
                )
                member_reference = max(first_major, second_major)
                if candidate_major < member_reference * 1.45:
                    continue
                if candidate_area < max(first_area, second_area) * 1.45:
                    continue

                delta_x = float(second.center[0] - first.center[0])
                delta_y = float(second.center[1] - first.center[1])
                separation = float(np.hypot(delta_x, delta_y))
                if separation < min(first_major, second_major) * 0.55:
                    continue

                pair_angle = float(
                    np.degrees(np.arctan2(delta_y, delta_x)) % 180.0
                )
                angle_difference = abs(pair_angle - major_angle)
                angle_difference = min(
                    angle_difference, 180.0 - angle_difference
                )
                if angle_difference > 28.0:
                    continue

                midpoint = (
                    (first.center[0] + second.center[0]) * 0.5,
                    (first.center[1] + second.center[1]) * 0.5,
                )
                midpoint_distance = float(
                    np.hypot(
                        midpoint[0] - candidate.center[0],
                        midpoint[1] - candidate.center[1],
                    )
                )
                if midpoint_distance <= candidate_major * 0.24:
                    return True
        return False

    @staticmethod
    def _ellipse_arc_quality(
        contour: np.ndarray,
        ellipse: tuple[tuple[float, float], tuple[float, float], float],
    ) -> tuple[float, float, float]:
        """Return angular coverage plus median/P90 normalized ellipse residuals."""
        ellipse_width, ellipse_height = ellipse[1]
        if ellipse_width <= 0 or ellipse_height <= 0 or len(contour) < 5:
            return 0.0, float("inf"), float("inf")

        points = contour[:, 0, :].astype(np.float32)
        delta_x = points[:, 0] - ellipse[0][0]
        delta_y = points[:, 1] - ellipse[0][1]
        angle = np.deg2rad(ellipse[2])
        cosine, sine = np.cos(angle), np.sin(angle)
        normalized_x = (
            delta_x * cosine + delta_y * sine
        ) / max(ellipse_width * 0.5, 1e-6)
        normalized_y = (
            -delta_x * sine + delta_y * cosine
        ) / max(ellipse_height * 0.5, 1e-6)
        radial_error = np.abs(
            np.sqrt(
                normalized_x * normalized_x + normalized_y * normalized_y
            )
            - 1.0
        )

        normalized_angles = (
            np.arctan2(normalized_y, normalized_x) + 2.0 * np.pi
        ) % (2.0 * np.pi)
        sector_count = 36
        sectors = np.clip(
            (
                normalized_angles
                / (2.0 * np.pi)
                * sector_count
            ).astype(np.int32),
            0,
            sector_count - 1,
        )
        occupied_count = np.count_nonzero(
            np.bincount(sectors, minlength=sector_count)
        )

        # NumPy's default percentile uses linear interpolation at (n - 1) * q.
        # One partition supplies the four neighbouring ranks needed for both
        # median and P90 without fully sorting every contour's residuals.
        sample_count = radial_error.size
        median_rank = (sample_count - 1) * 0.50
        p90_rank = (sample_count - 1) * 0.90
        median_low = int(median_rank)
        median_high = min(median_low + 1, sample_count - 1)
        p90_low = int(p90_rank)
        p90_high = min(p90_low + 1, sample_count - 1)
        partitioned = np.partition(
            radial_error,
            tuple(
                sorted(
                    {median_low, median_high, p90_low, p90_high}
                )
            ),
        )
        median_error = partitioned[median_low] + (
            partitioned[median_high] - partitioned[median_low]
        ) * (median_rank - median_low)
        p90_error = partitioned[p90_low] + (
            partitioned[p90_high] - partitioned[p90_low]
        ) * (p90_rank - p90_low)
        return (
            float(occupied_count) / sector_count,
            float(median_error),
            float(p90_error),
        )

    @staticmethod
    def _is_interstitial_gap(candidate: RingDetection, detections: list[RingDetection]) -> bool:
        candidate_area = candidate.ellipse[1][0] * candidate.ellipse[1][1]
        candidate_major = max(candidate.ellipse[1])
        candidate_minor = min(candidate.ellipse[1])
        # A genuine closed hole follows its fitted ellipse closely even when
        # perspective makes it smaller than the surrounding rings.  Dense piles
        # previously lost these small but high-quality holes, after which a weak
        # Hough proposal could appear to replace them.  Inter-ring pockets have
        # a ragged/concave contour and do not achieve this fill quality.
        strong_closed_primary = False
        if candidate.source == "primary":
            contour_area = float(cv2.contourArea(candidate.inner_contour))
            ellipse_area = (
                np.pi
                * float(candidate.ellipse[1][0])
                * float(candidate.ellipse[1][1])
                / 4.0
            )
            strong_closed_primary = (
                contour_area / max(ellipse_area, 1.0) >= 0.88
            )
        # A validated material band cannot be the dark pocket formed between
        # rings. Keep substantial elongated bands and merged U/C side views;
        # small near-square material fragments still pass through the surrounded
        # gap checks below.
        if candidate.source == "flat" and (
            (
                candidate_major >= 45.0
                and candidate_minor >= 16.0
                and candidate_major / max(candidate_minor, 1.0) >= 1.45
            )
            or (candidate_major >= 55.0 and candidate_minor >= 35.0)
        ):
            return False
        proven_rings = [
            detection
            for detection in detections
            if detection is not candidate
            and detection.source in {"primary", "overlap"}
            and detection.confidence >= 0.85
        ]
        if candidate.source == "residual":
            # Weak residual recovery is useful for a substantial bright band,
            # but a small, poorly filled patch near cropped dense rings is only
            # unexplained rim material.  Use the same-frame ring scale so this
            # does not impose a fixed object size.
            if len(proven_rings) >= 3:
                reference_major = float(
                    np.median(
                        [max(detection.ellipse[1]) for detection in proven_rings]
                    )
                )
                contour_area = float(cv2.contourArea(candidate.inner_contour))
                ellipse_area = (
                    np.pi
                    * float(candidate.ellipse[1][0])
                    * float(candidate.ellipse[1][1])
                    / 4.0
                )
                ellipse_fill = contour_area / max(ellipse_area, 1.0)
                if (
                    candidate.confidence <= 0.65
                    and ellipse_fill < 0.50
                    and candidate_major < reference_major * 0.90
                ):
                    return True
            return False
        if candidate.source == "fallback" and len(proven_rings) >= 3:
            reference_major = float(
                np.median(
                    [max(detection.ellipse[1]) for detection in proven_rings]
                )
            )
            nearby_proven = sum(
                np.hypot(
                    candidate.center[0] - detection.center[0],
                    candidate.center[1] - detection.center[1],
                )
                <= candidate_major * 0.75
                for detection in proven_rings
            )
            if (
                candidate_major > reference_major * 1.45
                and nearby_proven >= 2
            ):
                # One long edge ellipse spanning two nearby proven holes is the
                # shared outside of a cropped/touching group, not another ring.
                return True
        larger_neighbours: list[float] = []
        nearby_areas: list[float] = []
        nearby_major_axes: list[float] = []
        other_centres: list[tuple[int, int]] = []
        for other in detections:
            if other is candidate:
                continue
            other_centres.append(other.center)
            other_area = other.ellipse[1][0] * other.ellipse[1][1]
            other_size = max(other.ellipse[1])
            distance = np.hypot(
                candidate.center[0] - other.center[0],
                candidate.center[1] - other.center[1],
            )
            if distance <= max(52, other_size * 1.25):
                nearby_areas.append(other_area)
                nearby_major_axes.append(other_size)
            if other_area >= candidate_area * 2.0 and distance <= other_size * 1.45:
                larger_neighbours.append(other_area)

        # A ring lying on its side loses much of its projected hole area, but one
        # axis still preserves most of the normal ring diameter.  An inter-ring
        # pocket shrinks in both axes.  Keep this side-profile evidence before
        # applying the surrounded-small-region rejection below.
        flat_ring_profile = False
        if len(nearby_major_axes) >= 2:
            reference_major = float(np.median(nearby_major_axes))
            flat_ring_profile = (
                candidate_major >= reference_major * 0.72
                and candidate_minor / max(candidate_major, 1.0) <= 0.66
            )

        if (
            not flat_ring_profile
            and not strong_closed_primary
            and len(larger_neighbours) >= 3
            and candidate_area < float(np.median(larger_neighbours)) * 0.38
        ):
            return True

        # In a compact touching group, a dark pocket bounded by several unrelated
        # rims can be as large as a real hole. It is still strongly concave and
        # fills its fitted ellipse poorly. Apply those quality checks only when
        # the candidate is deep inside the other ring centres; isolated irregular
        # and collapsed rings must remain eligible.
        signed_distance: float | None = None
        if len(other_centres) >= 3:
            hull_points = np.asarray(other_centres, dtype=np.int32).reshape(-1, 1, 2)
            hull = cv2.convexHull(hull_points)
            signed_distance = cv2.pointPolygonTest(hull, candidate.center, True)
        if (
            candidate.source == "primary"
            and len(nearby_areas) >= 3
            and signed_distance is not None
            and signed_distance >= 10.0
        ):
            contour_area = float(cv2.contourArea(candidate.inner_contour))
            ellipse_area = (
                np.pi
                * float(candidate.ellipse[1][0])
                * float(candidate.ellipse[1][1])
                / 4.0
            )
            ellipse_fill = contour_area / max(ellipse_area, 1.0)
            if ellipse_fill < 0.70:
                return True
            if (
                len(nearby_areas) >= 4
                and signed_distance >= 20.0
                and candidate_major / max(candidate_minor, 1.0) >= 1.45
                and ellipse_fill < 0.86
            ):
                # A long smooth void can form where four crowded rims meet.
                # It is larger than the classic tiny diamond pocket, but is
                # still deeply enclosed, elongated and less ellipse-complete
                # than a genuine ring hole.
                return True

        # Keep the original relative-size protection for much smaller diamond
        # pockets. A genuine regular centre ring of normal area is retained.
        if (
            len(other_centres) >= 4
            and len(nearby_areas) >= 3
            and signed_distance is not None
        ):
            reference_area = float(np.median(nearby_areas))
            if (
                not flat_ring_profile
                and not strong_closed_primary
                and signed_distance >= 4.0
                and candidate_area < reference_area * 0.68
            ):
                return True
        return False

    def _fallback_ellipses(
        self,
        frame: np.ndarray,
        cardboard_mask: np.ndarray,
        known_detections: list[RingDetection] | None = None,
    ) -> list[RingDetection]:
        """Recover rings with an open/occluded inner hole from their visible oval rim."""
        gray = (
            self._frame_gray
            if self._frame_gray is not None
            and self._frame_gray.shape == frame.shape[:2]
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        blurred_gray = (
            self._frame_blurred_gray
            if self._frame_blurred_gray is not None
            and self._frame_blurred_gray.shape == frame.shape[:2]
            else cv2.GaussianBlur(gray, (5, 5), 0)
        )
        edges = cv2.Canny(blurred_gray, 35, 100)
        # Only fit edges that are close to material inside the configured field of
        # view. The previous full-frame scan fitted hundreds of rail, cable and
        # workstation contours that could never become valid ring detections.
        material_support = cv2.dilate(
            cardboard_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        edges = cv2.bitwise_and(edges, material_support)
        edges = cv2.bitwise_and(edges, self._roi_mask(edges.shape))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        provisional: list[RingDetection] = []
        height, width = cardboard_mask.shape
        roi_x1, roi_x2 = width * self.roi_left, width * self.roi_right
        roi_y1, roi_y2 = height * self.roi_top, height * self.roi_bottom
        minimum_minor = max(24.0, min(frame.shape[:2]) * 0.045)
        edge_clearance = max(8.0, min(frame.shape[:2]) * 0.02)

        for contour in contours:
            # Touching rings often produce an open arc with almost zero contour
            # area. Arc length plus ellipse-edge support is a better test.
            if len(contour) < 20 or cv2.arcLength(contour, False) < 55:
                continue
            _, _, contour_width, contour_height = cv2.boundingRect(contour)
            if contour_width < 16 or contour_height < 12:
                continue
            ellipse = cv2.fitEllipse(contour)
            major, minor = sorted(ellipse[1], reverse=True)
            if minor < minimum_minor or minor / major < 0.18:
                continue
            # Primary hierarchy holes have already proved these rings. Avoid
            # repeating the comparatively expensive arc-quality and local mask
            # measurements for an edge ellipse that the fusion stage would
            # immediately reject as the same object.
            if known_detections:
                probe = RingDetection(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    contour,
                    None,
                    ellipse,
                    fallback=True,
                    source="fallback",
                    confidence=0.70,
                )
                if any(
                    self._same_ring(probe, known)
                    for known in known_detections
                ):
                    continue

            angular_coverage, median_radial_error, p90_radial_error = (
                self._ellipse_arc_quality(contour, ellipse)
            )
            # True open rings still trace a long, coherent section of one ellipse.
            # Short belt/rail fragments may fit numerically, but either cover too
            # little angle or deviate strongly from the fitted radial profile.
            if (
                angular_coverage < 0.62
                or median_radial_error > 0.16
                or p90_radial_error > 0.50
            ):
                continue

            ellipse_area = np.pi * major * minor / 4
            contour_area = cv2.contourArea(contour)
            contour_fill = contour_area / max(ellipse_area, 1)
            center_x, center_y = ellipse[0]
            roi_rectangle = (
                ((roi_x1 + roi_x2) * 0.5, (roi_y1 + roi_y2) * 0.5),
                (roi_x2 - roi_x1, roi_y2 - roi_y1),
                0.0,
            )
            _, roi_intersection = cv2.rotatedRectangleIntersection(
                ellipse, roi_rectangle
            )
            intersection_area = (
                abs(float(cv2.contourArea(roi_intersection)))
                if roi_intersection is not None
                else 0.0
            )
            ellipse_box_area = max(float(ellipse[1][0] * ellipse[1][1]), 1.0)
            if (
                not roi_x1 + 2.0 < center_x < roi_x2 - 2.0
                or not roi_y1 + 2.0 < center_y < roi_y2 - 2.0
                or intersection_area / ellipse_box_area < 0.72
            ):
                continue

            # Compute mask statistics in the ellipse neighbourhood, not across a
            # new 640x480 array for every contour. This preserves the same tests
            # while removing most allocation and cv2.mean overhead.
            half_extent = max(16, int(np.ceil(major * 0.58)) + 6)
            x0 = max(0, int(center_x) - half_extent)
            x3 = min(width, int(center_x) + half_extent + 1)
            y0 = max(0, int(center_y) - half_extent)
            y3 = min(height, int(center_y) + half_extent + 1)
            local_shape = (y3 - y0, x3 - x0)
            if local_shape[0] <= 0 or local_shape[1] <= 0:
                continue
            local_ellipse = (
                (ellipse[0][0] - x0, ellipse[0][1] - y0),
                ellipse[1],
                ellipse[2],
            )
            rim = np.zeros(local_shape, dtype=np.uint8)
            cv2.ellipse(rim, local_ellipse, 255, max(3, round(minor * 0.07)))
            cardboard_support = cv2.mean(cardboard_mask[y0:y3, x0:x3], mask=rim)[0] / 255
            edge_support = cv2.mean(edges[y0:y3, x0:x3], mask=rim)[0] / 255
            hole = np.zeros(local_shape, dtype=np.uint8)
            hole_axes = (max(3, round(ellipse[1][0] * 0.28)), max(3, round(ellipse[1][1] * 0.28)))
            cv2.ellipse(hole, (local_ellipse[0], hole_axes, ellipse[2]), 255, -1)
            gray_local = gray[y0:y3, x0:x3]
            rim_brightness = cv2.mean(gray_local, mask=rim)[0]
            hole_brightness = cv2.mean(gray_local, mask=hole)[0]
            has_dark_interior = rim_brightness >= hole_brightness + 5
            # A tilted ring can look grey-blue under the Astra auto-exposure and therefore
            # have almost no beige-mask support. A complete, well-fitted oval edge is still
            # reliable when it is away from the ROI border. This recovers upright rings while
            # rejecting the long thin conveyor rails seen in the empty live frame.
            strong_neutral_outline = (
                minor >= 30
                and minor / major >= 0.27
                and 0.72 <= contour_fill <= 1.18
                and roi_x1 + width * 0.025 < center_x < roi_x2 - width * 0.025
                and roi_y1 + height * 0.025 < center_y < roi_y2 - height * 0.025
            )
            strong_partial_outline = (
                minor >= 28
                and minor / major >= 0.22
                and edge_support >= 0.16
                and cardboard_support >= 0.18
                and roi_x1 + width * 0.025 < center_x < roi_x2 - width * 0.025
                and roi_y1 + height * 0.025 < center_y < roi_y2 - height * 0.025
            )
            # Another light cardboard ring can pass directly through this
            # ring's projected hole. In that overlap case the interior is
            # brighter than the rim, but the long coherent ellipse arc and
            # cardboard-supported boundary still prove the occluded ring.
            occluded_interior_outline = (
                28.0 <= minor
                and major <= 140.0
                and minor / major >= 0.22
                and angular_coverage >= 0.82
                and median_radial_error <= 0.11
                and p90_radial_error <= 0.34
                and edge_support >= 0.14
                and cardboard_support >= 0.62
                and roi_x1 + width * 0.025 < center_x < roi_x2 - width * 0.025
                and roi_y1 + height * 0.025 < center_y < roi_y2 - height * 0.025
            )
            if not has_dark_interior and not occluded_interior_outline:
                continue
            if cardboard_support < 0.45 and not strong_neutral_outline and not strong_partial_outline:
                continue

            provisional.append(RingDetection(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                contour,
                None,
                ellipse,
                fallback=True,
                source="fallback",
                confidence=0.70,
            ))

        # Prefer the individual ring ellipses before a larger contour surrounding a touching
        # group. This is important for two or three rings arriving together.
        candidates: list[RingDetection] = []
        for candidate in sorted(provisional, key=lambda item: max(item.ellipse[1])):
            if not any(RingDetector._same_ring(candidate, existing) for existing in candidates):
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _debug_tile(image: np.ndarray, label: str) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (320, 30), (4, 12, 23), -1)
        cv2.putText(tile, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (162, 255, 240), 2, cv2.LINE_AA)
        return tile

    def debug_mosaic(self, input_frame: np.ndarray, final_frame: np.ndarray) -> np.ndarray:
        """Return the six live stages used to diagnose the machine-vision pipeline."""
        height, width = input_frame.shape[:2]
        empty = np.zeros((height, width), dtype=np.uint8)
        material = self._debug_material_mask if self._debug_material_mask is not None else empty
        foreground = self._debug_foreground_mask if self._debug_foreground_mask is not None else empty
        cleaned = self._debug_cleaned_mask if self._debug_cleaned_mask is not None else empty
        watershed = self._debug_watershed_view if self._debug_watershed_view is not None else input_frame
        tiles = [
            self._debug_tile(input_frame, "1  RGB INPUT"),
            self._debug_tile(material, "2  MATERIAL MASK"),
            self._debug_tile(foreground, "3  BACKGROUND CHANGE"),
            self._debug_tile(cleaned, "4  CLEANED MASK"),
            self._debug_tile(watershed, "5  AFTER WATERSHED"),
            self._debug_tile(final_frame, "6  FINAL VISION"),
        ]
        return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:])))

    def _hough_rings(self, frame: np.ndarray, cardboard_mask: np.ndarray) -> list[RingDetection]:
        """Recover near-round rings whose touching rim does not form a closed contour."""
        gray = (
            self._frame_gray
            if self._frame_gray is not None
            and self._frame_gray.shape == frame.shape[:2]
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        )
        height, width = gray.shape
        roi_x1 = max(0, int(width * self.roi_left))
        roi_x2 = min(width, int(np.ceil(width * self.roi_right)))
        roi_y1 = max(0, int(height * self.roi_top))
        roi_y2 = min(height, int(np.ceil(height * self.roi_bottom)))
        gray_roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        if gray_roi.shape[0] < 60 or gray_roi.shape[1] < 60:
            return []
        blurred = cv2.GaussianBlur(gray_roi, (7, 7), 1.3)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(28, round(min(frame.shape[:2]) * 0.065)),
            param1=80,
            param2=22,
            minRadius=max(20, round(min(frame.shape[:2]) * 0.045)),
            maxRadius=max(42, round(min(frame.shape[:2]) * 0.11)),
        )
        if circles is None:
            return []

        edges = cv2.Canny(blurred, 30, 95)
        x1, x2 = width * self.roi_left, width * self.roi_right
        y1, y2 = height * self.roi_top, height * self.roi_bottom
        edge_clearance = max(8.0, min(frame.shape[:2]) * 0.015)
        candidates: list[RingDetection] = []
        angular_samples = 24
        sample_angles = np.linspace(
            0, 2 * np.pi, angular_samples, endpoint=False
        )
        angle_cosines = np.cos(sample_angles)[:, None]
        angle_sines = np.sin(sample_angles)[:, None]
        radius_offsets = np.linspace(-5.0, 5.0, 11)[None, :]
        for x_value, y_value, radius_value in circles[0]:
            x = int(round(x_value)) + roi_x1
            y = int(round(y_value)) + roi_y1
            radius = int(round(radius_value))
            if not (
                x1 + radius + edge_clearance
                < x
                < x2 - radius - edge_clearance
                and y1 + radius + edge_clearance
                < y
                < y2 - radius - edge_clearance
            ):
                # Hough is especially vulnerable to a partial ring arc plus a
                # straight conveyor rail. Require a small complete-circle margin
                # before this recovery path may create an independent object.
                continue

            patch_radius = radius + 6
            x0, x3 = max(roi_x1, x - patch_radius), min(roi_x2, x + patch_radius + 1)
            y0, y3 = max(roi_y1, y - patch_radius), min(roi_y2, y + patch_radius + 1)
            local_shape = (y3 - y0, x3 - x0)
            local_center = (x - x0, y - y0)
            outer_disk = np.zeros(local_shape, dtype=np.uint8)
            inner_disk = np.zeros(local_shape, dtype=np.uint8)
            cv2.circle(outer_disk, local_center, radius + 4, 255, -1)
            cv2.circle(inner_disk, local_center, max(4, radius - 5), 255, -1)
            rim = cv2.subtract(outer_disk, inner_disk)
            hole = np.zeros(local_shape, dtype=np.uint8)
            cv2.circle(hole, local_center, max(4, round(radius * 0.52)), 255, -1)
            gray_local = gray[y0:y3, x0:x3]
            edges_local = edges[y0 - roi_y1:y3 - roi_y1, x0 - roi_x1:x3 - roi_x1]
            material_local = cardboard_mask[y0:y3, x0:x3]
            rim_brightness = cv2.mean(gray_local, mask=rim)[0]
            hole_brightness = cv2.mean(gray_local, mask=hole)[0]
            edge_support = cv2.mean(edges_local, mask=rim)[0] / 255
            material_support = cv2.mean(material_local, mask=rim)[0] / 255
            sample_radii = float(radius) + radius_offsets
            sample_x = np.rint(
                local_center[0] + angle_cosines * sample_radii
            ).astype(np.int32)
            sample_y = np.rint(
                local_center[1] + angle_sines * sample_radii
            ).astype(np.int32)
            valid = (
                (sample_x >= 0)
                & (sample_x < local_shape[1])
                & (sample_y >= 0)
                & (sample_y < local_shape[0])
            )
            edge_hits = np.zeros(valid.shape, dtype=bool)
            edge_hits[valid] = (
                edges_local[sample_y[valid], sample_x[valid]] > 0
            )
            angular_coverage = float(np.mean(np.any(edge_hits, axis=1)))
            strong_circle = (
                edge_support >= 0.08
                # A real standalone ring contributes a coherent circumference.
                # Lower coverage allowed nearby ring arcs and conveyor edges to
                # vote for a fake circle in the middle of dense piles.
                and angular_coverage >= 0.67
                and material_support >= 0.26
            )
            partial_overlap_circle = (
                edge_support >= 0.055
                and angular_coverage >= 0.40
                and material_support >= 0.35
            )
            if rim_brightness < hole_brightness + 10 or not (
                strong_circle or partial_overlap_circle
            ):
                continue

            contour = cv2.ellipse2Poly((x, y), (radius, radius), 0, 0, 360, 5).reshape(-1, 1, 2)
            ellipse = ((float(x), float(y)), (float(radius * 2), float(radius * 2)), 0.0)
            candidate = RingDetection(
                (x, y),
                contour,
                None,
                ellipse,
                fallback=True,
                source="hough",
                confidence=0.76 if strong_circle else 0.58,
            )
            if not any(self._same_ring(candidate, existing) for existing in candidates):
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _watershed_ovals(
        frame: np.ndarray,
        mask: np.ndarray,
        known: list[RingDetection],
    ) -> list[RingDetection]:
        """Split touching rings with inner-hole and distance-peak watershed markers."""
        height, width = mask.shape
        image_area = height * width
        minimum_area = image_area * 0.003
        maximum_area = image_area * 0.40
        external, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        solid = np.zeros_like(mask)
        valid_external: list[np.ndarray] = []
        for contour in external:
            area = cv2.contourArea(contour)
            if minimum_area <= area <= maximum_area:
                cv2.drawContours(solid, [contour], -1, 255, cv2.FILLED)
                valid_external.append(contour)
        if not valid_external:
            return []

        markers = np.zeros(mask.shape, dtype=np.int32)
        markers[solid == 0] = 1
        next_label = 2
        marker_points: list[tuple[int, int]] = []

        # Strong markers: the independent holes of normal/overlapping rings.
        for detection in known:
            x, y = detection.center
            if 0 <= x < width and 0 <= y < height and solid[y, x]:
                cv2.circle(markers, (x, y), max(4, round(min(height, width) * 0.012)), next_label, -1)
                marker_points.append((x, y))
                next_label += 1

        # Recovery markers: true local maxima. Unlike one global distance threshold,
        # this creates a marker for each ring in a chain of touching rings.
        for contour in valid_external:
            contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)
            local_contour = contour - np.array([[[contour_x, contour_y]]], dtype=contour.dtype)
            component = np.zeros((contour_height, contour_width), dtype=np.uint8)
            cv2.drawContours(component, [local_contour], -1, 255, cv2.FILLED)
            distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
            if distance.max() <= 0:
                continue
            minimum_distance = max(18, round(min(height, width) * 0.045))
            peak_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (minimum_distance * 2 + 1, minimum_distance * 2 + 1)
            )
            local_maximum = distance >= cv2.dilate(distance, peak_kernel)
            strong_enough = distance >= 0.34 * distance.max()
            peak_mask = np.uint8(local_maximum & strong_enough & (component > 0)) * 255
            peak_count, _, stats, centroids = cv2.connectedComponentsWithStats(peak_mask)
            # Process the strongest peaks first, then enforce non-maximum suppression.
            ranked_peaks = sorted(
                range(1, peak_count), key=lambda label: distance[int(centroids[label][1]), int(centroids[label][0])], reverse=True
            )
            for peak_label in ranked_peaks:
                if stats[peak_label, cv2.CC_STAT_AREA] < 2:
                    continue
                local_x, local_y = (int(value) for value in centroids[peak_label])
                x, y = local_x + contour_x, local_y + contour_y
                if any(np.hypot(x - px, y - py) < max(25, min(height, width) * 0.06) for px, py in marker_points):
                    continue
                cv2.circle(markers, (x, y), max(4, round(min(height, width) * 0.012)), next_label, -1)
                marker_points.append((x, y))
                next_label += 1

        if next_label == 2:
            return []

        markers = cv2.watershed(frame.copy(), markers)
        candidates: list[RingDetection] = []
        border_x, border_y = width * 0.05, height * 0.05
        for label in range(2, next_label):
            region = np.uint8(markers == label) * 255
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            if len(contour) < 5 or not minimum_area <= area <= maximum_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if x <= border_x or y <= border_y or x + w >= width - border_x or y + h >= height - border_y:
                continue
            ellipse = cv2.fitEllipse(contour)
            major, minor = sorted(ellipse[1], reverse=True)
            if minor < 18 or minor / major < 0.14:
                continue
            candidate = RingDetection(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                contour,
                contour,
                ellipse,
                fallback=True,
                source="watershed",
                confidence=0.56,
            )
            if not any(RingDetector._same_ring(candidate, existing) for existing in candidates):
                candidates.append(candidate)
        return candidates

    def _flat_material_profiles(self, frame: np.ndarray) -> list[RingDetection]:
        """Recover solid side-on rings from multi-scale material cores and edge pairs."""
        if self.background is None or self._foreground_mask is None:
            return []

        height, width = frame.shape[:2]
        hsv = (
            self._frame_hsv
            if self._frame_hsv is not None
            and self._frame_hsv.shape[:2] == frame.shape[:2]
            else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        )
        roi = self._roi_mask((height, width))
        beige_material = cv2.inRange(
            hsv, np.array((4, 8, 75)), np.array((42, 235, 255))
        )
        neutral_bright = cv2.inRange(
            hsv, np.array((0, 0, 100)), np.array((179, 72, 255))
        )
        # A fully side-on ring has no hole, but still has a thick, light cardboard
        # surface.  Foreground agreement is mandatory so static rails and belt
        # texture cannot enter this recovery path.
        flat_mask = cv2.bitwise_or(beige_material, neutral_bright)
        foreground_support = cv2.dilate(
            self._foreground_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        flat_mask = cv2.bitwise_and(flat_mask, foreground_support)
        flat_mask = cv2.bitwise_and(flat_mask, roi)
        flat_mask = cv2.morphologyEx(
            flat_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        image_area = height * width
        minimum_core_area = image_area * 0.00016
        maximum_core_area = image_area * 0.030
        if cv2.countNonZero(flat_mask) < minimum_core_area:
            return []
        # All valid proposals must remain inside the configured conveyor ROI.
        # Run the expensive erosion/directional kernels on that crop instead of
        # processing the zero-filled remainder of the 640x480 camera frame.
        # Twenty pixels preserve the largest kernel's border behaviour.
        work_padding = 20
        work_x1 = max(0, int(width * self.roi_left) - work_padding)
        work_x2 = min(
            width, int(np.ceil(width * self.roi_right)) + work_padding
        )
        work_y1 = max(0, int(height * self.roi_top) - work_padding)
        work_y2 = min(
            height, int(np.ceil(height * self.roi_bottom)) + work_padding
        )
        working_flat_mask = flat_mask[work_y1:work_y2, work_x1:work_x2]
        work_height, work_width = working_flat_mask.shape
        candidates: list[RingDetection] = []

        # Different camera angles leave very different side-wall thicknesses.
        # Large erosions separate touching bands; smaller erosions recover narrow
        # or curved bands that the former fixed 11-pixel erosion deleted.
        for kernel_size in (11, 9, 7, 5):
            core_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            core = cv2.erode(working_flat_mask, core_kernel, iterations=1)
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(core)
            for label in range(1, component_count):
                core_area = float(stats[label, cv2.CC_STAT_AREA])
                if not minimum_core_area <= core_area <= maximum_core_area:
                    continue
                component_x = int(stats[label, cv2.CC_STAT_LEFT])
                component_y = int(stats[label, cv2.CC_STAT_TOP])
                component_width = int(stats[label, cv2.CC_STAT_WIDTH])
                component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
                padding = kernel_size // 2 + 2
                crop_x1 = max(0, component_x - padding)
                crop_y1 = max(0, component_y - padding)
                crop_x2 = min(work_width, component_x + component_width + padding)
                crop_y2 = min(work_height, component_y + component_height + padding)
                component = (
                    np.uint8(
                        labels[crop_y1:crop_y2, crop_x1:crop_x2] == label
                    )
                    * 255
                )
                profile = cv2.dilate(component, core_kernel, iterations=1)
                contours, _ = cv2.findContours(
                    profile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    continue
                contour = max(contours, key=cv2.contourArea)
                contour = contour + np.array(
                    [[[crop_x1 + work_x1, crop_y1 + work_y1]]],
                    dtype=contour.dtype,
                )
                candidate = self._validated_flat_candidate(
                    hsv[:, :, 2],
                    flat_mask,
                    foreground_support,
                    beige_material,
                    contour,
                    confidence=0.90,
                )
                if candidate is not None:
                    self._append_flat_candidate(candidates, candidate)

        # A dense pile can join two side-on rings into one component before the
        # isotropic erosion above. Probe the material with long kernels at
        # several angles: a real cardboard band survives along its own axis,
        # while a crossing neighbour is removed by the directional opening.
        # This adds one proposal per visible band without assuming that every
        # ring is horizontal or that its two long edges remain unobstructed.
        line_kernel_size = 35
        line_kernel_center = line_kernel_size // 2
        line_half_length = 15
        # Six probes cover every undirected orientation within 15 degrees,
        # which is inside the 18-degree acceptance tolerance below. The
        # parallel-edge branch remains a continuous-angle backup.
        for probe_angle in range(0, 180, 30):
            radians = np.deg2rad(float(probe_angle))
            direction = np.array(
                (np.cos(radians), np.sin(radians)), dtype=np.float32
            )
            start = np.rint(
                np.array((line_kernel_center, line_kernel_center), dtype=np.float32)
                - direction * line_half_length
            ).astype(np.int32)
            end = np.rint(
                np.array((line_kernel_center, line_kernel_center), dtype=np.float32)
                + direction * line_half_length
            ).astype(np.int32)
            directional_kernel = np.zeros(
                (line_kernel_size, line_kernel_size), dtype=np.uint8
            )
            cv2.line(
                directional_kernel,
                tuple(start),
                tuple(end),
                1,
                5,
                cv2.LINE_8,
            )
            oriented = cv2.morphologyEx(
                working_flat_mask,
                cv2.MORPH_OPEN,
                directional_kernel,
                iterations=1,
            )
            oriented = cv2.morphologyEx(
                oriented,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            oriented_contours, _ = cv2.findContours(
                oriented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in oriented_contours:
                contour_area = float(cv2.contourArea(contour))
                if contour_area < minimum_core_area * 1.4:
                    continue
                contour = contour + np.array(
                    [[[work_x1, work_y1]]], dtype=contour.dtype
                )
                rect = cv2.minAreaRect(contour)
                rect_major, rect_minor = sorted(rect[1], reverse=True)
                rectangular_extent = contour_area / max(
                    rect_major * rect_minor, 1.0
                )
                # Directional openings also retain short, nearly straight pieces
                # of an ordinary oval rim. A true solid side profile is a broad,
                # substantially filled band; keep narrow ring-wall arcs on the
                # hierarchy/ellipse path where they belong.
                if (
                    rect_minor < 22.0
                    or rect_major / max(rect_minor, 1.0) < 1.65
                    or rectangular_extent < 0.70
                ):
                    continue
                rect_edges = np.roll(
                    cv2.boxPoints(rect), -1, axis=0
                ) - cv2.boxPoints(rect)
                long_edge = rect_edges[
                    int(np.argmax(np.linalg.norm(rect_edges, axis=1)))
                ]
                contour_angle = float(
                    np.rad2deg(np.arctan2(long_edge[1], long_edge[0])) % 180.0
                )
                angle_delta = abs(contour_angle - float(probe_angle))
                angle_delta = min(angle_delta, 180.0 - angle_delta)
                if angle_delta > 18.0:
                    continue
                candidate = self._validated_flat_candidate(
                    hsv[:, :, 2],
                    flat_mask,
                    foreground_support,
                    beige_material,
                    contour,
                    confidence=0.88,
                )
                if candidate is not None:
                    self._append_flat_candidate(candidates, candidate)

        # Curved or partly occluded bands may not retain one connected erosion
        # core.  Two long, parallel material edges are independent evidence for
        # the same solid side wall.
        flat_edges = cv2.Canny(working_flat_mask, 35, 95)
        lines = cv2.HoughLinesP(
            flat_edges,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=28,
            maxLineGap=10,
        )
        line_records: list[
            tuple[np.ndarray, float, float, np.ndarray, np.ndarray]
        ] = []
        line_clearance = max(16.0, min(height, width) * 0.03)
        line_roi_x1 = width * self.roi_left + line_clearance
        line_roi_x2 = width * self.roi_right - line_clearance
        line_roi_y1 = height * self.roi_top + line_clearance
        line_roi_y2 = height * self.roi_bottom - line_clearance
        if lines is not None:
            for raw_line in lines.reshape(-1, 4):
                offset = np.array((work_x1, work_y1), dtype=np.float32)
                start = raw_line[:2].astype(np.float32) + offset
                end = raw_line[2:].astype(np.float32) + offset
                vector = end - start
                length = float(np.linalg.norm(vector))
                if not 28.0 <= length <= 170.0:
                    continue
                angle = float(np.arctan2(vector[1], vector[0]) % np.pi)
                direction = np.array((np.cos(angle), np.sin(angle)), dtype=np.float32)
                midpoint = (start + end) * 0.5
                # Rail and ROI-border lines were consuming the strongest-line
                # budget and intermittently hiding shorter U-shaped ring walls.
                if not (
                    line_roi_x1 < float(midpoint[0]) < line_roi_x2
                    and line_roi_y1 < float(midpoint[1]) < line_roi_y2
                ):
                    continue
                line_records.append((direction, angle, length, midpoint, np.vstack((start, end))))
        # Keep the strongest independent edge evidence while bounding the
        # quadratic edge-pair search on textured cardboard and noisy belt frames.
        line_records.sort(key=lambda record: record[2], reverse=True)
        line_records = line_records[:64]

        for first_index, first in enumerate(line_records):
            first_direction, first_angle, _, first_midpoint, first_points = first
            for second in line_records[first_index + 1 :]:
                _, second_angle, _, second_midpoint, second_points = second
                angle_delta = abs(first_angle - second_angle)
                angle_delta = min(angle_delta, np.pi - angle_delta)
                if angle_delta > np.deg2rad(9.0):
                    continue
                normal = np.array(
                    (-first_direction[1], first_direction[0]), dtype=np.float32
                )
                separation = abs(float(np.dot(second_midpoint - first_midpoint, normal)))
                if not 8.0 <= separation <= 48.0:
                    continue
                first_projection = first_points @ first_direction
                second_projection = second_points @ first_direction
                overlap = min(float(first_projection.max()), float(second_projection.max())) - max(
                    float(first_projection.min()), float(second_projection.min())
                )
                if overlap < 28.0:
                    continue
                all_points = np.vstack((first_points, second_points))
                along = all_points @ first_direction
                across = all_points @ normal
                major = float(along.max() - along.min()) + 4.0
                minor = float(across.max() - across.min()) + 4.0
                center = (
                    first_direction * ((along.min() + along.max()) * 0.5)
                    + normal * ((across.min() + across.max()) * 0.5)
                )
                rect = (
                    (float(center[0]), float(center[1])),
                    (major, minor),
                    float(np.rad2deg(first_angle)),
                )
                contour = np.int32(np.round(cv2.boxPoints(rect))).reshape(-1, 1, 2)
                candidate = self._validated_flat_candidate(
                    hsv[:, :, 2],
                    flat_mask,
                    foreground_support,
                    beige_material,
                    contour,
                    confidence=0.82,
                )
                if candidate is not None:
                    self._append_flat_candidate(candidates, candidate)
        return self._merge_connected_side_walls(candidates, flat_mask)

    @staticmethod
    def _major_axis_geometry(
        candidate: RingDetection,
    ) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        """Return stable major/normal axes for a rotated-rectangle proposal."""
        box = cv2.boxPoints(candidate.ellipse).astype(np.float32)
        edges = np.roll(box, -1, axis=0) - box
        lengths = np.linalg.norm(edges, axis=1)
        index = int(np.argmax(lengths))
        major = max(float(lengths[index]), 1.0)
        direction = edges[index] / major
        if direction[0] < 0 or (abs(float(direction[0])) < 1e-6 and direction[1] < 0):
            direction = -direction
        normal = np.array((-direction[1], direction[0]), dtype=np.float32)
        minor = max(float(np.min(lengths)), 1.0)
        center = np.asarray(candidate.ellipse[0], dtype=np.float32)
        return direction, normal, major, minor, center

    @staticmethod
    def _oriented_fill(
        mask: np.ndarray,
        direction: np.ndarray,
        normal: np.ndarray,
        along: float,
        across: float,
        length: float,
        width: float,
    ) -> float:
        """Measure mask coverage inside one small oriented sampling rectangle."""
        if length <= 1.0 or width <= 1.0:
            return 0.0
        center = direction * along + normal * across
        angle = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
        rectangle = (
            (float(center[0]), float(center[1])),
            (float(length), float(width)),
            angle,
        )
        region = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(region, np.int32(np.round(cv2.boxPoints(rectangle))), 255)
        area = cv2.countNonZero(region)
        if area <= 0:
            return 0.0
        return cv2.countNonZero(cv2.bitwise_and(mask, region)) / float(area)

    def _merge_connected_side_walls(
        self, candidates: list[RingDetection], material_mask: np.ndarray
    ) -> list[RingDetection]:
        """Merge two connected parallel walls into one tilted open-ring proposal.

        A steep ring can expose both opposite cardboard walls. Counting each wall
        separately produced the 11-for-8 error seen in the live pile. Two unrelated
        side-on rings remain separate unless material actually bridges the dark gap
        at an end, as it does around a real U/C-shaped ring.
        """
        if len(candidates) < 2:
            return candidates

        pair_proposals: list[tuple[float, int, int, RingDetection]] = []
        for first_index, first in enumerate(candidates):
            first_direction, first_normal, first_major, first_minor, first_center = (
                self._major_axis_geometry(first)
            )
            for second_index in range(first_index + 1, len(candidates)):
                second = candidates[second_index]
                (
                    second_direction,
                    _,
                    second_major,
                    second_minor,
                    second_center,
                ) = self._major_axis_geometry(second)
                alignment = abs(float(np.dot(first_direction, second_direction)))
                if alignment < np.cos(np.deg2rad(12.0)):
                    continue

                center_delta = second_center - first_center
                along_offset = abs(float(np.dot(center_delta, first_direction)))
                separation = abs(float(np.dot(center_delta, first_normal)))
                average_minor = (first_minor + second_minor) * 0.5
                dark_gap_width = separation - average_minor
                minimum_major = min(first_major, second_major)
                overlap = minimum_major - along_offset
                if (
                    not 7.0 <= dark_gap_width <= 48.0
                    or overlap < minimum_major * 0.52
                    or along_offset > minimum_major * 0.48
                ):
                    continue

                first_along = float(np.dot(first_center, first_direction))
                second_along = float(np.dot(second_center, first_direction))
                common_start = max(
                    first_along - first_major * 0.5,
                    second_along - second_major * 0.5,
                )
                common_end = min(
                    first_along + first_major * 0.5,
                    second_along + second_major * 0.5,
                )
                common_length = common_end - common_start
                if common_length < minimum_major * 0.50:
                    continue

                across_midpoint = (
                    float(np.dot(first_center, first_normal))
                    + float(np.dot(second_center, first_normal))
                ) * 0.5
                sample_length = float(np.clip(common_length * 0.20, 10.0, 20.0))
                sample_width = max(5.0, dark_gap_width * 0.86)
                start_fill = self._oriented_fill(
                    material_mask,
                    first_direction,
                    first_normal,
                    common_start + sample_length * 0.5,
                    across_midpoint,
                    sample_length,
                    sample_width,
                )
                end_fill = self._oriented_fill(
                    material_mask,
                    first_direction,
                    first_normal,
                    common_end - sample_length * 0.5,
                    across_midpoint,
                    sample_length,
                    sample_width,
                )
                middle_fill = self._oriented_fill(
                    material_mask,
                    first_direction,
                    first_normal,
                    (common_start + common_end) * 0.5,
                    across_midpoint,
                    max(10.0, common_length * 0.34),
                    sample_width,
                )
                bridge_fill = max(start_fill, end_fill)
                if bridge_fill < 0.34 or middle_fill > 0.42:
                    continue
                if bridge_fill - middle_fill < 0.16:
                    continue

                first_box = cv2.boxPoints(first.ellipse).astype(np.float32)
                second_box = cv2.boxPoints(second.ellipse).astype(np.float32)
                all_points = np.vstack((first_box, second_box))
                along_values = all_points @ first_direction
                across_values = all_points @ first_normal
                merged_major = float(along_values.max() - along_values.min())
                merged_minor = float(across_values.max() - across_values.min())
                if merged_major > 170.0 or merged_minor > 110.0:
                    continue
                merged_center = (
                    first_direction * ((along_values.min() + along_values.max()) * 0.5)
                    + first_normal * ((across_values.min() + across_values.max()) * 0.5)
                )
                merged_angle = float(
                    np.rad2deg(np.arctan2(first_direction[1], first_direction[0]))
                )
                merged_ellipse = (
                    (float(merged_center[0]), float(merged_center[1])),
                    (merged_major, merged_minor),
                    merged_angle,
                )
                merged_contour = np.int32(
                    np.round(cv2.boxPoints(merged_ellipse))
                ).reshape(-1, 1, 2)
                merged = RingDetection(
                    (int(round(merged_center[0])), int(round(merged_center[1]))),
                    merged_contour,
                    merged_contour,
                    merged_ellipse,
                    fallback=True,
                    source="flat",
                    confidence=max(first.confidence, second.confidence, 0.92),
                )
                pair_proposals.append(
                    (bridge_fill - middle_fill, first_index, second_index, merged)
                )

        consumed: set[int] = set()
        merged_candidates: list[RingDetection] = []
        for _, first_index, second_index, merged in sorted(
            pair_proposals, key=lambda proposal: proposal[0], reverse=True
        ):
            if first_index in consumed or second_index in consumed:
                continue
            consumed.update((first_index, second_index))
            merged_candidates.append(merged)
        merged_candidates.extend(
            candidate
            for index, candidate in enumerate(candidates)
            if index not in consumed
        )
        return merged_candidates

    def _validated_flat_candidate(
        self,
        value_channel: np.ndarray,
        flat_mask: np.ndarray,
        foreground_support: np.ndarray,
        beige_material: np.ndarray,
        contour: np.ndarray,
        confidence: float,
    ) -> RingDetection | None:
        """Validate one solid band proposal with geometry, colour and foreground."""
        height, width = flat_mask.shape
        area = float(cv2.contourArea(contour))
        if not 220.0 <= area <= 8000.0:
            return None
        rect = cv2.minAreaRect(contour)
        major, minor = sorted(rect[1], reverse=True)
        if not 32.0 <= major <= 165.0 or not 8.0 <= minor <= 60.0:
            return None
        # A pair of noisy vertical belt/rail edges can form a convincing but
        # extremely thin rectangle over otherwise empty belt. Multi-scale
        # material cores have stronger evidence (0.90); apply this guard only
        # to the weaker Hough-line-pair path (0.82).
        if confidence < 0.85 and minor < 18.0:
            return None
        aspect = major / max(minor, 1.0)
        if not 1.25 <= aspect <= 8.5:
            return None
        extent = area / max(major * minor, 1.0)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / max(hull_area, 1.0)
        if extent < 0.38 or solidity < 0.68:
            return None

        box = cv2.boxPoints(rect)
        roi_x1, roi_x2 = width * self.roi_left, width * self.roi_right
        roi_y1, roi_y2 = height * self.roi_top, height * self.roi_bottom
        # Flat edge fragments next to the bright conveyor rail are especially
        # convincing false bands. Require the whole side profile to be farther
        # inside the configured belt ROI; a genuine ring is counted once its
        # complete projected wall is visible.
        edge_clearance = max(16.0, min(height, width) * 0.03)
        if (
            float(np.min(box[:, 0])) <= roi_x1 + edge_clearance
            or float(np.max(box[:, 0])) >= roi_x2 - edge_clearance
            or float(np.min(box[:, 1])) <= roi_y1 + edge_clearance
            or float(np.max(box[:, 1])) >= roi_y2 - edge_clearance
            or major > (roi_x2 - roi_x1) * 0.55
        ):
            return None

        box_points = np.int32(np.round(box))
        region_x, region_y, region_width, region_height = cv2.boundingRect(
            box_points
        )
        region_x = max(0, region_x)
        region_y = max(0, region_y)
        region_width = min(region_width, width - region_x)
        region_height = min(region_height, height - region_y)
        if region_width <= 0 or region_height <= 0:
            return None
        region = np.zeros((region_height, region_width), dtype=np.uint8)
        local_box = box_points - np.array((region_x, region_y), dtype=np.int32)
        cv2.fillConvexPoly(region, local_box, 255)
        region_area = max(float(cv2.countNonZero(region)), 1.0)
        y_slice = slice(region_y, region_y + region_height)
        x_slice = slice(region_x, region_x + region_width)
        material_fill = (
            cv2.countNonZero(
                cv2.bitwise_and(flat_mask[y_slice, x_slice], region)
            )
            / region_area
        )
        foreground_fill = (
            cv2.countNonZero(
                cv2.bitwise_and(foreground_support[y_slice, x_slice], region)
            )
            / region_area
        )
        beige_fill = (
            cv2.countNonZero(
                cv2.bitwise_and(beige_material[y_slice, x_slice], region)
            )
            / region_area
        )
        mean_value = cv2.mean(
            value_channel[y_slice, x_slice], mask=region
        )[0]
        minimum_mean_value = 118.0 if confidence < 0.90 else 108.0
        if (
            material_fill < 0.55
            or foreground_fill < 0.62
            or mean_value < minimum_mean_value
        ):
            return None

        edges = np.roll(box, -1, axis=0) - box
        edge_lengths = np.linalg.norm(edges, axis=1)
        long_edge = edges[int(np.argmax(edge_lengths))]
        orientation = float(np.rad2deg(np.arctan2(long_edge[1], long_edge[0])) % 180.0)
        horizontal_delta = min(orientation, 180.0 - orientation)
        # The moving belt joint is long, almost horizontal and dark/neutral.  A
        # real horizontal side wall remains compact and strongly cardboard-like.
        if (
            horizontal_delta <= 12.0
            and major > 125.0
            and (beige_fill < 0.70 or mean_value < 120.0)
        ):
            return None

        center_x, center_y = rect[0]
        return RingDetection(
            (int(round(center_x)), int(round(center_y))),
            contour,
            contour,
            (rect[0], rect[1], rect[2]),
            fallback=True,
            source="flat",
            confidence=confidence,
        )

    @staticmethod
    def _rotated_rect_iou(
        first: tuple[tuple[float, float], tuple[float, float], float],
        second: tuple[tuple[float, float], tuple[float, float], float],
    ) -> float:
        first_area = max(float(first[1][0] * first[1][1]), 1.0)
        second_area = max(float(second[1][0] * second[1][1]), 1.0)
        _, intersection = cv2.rotatedRectangleIntersection(first, second)
        if intersection is None:
            return 0.0
        intersection_area = abs(float(cv2.contourArea(intersection)))
        return intersection_area / max(first_area + second_area - intersection_area, 1.0)

    @classmethod
    def _append_flat_candidate(
        cls, candidates: list[RingDetection], candidate: RingDetection
    ) -> None:
        """Add a flat candidate unless a stronger proposal covers the same band."""
        for index, existing in enumerate(candidates):
            center_distance = float(
                np.hypot(
                    candidate.center[0] - existing.center[0],
                    candidate.center[1] - existing.center[1],
                )
            )
            duplicate = (
                cls._rotated_rect_iou(candidate.ellipse, existing.ellipse) >= 0.42
                or center_distance
                < max(7.0, min(min(candidate.ellipse[1]), min(existing.ellipse[1])) * 0.60)
            )
            if not duplicate:
                continue
            if candidate.confidence > existing.confidence:
                candidates[index] = candidate
            return
        candidates.append(candidate)

    @staticmethod
    def _center_overlaps_flat_profile(
        candidate: RingDetection, flat_profile: RingDetection
    ) -> bool:
        """Return whether an oval centre lies on a detected solid side wall."""
        center, size, angle = flat_profile.ellipse
        guard_rect = (
            center,
            (float(size[0]) + 18.0, float(size[1]) + 18.0),
            angle,
        )
        guard = np.int32(np.round(cv2.boxPoints(guard_rect))).reshape(-1, 1, 2)
        return cv2.pointPolygonTest(guard, candidate.center, False) >= 0

    @staticmethod
    def _is_oversized_interior_flat_bridge(
        flat_candidate: RingDetection, detections: list[RingDetection]
    ) -> bool:
        """Reject a long directional strip formed between dense ring columns."""
        proven = [
            detection
            for detection in detections
            if detection.source in {"primary", "overlap"}
            and detection.confidence >= 0.85
        ]
        if len(proven) < 4:
            return False
        centers = np.asarray(
            [detection.center for detection in proven], dtype=np.int32
        ).reshape(-1, 1, 2)
        hull = cv2.convexHull(centers)
        if cv2.pointPolygonTest(hull, flat_candidate.center, True) < 4.0:
            return False
        reference_major = float(
            np.median([max(detection.ellipse[1]) for detection in proven])
        )
        return max(flat_candidate.ellipse[1]) > reference_major * 1.70

    @staticmethod
    def _is_undersized_dense_flat_fragment(
        flat_candidate: RingDetection, detections: list[RingDetection]
    ) -> bool:
        """Reject a short rim fragment when the scene provides a size reference."""
        proven = [
            detection
            for detection in detections
            if detection.source in {"primary", "overlap"}
            and detection.confidence >= 0.85
        ]
        if len(proven) < 3:
            return False
        reference_major = float(
            np.median([max(detection.ellipse[1]) for detection in proven])
        )
        flat_major = max(flat_candidate.ellipse[1])
        return flat_major < max(42.0, reference_major * 0.82)

    @staticmethod
    def _is_duplicate_open_ring_wall(
        flat_candidate: RingDetection, detections: list[RingDetection]
    ) -> bool:
        """Reject a broad flat proposal that duplicates one proven open ring."""
        flat_major, flat_minor = sorted(
            flat_candidate.ellipse[1], reverse=True
        )
        if flat_major / max(flat_minor, 1.0) > 2.25:
            return False

        for ring in detections:
            if ring.source not in {"primary", "overlap"}:
                continue
            ring_major, ring_minor = sorted(ring.ellipse[1], reverse=True)
            center_distance = float(
                np.hypot(
                    flat_candidate.center[0] - ring.center[0],
                    flat_candidate.center[1] - ring.center[1],
                )
            )
            if (
                center_distance
                <= max(18.0, min(flat_major, ring_major) * 0.55)
                and ring_major * 0.70
                <= flat_major
                <= ring_major * 1.30
                and ring_minor * 0.60
                <= flat_minor
                <= ring_minor * 1.55
            ):
                return True
        return False

    @staticmethod
    def _ellipse_radius_and_tangent(
        point: tuple[float, float],
        ellipse: tuple[tuple[float, float], tuple[float, float], float],
    ) -> tuple[float, float | None]:
        """Return normalized ellipse radius and local tangent angle at a point."""
        (center_x, center_y), (width, height), angle = ellipse
        radians = np.deg2rad(angle)
        cosine = float(np.cos(radians))
        sine = float(np.sin(radians))
        delta_x = float(point[0] - center_x)
        delta_y = float(point[1] - center_y)
        local_x = delta_x * cosine + delta_y * sine
        local_y = -delta_x * sine + delta_y * cosine
        radius_x = max(float(width) * 0.5, 1e-6)
        radius_y = max(float(height) * 0.5, 1e-6)
        normalized_radius = float(
            np.hypot(local_x / radius_x, local_y / radius_y)
        )

        gradient_x = local_x / (radius_x * radius_x)
        gradient_y = local_y / (radius_y * radius_y)
        tangent_local_x = -gradient_y
        tangent_local_y = gradient_x
        if np.hypot(tangent_local_x, tangent_local_y) < 1e-9:
            return normalized_radius, None
        tangent_x = tangent_local_x * cosine - tangent_local_y * sine
        tangent_y = tangent_local_x * sine + tangent_local_y * cosine
        tangent_angle = float(np.degrees(np.arctan2(tangent_y, tangent_x)) % 180.0)
        return normalized_radius, tangent_angle

    @classmethod
    def _is_tangent_rim_fragment(
        cls, flat_candidate: RingDetection, ring: RingDetection
    ) -> bool:
        """Identify a short flat proposal cut from the tangent wall of a ring."""
        if ring.source not in {
            "primary",
            "fallback",
            "hough",
            "material",
            "overlap",
        }:
            return False
        flat_major = max(flat_candidate.ellipse[1])
        flat_minor = min(flat_candidate.ellipse[1])
        ring_major = max(ring.ellipse[1])
        if (
            flat_major > ring_major * 1.15
            or flat_minor > ring_major * 0.45
        ):
            return False

        normalized_radius, tangent_angle = cls._ellipse_radius_and_tangent(
            flat_candidate.ellipse[0], ring.ellipse
        )
        if tangent_angle is None or not 0.85 <= normalized_radius <= 2.05:
            return False

        box = cv2.boxPoints(flat_candidate.ellipse).astype(np.float64)
        edges = np.roll(box, -1, axis=0) - box
        edge_lengths = np.linalg.norm(edges, axis=1)
        longest_edge = edges[int(np.argmax(edge_lengths))]
        if np.hypot(longest_edge[0], longest_edge[1]) < 1e-9:
            return False
        flat_angle = float(
            np.degrees(np.arctan2(longest_edge[1], longest_edge[0])) % 180.0
        )
        raw_difference = abs(flat_angle - tangent_angle)
        angle_difference = min(raw_difference, 180.0 - raw_difference)
        return angle_difference <= 40.0

    def _material_ovals(
        self, mask: np.ndarray, frame_shape: tuple[int, int]
    ) -> list[RingDetection]:
        """Find a flattened/open oval from its complete cardboard material outline."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame_shape
        image_area = height * width
        minimum_area = image_area * 0.003
        maximum_area = image_area * 0.40
        clearance = max(8.0, min(height, width) * 0.02)
        roi_x1, roi_x2 = width * self.roi_left, width * self.roi_right
        roi_y1, roi_y2 = height * self.roi_top, height * self.roi_bottom
        candidates: list[RingDetection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if len(contour) < 5 or not minimum_area <= area <= maximum_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area / max(hull_area, 1.0)
            # A moving belt joint can enter the foreground mask as a large,
            # concave L-shaped region.  It may have an acceptable fitted ellipse
            # area, but it has neither the closedness nor convexity of cardboard.
            if circularity < 0.22 or solidity < 0.72:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if (
                x <= roi_x1 + clearance
                or y <= roi_y1 + clearance
                or x + w >= roi_x2 - clearance
                or y + h >= roi_y2 - clearance
            ):
                continue
            ellipse = cv2.fitEllipse(contour)
            major, minor = sorted(ellipse[1], reverse=True)
            if minor < 18 or minor / major < 0.16:
                continue
            ellipse_area = np.pi * major * minor / 4
            if ellipse_area == 0 or not 0.55 <= area / ellipse_area <= 1.28:
                continue
            candidate = RingDetection(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                contour,
                contour,
                ellipse,
                fallback=True,
                source="material",
                confidence=0.62,
            )
            if not any(RingDetector._same_ring(candidate, existing) for existing in candidates):
                candidates.append(candidate)
        return candidates

    def _update_tracks(
        self,
        detections: list[RingDetection],
        gate_y: int,
        diagonal: float,
        frame_height: int,
    ) -> dict[int, RingDetection]:
        self._tracking_frame_index += 1
        matches: list[tuple[float, int, int]] = []
        maximum_distance = max(42.0, diagonal * 0.085)
        crossing_margin = max(10, round(diagonal * 0.0125))
        duplicate_exit_window = 2
        duplicate_exit_distance = min(32.0, max(24.0, diagonal * 0.045))
        self.exit_recent_events = [
            event
            for event in self.exit_recent_events
            if self._tracking_frame_index - event[0] <= duplicate_exit_window
        ]
        roi_top_y = frame_height * self.roi_top
        roi_bottom_y = frame_height * self.roi_bottom
        fov_height = max(1.0, roi_bottom_y - roi_top_y)
        # The center must be close to the real top/bottom boundary, but allow
        # the final accepted ellipse center to vanish roughly one small radius
        # before the geometric edge because ROI clipping rejects later frames.
        exit_confirmation_band = max(32.0, fov_height * 0.070)
        minimum_exit_travel = max(60.0, fov_height * 0.14)
        upward_exit_y = roi_top_y + exit_confirmation_band
        downward_exit_y = roi_bottom_y - exit_confirmation_band

        def is_in_exit_zone(center_y: int) -> bool:
            if self.motion_direction < 0:
                return center_y <= upward_exit_y
            if self.motion_direction > 0:
                return center_y >= downward_exit_y
            return False

        for detection_index, detection in enumerate(detections):
            for track_id, (old_x, old_y, _, missing, confirmed, velocity_x, velocity_y) in self.tracks.items():
                delta_y = detection.center[1] - old_y
                # Do not transfer a mature upward-moving identity backward onto
                # the next ring in a touching train when one contour disappears.
                if confirmed >= 3 and self.motion_direction < 0 and delta_y > max(10, abs(velocity_y) * 2 + 5):
                    continue
                steps = missing + 1
                predicted_x = old_x + velocity_x * steps
                predicted_y = old_y + velocity_y * steps
                matches.append((
                    float(np.hypot(detection.center[0] - predicted_x, detection.center[1] - predicted_y)),
                    detection_index,
                    track_id,
                ))

        updated: dict[int, tuple[int, int, bool, int, int, float, float]] = {}
        indexed: dict[int, RingDetection] = {}
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for distance, detection_index, track_id in sorted(matches):
            if distance > maximum_distance or detection_index in used_detections or track_id in used_tracks:
                continue
            detection = detections[detection_index]
            old_x, old_y, counted, missing_frames, confirmed_frames, velocity_x, velocity_y = self.tracks[track_id]
            required_confirmations = self._required_confirmations(detection)
            steps = missing_frames + 1
            measured_vx = (detection.center[0] - old_x) / steps
            measured_vy = (detection.center[1] - old_y) / steps
            new_vx = velocity_x * 0.65 + measured_vx * 0.35
            new_vy = velocity_y * 0.65 + measured_vy * 0.35
            new_confirmation_count = confirmed_frames + 1

            if self.count_mode == "exit":
                # A track is a production entry only while it is first proven
                # on the approach side of the counting line. A touching group
                # can temporarily hide a counted ring and recreate its identity
                # above the line; treating every such mid/exit-field identity as
                # a new entry produced a second count when it left the top.
                on_entry_side = (
                    self.motion_direction < 0
                    and detection.center[1]
                    >= gate_y + crossing_margin
                ) or (
                    self.motion_direction > 0
                    and detection.center[1]
                    <= gate_y - crossing_margin
                )
                if self.exit_system_armed and on_entry_side:
                    self.exit_entry_tracks.add(track_id)
                    self.exit_track_start_x.setdefault(
                        track_id, detection.center[0]
                    )
                    self.exit_track_start_y.setdefault(
                        track_id, detection.center[1]
                    )
                if new_confirmation_count >= required_confirmations:
                    self.exit_confirmed_tracks.add(track_id)

                # Count a proven conveyor entry as soon as it completes a
                # directed crossing of the red line.  The full-FOV exit check
                # below remains as a fallback for a temporarily hidden line
                # crossing.  Both paths share the track's ``counted`` flag, so
                # the same physical ring cannot increment twice.
                if (
                    self.exit_system_armed
                    and new_confirmation_count >= required_confirmations
                    and (
                        (
                            self.motion_direction < 0
                            and detection.center[1]
                            >= gate_y + crossing_margin
                        )
                        or (
                            self.motion_direction > 0
                            and detection.center[1]
                            <= gate_y - crossing_margin
                        )
                    )
                ):
                    self.track_armed.add(track_id)

                if self.motion_direction < 0:
                    completed_line_crossing = (
                        detection.center[1] <= gate_y - crossing_margin
                        and new_vy < -0.20
                    )
                elif self.motion_direction > 0:
                    completed_line_crossing = (
                        detection.center[1] >= gate_y + crossing_margin
                        and new_vy > 0.20
                    )
                else:
                    completed_line_crossing = False

                if (
                    self.exit_system_armed
                    and not counted
                    and track_id in self.track_armed
                    and new_confirmation_count >= required_confirmations
                    and completed_line_crossing
                ):
                    counted = True
                    self.track_armed.discard(track_id)
                    start_x = self.exit_track_start_x.get(
                        track_id, detection.center[0]
                    )
                    duplicate_line_crossing = any(
                        abs(start_x - previous_start_x)
                        <= duplicate_exit_distance
                        and abs(detection.center[0] - previous_crossing_x)
                        <= duplicate_exit_distance
                        for (
                            _,
                            previous_start_x,
                            previous_crossing_x,
                        ) in self.exit_recent_events
                    )
                    if not duplicate_line_crossing:
                        self.track_crossing_count += 1
                        self.exit_recent_events.append(
                            (
                                self._tracking_frame_index,
                                start_x,
                                detection.center[0],
                            )
                        )
            else:
                # Hysteresis prevents a stationary or shape-changing proposal
                # from repeatedly jittering across one pixel of the red line.
                if new_confirmation_count >= required_confirmations:
                    if (
                        self.motion_direction < 0
                        and detection.center[1] >= gate_y + crossing_margin
                    ) or (
                        self.motion_direction > 0
                        and detection.center[1] <= gate_y - crossing_margin
                    ):
                        self.track_armed.add(track_id)

                if self.motion_direction < 0:
                    completed_crossing = (
                        detection.center[1] <= gate_y - crossing_margin
                        and new_vy < -0.20
                        and self.direction_motion_frames >= 3
                    )
                elif self.motion_direction > 0:
                    completed_crossing = (
                        detection.center[1] >= gate_y + crossing_margin
                        and new_vy > 0.20
                        and self.direction_motion_frames >= 3
                    )
                else:
                    completed_crossing = False
                if (
                    not counted
                    and track_id in self.track_armed
                    and new_confirmation_count >= required_confirmations
                    and completed_crossing
                ):
                    counted = True
                    self.track_armed.discard(track_id)
                    self.track_crossing_count += 1
            updated[track_id] = (*detection.center, counted, 0, confirmed_frames + 1, new_vx, new_vy)
            indexed[track_id] = detection
            used_detections.add(detection_index)
            used_tracks.add(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index not in used_detections:
                new_track_id = self.next_id
                updated[new_track_id] = (*detection.center, False, 0, 1, 0.0, 0.0)
                indexed[new_track_id] = detection
                if (
                    self.count_mode == "exit"
                    and self.exit_system_armed
                    and (
                        (
                            self.motion_direction < 0
                            and detection.center[1]
                            >= gate_y + crossing_margin
                        )
                        or (
                            self.motion_direction > 0
                            and detection.center[1]
                            <= gate_y - crossing_margin
                        )
                    )
                ):
                    self.exit_entry_tracks.add(new_track_id)
                    self.exit_track_start_x[new_track_id] = detection.center[0]
                    self.exit_track_start_y[new_track_id] = detection.center[1]
                    if (
                        self.motion_direction < 0
                        and detection.center[1]
                        >= gate_y + crossing_margin
                    ) or (
                        self.motion_direction > 0
                        and detection.center[1]
                        <= gate_y - crossing_margin
                    ):
                        self.track_armed.add(new_track_id)
                self.next_id += 1

        for track_id, track in self.tracks.items():
            if track_id not in used_tracks:
                missed = track[3] + 1
                if missed <= self.max_missing_frames:
                    old_x, old_y, counted, _, confirmed, velocity_x, velocity_y = (
                        track
                    )
                    moving_with_belt = (
                        (
                            self.motion_direction < 0
                            and velocity_y <= -0.35
                        )
                        or (
                            self.motion_direction > 0
                            and velocity_y >= 0.35
                        )
                    )
                    if self.count_mode == "exit":
                        # A confirmed object counts only after it entered through
                        # the correct FOV edge, travelled in the conveyor
                        # direction, reached the opposite edge, and vanished for
                        # two frames. The red guide line is deliberately absent
                        # from this decision.
                        occlusion_is_trusted = (
                            track_id in self.exit_confirmed_tracks
                            and moving_with_belt
                        )
                        start_y = self.exit_track_start_y.get(track_id)
                        travelled_toward_exit = (
                            start_y is not None
                            and (
                                (
                                    self.motion_direction < 0
                                    and start_y - old_y >= minimum_exit_travel
                                )
                                or (
                                    self.motion_direction > 0
                                    and old_y - start_y >= minimum_exit_travel
                                )
                            )
                        )
                        completed_exit = (
                            missed >= 2
                            and track_id in self.exit_entry_tracks
                            and track_id in self.exit_confirmed_tracks
                            and moving_with_belt
                            and travelled_toward_exit
                            and is_in_exit_zone(old_y)
                            and self.exit_motion_grace_frames > 0
                        )
                        if not counted and completed_exit:
                            counted = True
                            start_x = self.exit_track_start_x.get(
                                track_id, old_x
                            )
                            duplicate_exit = any(
                                abs(start_x - previous_start_x)
                                <= duplicate_exit_distance
                                and abs(old_x - previous_exit_x)
                                <= duplicate_exit_distance
                                for (
                                    _,
                                    previous_start_x,
                                    previous_exit_x,
                                ) in self.exit_recent_events
                            )
                            if not duplicate_exit:
                                self.track_crossing_count += 1
                                self.exit_recent_events.append(
                                    (
                                        self._tracking_frame_index,
                                        start_x,
                                        old_x,
                                    )
                                )
                    else:
                        # A close group can temporarily hide a previously proven
                        # ring exactly at the counting line. Preserve that armed
                        # identity for a short, bounded occlusion.
                        occlusion_is_trusted = (
                            track_id in self.track_armed
                            and confirmed >= 4
                            and missed <= 8
                            and moving_with_belt
                            and self.direction_motion_frames >= 3
                        )
                        if occlusion_is_trusted and not counted:
                            predicted_y = old_y + velocity_y * missed
                            completed_hidden_crossing = (
                                self.motion_direction < 0
                                and predicted_y <= gate_y - crossing_margin
                            ) or (
                                self.motion_direction > 0
                                and predicted_y >= gate_y + crossing_margin
                            )
                            if completed_hidden_crossing:
                                counted = True
                                self.track_armed.discard(track_id)
                                self.track_crossing_count += 1
                    # Confirmation means consecutive visual evidence.  A later
                    # seam/edge proposal must not inherit maturity from a ring
                    # that disappeared several frames earlier. The sole
                    # exception is the bounded, armed occlusion above.
                    updated[track_id] = (
                        old_x,
                        old_y,
                        counted,
                        missed,
                        confirmed if occlusion_is_trusted else 0,
                        velocity_x,
                        velocity_y,
                    )

        self.tracks = updated
        self.track_armed.intersection_update(updated)
        self.exit_entry_tracks.intersection_update(updated)
        self.exit_confirmed_tracks.intersection_update(updated)
        self.exit_track_start_x = {
            track_id: start_x
            for track_id, start_x in self.exit_track_start_x.items()
            if track_id in updated
        }
        self.exit_track_start_y = {
            track_id: start_y
            for track_id, start_y in self.exit_track_start_y.items()
            if track_id in updated
        }
        return indexed

    def _update_scene_motion(self, frame: np.ndarray) -> None:
        """Estimate sustained belt motion independently of contour identities."""
        height, width = frame.shape[:2]
        x1 = max(0, int(width * self.roi_left))
        x2 = min(width, int(width * self.roi_right))
        y1 = max(0, int(height * self.roi_top))
        y2 = min(height, int(height * self.roi_bottom))
        if x2 - x1 < 40 or y2 - y1 < 40:
            self.direction_motion_frames = 0
            self.scene_velocity_y = 0.0
            self.exit_motion_grace_frames = 0
            return
        gray = (
            self._frame_gray[y1:y2, x1:x2]
            if self._frame_gray is not None
            and self._frame_gray.shape == frame.shape[:2]
            else cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        )
        reduced_width = max(40, (x2 - x1) // 2)
        reduced_height = max(40, (y2 - y1) // 2)
        reduced = cv2.resize(
            gray,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        self._motion_frames.append(reduced)
        if len(self._motion_frames) > 5:
            self._motion_frames.pop(0)
        if len(self._motion_frames) < 5:
            self.scene_velocity_y = 0.0
            self.direction_motion_frames = 0
            self.exit_motion_grace_frames = max(
                0, self.exit_motion_grace_frames - 1
            )
            return

        shift, response = cv2.phaseCorrelate(
            self._motion_frames[0], self._motion_frames[-1]
        )
        scale_y = (y2 - y1) / float(reduced_height)
        per_frame_y = float(shift[1]) * scale_y / 4.0
        if response < 0.12 or abs(per_frame_y) > 12.0:
            per_frame_y = 0.0
        self.scene_velocity_y = self.scene_velocity_y * 0.55 + per_frame_y * 0.45

        correct_direction = (
            self.motion_direction < 0 and self.scene_velocity_y < -0.12
        ) or (
            self.motion_direction > 0 and self.scene_velocity_y > 0.12
        )
        if correct_direction:
            self.direction_motion_frames = min(
                self.direction_motion_frames + 1, 30
            )
            self.exit_motion_grace_frames = 5
        else:
            self.direction_motion_frames = 0
            self.exit_motion_grace_frames = max(
                0, self.exit_motion_grace_frames - 1
            )

    def _stable_display_detections(
        self, detections: list[RingDetection]
    ) -> list[RingDetection]:
        """Return only objects continuously supported across recent frames."""
        self._display_history.append(list(detections))
        if len(self._display_history) > 7:
            self._display_history.pop(0)

        clusters: list[dict[str, object]] = []
        for frame_index, frame_detections in enumerate(self._display_history):
            for detection in frame_detections:
                best_cluster: dict[str, object] | None = None
                best_distance = float("inf")
                for cluster in clusters:
                    representative = cluster["representative"]
                    assert isinstance(representative, RingDetection)
                    distance = float(
                        np.hypot(
                            detection.center[0] - representative.center[0],
                            detection.center[1] - representative.center[1],
                        )
                    )
                    proximity = min(
                        34.0,
                        max(
                            20.0,
                            max(
                                max(detection.ellipse[1]),
                                max(representative.ellipse[1]),
                            )
                            * 0.42,
                        ),
                    )
                    if distance <= proximity and distance < best_distance:
                        best_cluster = cluster
                        best_distance = distance
                if best_cluster is None:
                    clusters.append({
                        "representative": detection,
                        "frames": {frame_index},
                        "latest_frame": frame_index,
                    })
                    continue
                frames = best_cluster["frames"]
                assert isinstance(frames, set)
                frames.add(frame_index)
                latest_frame = int(best_cluster["latest_frame"])
                representative = best_cluster["representative"]
                assert isinstance(representative, RingDetection)
                if (
                    frame_index > latest_frame
                    or (
                        frame_index == latest_frame
                        and detection.confidence > representative.confidence
                    )
                ):
                    best_cluster["representative"] = detection
                    best_cluster["latest_frame"] = frame_index

        # A heavily occluded ring can switch between a hole centre, wall centre,
        # and residual-material centre. Those proposal clusters often sit 35–60
        # pixels apart but occur in different frames. Merge only temporally
        # disjoint nearby clusters; two real neighboring rings coexist in the
        # same frames and therefore remain separate.
        merged = True
        while merged:
            merged = False
            for first_index, first in enumerate(clusters):
                first_detection = first["representative"]
                first_frames = first["frames"]
                assert isinstance(first_detection, RingDetection)
                assert isinstance(first_frames, set)
                for second_index in range(first_index + 1, len(clusters)):
                    second = clusters[second_index]
                    second_detection = second["representative"]
                    second_frames = second["frames"]
                    assert isinstance(second_detection, RingDetection)
                    assert isinstance(second_frames, set)
                    if not first_frames.isdisjoint(second_frames):
                        continue
                    distance = float(
                        np.hypot(
                            first_detection.center[0]
                            - second_detection.center[0],
                            first_detection.center[1]
                            - second_detection.center[1],
                        )
                    )
                    merge_distance = min(
                        60.0,
                        max(
                            36.0,
                            max(
                                max(first_detection.ellipse[1]),
                                max(second_detection.ellipse[1]),
                            )
                            * 0.68,
                        ),
                    )
                    if distance > merge_distance:
                        continue
                    first_frames.update(second_frames)
                    if int(second["latest_frame"]) >= int(first["latest_frame"]):
                        first["representative"] = second_detection
                        first["latest_frame"] = second["latest_frame"]
                    clusters.pop(second_index)
                    merged = True
                    break
                if merged:
                    break

        stable: list[RingDetection] = []
        newest_frame_index = len(self._display_history) - 1
        for cluster in clusters:
            frames = cluster["frames"]
            representative = cluster["representative"]
            assert isinstance(frames, set)
            assert isinstance(representative, RingDetection)
            # Never draw a historical proposal that is absent from the current
            # camera frame. In dense piles, one physical ring can alternate
            # between hole, fallback and flat evidence; keeping each old cluster
            # for two more frames produced convincing fake green outlines.
            if newest_frame_index not in frames:
                continue
            strong_current_ring = (
                representative.source in {"primary", "overlap"}
                and representative.confidence >= 0.85
            )
            # A green guide is an operator-facing accepted result, so even a
            # high-confidence hole must remain present for four consecutive
            # frames. Weak recovery proposals need five. This intentionally
            # rejects the short flashes caused by highlights and touching rims.
            required_consecutive = 4 if strong_current_ring else 5
            stable_enough = newest_frame_index >= required_consecutive - 1 and all(
                newest_frame_index - offset in frames
                for offset in range(required_consecutive)
            )
            if stable_enough:
                stable.append(representative)
        return stable

    @staticmethod
    def _required_confirmations(detection: RingDetection) -> int:
        """Frames required before a proposal is visible or countable."""
        if detection.source in {"primary", "overlap"} and detection.confidence >= 0.85:
            return 2
        if detection.source == "flat":
            return 3
        return 3

    @staticmethod
    def _display_confirmations(
        detection: RingDetection, belt_moving: bool
    ) -> int:
        """Keep static filtering strict without losing fast conveyor objects."""
        strong = (
            detection.source in {"primary", "overlap"}
            and detection.confidence >= 0.85
        )
        if belt_moving:
            return 2 if strong else 3
        return 4 if strong else 5

    def _update_gate_counter(self, detections: list[RingDetection], gate_y: int, height: int) -> None:
        """Track counting-band occupancy for status without creating a count."""
        half_band = max(9, round(height * 0.022))
        occupied = any(abs(detection.center[1] - gate_y) <= half_band for detection in detections)
        if occupied:
            self.gate_occupied = True
            self.gate_clear_frames = 0
        else:
            self.gate_clear_frames += 1
            if self.gate_clear_frames >= self.gate_rearm_frames:
                self.gate_occupied = False

    def _stabilize_visible_count(
        self, current_visible: int, has_raw_detections: bool
    ) -> int:
        """Publish only a persistent consensus count, but clear immediately when empty."""
        if not has_raw_detections:
            self._visible_count_history.clear()
            self._visible_lower_frames = 0
            self._visible_change_candidate = None
            self._visible_change_frames = 0
            self.last_visible = 0
            return 0

        self._visible_count_history.append(current_visible)
        if len(self._visible_count_history) > 24:
            self._visible_count_history.pop(0)

        # Give the dashboard an initial value promptly. Every later change must
        # survive the rolling consensus and a persistence timer, preventing one
        # weak contour from making a dense group flicker between adjacent counts.
        if self.last_visible == 0 and current_visible > 0:
            self.last_visible = current_visible
            self._visible_change_candidate = None
            self._visible_change_frames = 0
            return self.last_visible

        consensus = max(
            set(self._visible_count_history),
            key=lambda value: (self._visible_count_history.count(value), value),
        )
        if consensus == self.last_visible:
            self._visible_change_candidate = None
            self._visible_change_frames = 0
            return self.last_visible

        if consensus == self._visible_change_candidate:
            self._visible_change_frames += 1
        else:
            self._visible_change_candidate = consensus
            self._visible_change_frames = 1

        required_frames = 12 if consensus > self.last_visible else 24
        if self._visible_change_frames >= required_frames:
            self.last_visible = consensus
            self._visible_count_history = [consensus]
            self._visible_change_candidate = None
            self._visible_change_frames = 0
        return self.last_visible

    @staticmethod
    def _canonical_overlay_ellipse(
        detection: RingDetection,
    ) -> tuple[float, float, float, float, float]:
        """Return a consistent outer ellipse even when OpenCV swaps its axes."""
        contour = detection.outer_contour
        if detection.source in {"primary", "overlap"}:
            # For normal rings, the enclosed dark hole is much steadier than the
            # beige outside edge when rings touch. Convert that stable hole fit
            # to an outer guide with one fixed wall allowance.
            (center_x, center_y), (width, height), angle = detection.ellipse
            width *= 1.28
            height *= 1.28
        elif contour is not None and len(contour) >= 5:
            (center_x, center_y), (width, height), angle = cv2.fitEllipse(contour)
        else:
            (center_x, center_y), (width, height), angle = detection.ellipse
            width *= 1.16
            height *= 1.16

        if width > height:
            width, height = height, width
            angle += 90.0
        return (
            float(center_x),
            float(center_y),
            max(2.0, float(width)),
            max(2.0, float(height)),
            float(angle % 180.0),
        )

    def _smoothed_overlay_ellipse(
        self, overlay_id: int, detection: RingDetection
    ) -> tuple[tuple[float, float], tuple[float, float], float]:
        """Smooth only the green guide while allowing real conveyor movement."""
        center_x, center_y, width, height, angle = (
            self._canonical_overlay_ellipse(detection)
        )
        previous = self._overlay_geometry.get(overlay_id)
        if previous is not None:
            (
                previous_x,
                previous_y,
                previous_width,
                previous_height,
                previous_angle,
                previous_frame,
            ) = previous
            if self._tracking_frame_index - previous_frame <= 2:
                delta_x = center_x - previous_x
                delta_y = center_y - previous_y
                distance = float(np.hypot(delta_x, delta_y))
                # A track association is occasionally pulled toward a neighbour
                # in a touching pile. Limit that visual jump without changing the
                # underlying detection or identity used for counting.
                maximum_step = max(12.0, previous_height * 0.20)
                if distance > maximum_step:
                    scale = maximum_step / distance
                    center_x = previous_x + delta_x * scale
                    center_y = previous_y + delta_y * scale

                center_alpha = 0.38
                size_alpha = 0.08
                if abs(center_x - previous_x) < 0.70:
                    center_x = previous_x
                else:
                    center_x = (
                        previous_x * (1.0 - center_alpha)
                        + center_x * center_alpha
                    )
                if abs(center_y - previous_y) < 0.70:
                    center_y = previous_y
                else:
                    center_y = (
                        previous_y * (1.0 - center_alpha)
                        + center_y * center_alpha
                    )
                if detection.source not in {"primary", "overlap"}:
                    # Recovery proposals can alternate between a partial arc,
                    # material box and Hough circle for the same physical ring.
                    # Preserve its proven display shape instead of breathing.
                    width = previous_width
                    height = previous_height
                else:
                    width = (
                        previous_width
                        if abs(width - previous_width) < 1.0
                        else previous_width * (1.0 - size_alpha)
                        + width * size_alpha
                    )
                    height = (
                        previous_height
                        if abs(height - previous_height) < 1.0
                        else previous_height * (1.0 - size_alpha)
                        + height * size_alpha
                    )

                # Rotation is visually meaningless for a near-circle and is the
                # largest source of apparent shaking when fitEllipse changes by
                # 90 degrees between frames.
                if detection.source not in {"primary", "overlap"}:
                    angle = previous_angle
                elif width / max(height, 1.0) >= 0.86:
                    angle = previous_angle
                else:
                    angle_delta = ((angle - previous_angle + 90.0) % 180.0) - 90.0
                    angle = (previous_angle + angle_delta * 0.16) % 180.0
        elif width / max(height, 1.0) >= 0.86:
            angle = 0.0

        self._overlay_geometry[overlay_id] = (
            center_x,
            center_y,
            width,
            height,
            angle,
            self._tracking_frame_index,
        )
        return (center_x, center_y), (width, height), angle

    def _assign_overlay_ids(
        self, detections: list[RingDetection]
    ) -> list[tuple[int, RingDetection]]:
        """Keep one visual identity even if the counting tracker changes IDs."""
        recent_geometry = {
            overlay_id: geometry
            for overlay_id, geometry in self._overlay_geometry.items()
            if self._tracking_frame_index - geometry[5] <= 4
        }
        candidate_matches: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            center_x, center_y = detection.center
            for overlay_id, geometry in recent_geometry.items():
                distance = float(
                    np.hypot(center_x - geometry[0], center_y - geometry[1])
                )
                maximum_distance = min(58.0, max(30.0, geometry[3] * 0.52))
                if distance <= maximum_distance:
                    candidate_matches.append(
                        (distance, detection_index, overlay_id)
                    )

        assigned_detection_ids: dict[int, int] = {}
        used_overlay_ids: set[int] = set()
        for _, detection_index, overlay_id in sorted(candidate_matches):
            if (
                detection_index in assigned_detection_ids
                or overlay_id in used_overlay_ids
            ):
                continue
            assigned_detection_ids[detection_index] = overlay_id
            used_overlay_ids.add(overlay_id)

        assignments: list[tuple[int, RingDetection]] = []
        for detection_index, detection in enumerate(detections):
            overlay_id = assigned_detection_ids.get(detection_index)
            if overlay_id is None:
                overlay_id = self._next_overlay_id
                self._next_overlay_id += 1
            assignments.append((overlay_id, detection))
        return assignments

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
        started_at = perf_counter()
        self._update_scene_motion(frame)
        detections, _ = self.detect(frame)
        if self.count_mode == "exit" and not self.exit_system_armed:
            if detections:
                self.exit_clear_frames = 0
            else:
                self.exit_clear_frames += 1
                if self.exit_clear_frames >= self.exit_arm_clear_frames:
                    self.exit_system_armed = True
        display = frame.copy()
        height, width = display.shape[:2]
        gate_y = int(height * self.gate_ratio)
        indexed = self._update_tracks(
            detections,
            gate_y,
            float(np.hypot(width, height)),
            height,
        )
        confirmed_indexed = {
            track_id: detection
            for track_id, detection in indexed.items()
            if self.tracks[track_id][4] >= self._required_confirmations(detection)
        }
        confirmed_detections = list(confirmed_indexed.values())
        belt_moving = (
            self.direction_motion_frames >= 2
            or (
                self.motion_direction < 0
                and self.scene_velocity_y <= -0.20
            )
            or (
                self.motion_direction > 0
                and self.scene_velocity_y >= 0.20
            )
        )
        # Use the velocity-aware one-to-one tracker for the final view. The old
        # history clustering was excellent while static, but at low processing
        # FPS a moving ring travelled beyond its cluster before collecting four
        # frames, reducing an 11-ring batch to only four visible guides.
        stable_display = [
            detection
            for track_id, detection in indexed.items()
            if self.tracks[track_id][4]
            >= self._display_confirmations(detection, belt_moving)
        ]
        display_overlays = self._assign_overlay_ids(stable_display)

        self._stabilize_visible_count(
            len(stable_display), bool(detections)
        )
        self._update_gate_counter(confirmed_detections, gate_y, height)
        # Only a confirmed track moving in the configured conveyor direction may
        # increase the count. Initial/static line occupancy is never a crossing.
        self.total_count = max(self.total_count, self.track_crossing_count)

        active_overlay_ids = {overlay_id for overlay_id, _ in display_overlays}
        held_overlay_guides = [
            ((geometry[0], geometry[1]), (geometry[2], geometry[3]), geometry[4])
            for overlay_id, geometry in self._overlay_geometry.items()
            if detections
            and not belt_moving
            and overlay_id not in active_overlay_ids
            and self._tracking_frame_index - geometry[5] <= 2
        ]
        self._overlay_geometry = {
            overlay_id: geometry
            for overlay_id, geometry in self._overlay_geometry.items()
            if overlay_id in active_overlay_ids
            or self._tracking_frame_index - geometry[5] <= 2
        }
        for overlay_id, detection in display_overlays:
            guide = self._smoothed_overlay_ellipse(overlay_id, detection)
            cv2.ellipse(display, guide, (0, 255, 0), 3, cv2.LINE_AA)
        for guide in held_overlay_guides:
            cv2.ellipse(display, guide, (0, 255, 0), 3, cv2.LINE_AA)

        x1, x2 = int(width * self.roi_left), int(width * self.roi_right)
        y1, y2 = int(height * self.roi_top), int(height * self.roi_bottom)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 2, cv2.LINE_AA)
        cv2.line(display, (x1, gate_y), (x2, gate_y), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "COUNTING LINE", (x1 + 12, gate_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(display, (10, 10), (285, 114), (7, 17, 31), -1)
        cv2.putText(display, f"VISIBLE RINGS: {self.last_visible}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (162, 255, 240), 2, cv2.LINE_AA)
        cv2.putText(display, f"TOTAL COUNT: {self.total_count}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        elapsed = perf_counter() - started_at
        instantaneous_processing_fps = 1 / max(elapsed, 0.0001)
        self.processing_fps = instantaneous_processing_fps if self.processing_fps == 0 else self.processing_fps * 0.85 + instantaneous_processing_fps * 0.15
        cv2.putText(display, f"PROC FPS: {self.processing_fps:.1f}", (20, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (162, 255, 240), 2, cv2.LINE_AA)
        return display, self.status()
