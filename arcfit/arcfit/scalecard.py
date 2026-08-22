"""Finding the checkerboard scale card automatically.

Every photograph needs its own homography, because the camera moved between
shots. Clicking four corners on 167 field photographs is both slow and a source
of operator error, so the card is located automatically and the operator only
confirms or overrides.

Detection deliberately does *not* use OpenCV's chessboard finder. That routine
wants a grid of interior corners at least 3x3, and a two-row scale bar presents
only a single interior row. Instead the card is found as a high-contrast
quadrilateral and then *verified* by rectifying it and checking that the patch
really does contain the expected alternating squares. Verification is what makes
this safe: a bright rock or a patch of shadow can produce a plausible quad, but
it will not produce a checkerboard.

Several thresholding strategies are run and their candidates pooled. Field
lighting varies enormously across these photographs -- direct sun, open shade,
the card's own cast shadow -- and no single threshold handles all of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .snap import refine_polyline_edge

__all__ = ["CardSpec", "CardDetection", "detect_card", "order_card_corners",
           "CONFIDENCE_THRESHOLD", "HAVE_CV2"]

# Below this, a detection is reported as needing a look rather than trusted.
# Deliberately conservative: confirming a good detection costs one keystroke in
# the digitiser, whereas a silently wrong homography mis-scales every
# measurement for that object. Raise it if too many photographs are being
# flagged, but check a sample of the flagged ones first.
CONFIDENCE_THRESHOLD = 0.55

try:
    import cv2
    HAVE_CV2 = True
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None
    HAVE_CV2 = False


@dataclass(frozen=True)
class CardSpec:
    """Physical description of the scale card.

    Defaults describe a 10 cm bar of 1 cm squares in two rows. These are
    configuration, not constants: if the real card differs, set it once and every
    measurement rescales without any re-digitising.
    """

    width_cm: float = 10.0
    height_cm: float = 2.0
    cols: int = 10
    rows: int = 2
    row_spec: Tuple[Tuple[float, int], ...] = ()

    @property
    def aspect(self) -> float:
        return self.width_cm / self.height_cm

    @property
    def row_layout(self) -> Tuple[Tuple[float, int], ...]:
        """Rows as ``(height_cm, cells_across)``, along the short side.

        ``rows`` and ``cols`` describe a uniform grid, which is all most
        chequered bars need. Archaeological scales frequently are not uniform --
        the card used here runs two rows of 1 cm squares against one row of 2 cm
        squares -- and a single (rows, cols) pair cannot express that. Setting
        ``row_spec`` overrides them; leaving it empty keeps the uniform grid.
        """
        if self.row_spec:
            return tuple((float(h), int(n)) for h, n in self.row_spec)
        return tuple((self.height_cm / self.rows, self.cols)
                     for _ in range(self.rows))

    def validate(self) -> None:
        if self.width_cm <= 0 or self.height_cm <= 0:
            raise ValueError("card dimensions must be positive")
        if self.cols < 2 or self.rows < 1:
            raise ValueError("card must have at least 2 columns and 1 row")
        if self.width_cm < self.height_cm:
            raise ValueError("width_cm must be the long side of the card")
        if self.row_spec:
            if any(float(h) <= 0 or int(n) < 2 for h, n in self.row_spec):
                raise ValueError(
                    "each card row needs a positive height and at least 2 cells")
            total = sum(float(h) for h, _ in self.row_spec)
            if abs(total - self.height_cm) > 1e-6:
                raise ValueError(
                    f"card row heights sum to {total:g} cm, but height_cm is "
                    f"{self.height_cm:g} cm")


@dataclass
class CardDetection:
    """A located card. ``corners_px`` are ordered long-edge-first."""

    corners_px: np.ndarray
    score: float
    method: str
    aspect_error: float = 0.0
    subpixel: bool = False
    edge_rms_px: Optional[float] = None
    interior_px: Optional[np.ndarray] = None
    interior_cm: Optional[np.ndarray] = None
    notes: str = ""

    @property
    def is_confident(self) -> bool:
        return self.score >= CONFIDENCE_THRESHOLD


def order_card_corners(quad: np.ndarray, spec: CardSpec) -> np.ndarray:
    """Order four corners so edge 0->1 is the card's long side.

    Getting this wrong swaps the card's width and height and silently rescales
    every measurement by the card's aspect ratio, so the long side is identified
    from opposite-edge averages rather than from any single edge, which
    perspective can foreshorten badly.
    """
    quad = np.asarray(quad, float).reshape(4, 2)

    # Deterministic winding, then a deterministic starting corner. Both matter:
    # inconsistent winding would flip the rectified coordinate frame between
    # photographs, which negates reported orientations for no reason.
    centre = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    quad = quad[np.argsort(ang)]
    quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)

    def edge(i: int) -> float:
        return float(np.hypot(*(quad[(i + 1) % 4] - quad[i])))

    pair_a = 0.5 * (edge(0) + edge(2))
    pair_b = 0.5 * (edge(1) + edge(3))
    if pair_b > pair_a:
        quad = np.roll(quad, -1, axis=0)
    return quad


def _rectify_patch(gray: np.ndarray, quad: np.ndarray, spec: CardSpec,
                   px_per_cm: int = 12) -> Optional[np.ndarray]:
    # Cap the output resolution at what the card actually occupies. An oblique
    # card is heavily foreshortened, and upsampling its short side to a nominal
    # resolution just interpolates blur across the squares, washing out the very
    # pattern being checked for.
    short_px = min(np.hypot(*(quad[2] - quad[1])), np.hypot(*(quad[0] - quad[3])))
    long_px = min(np.hypot(*(quad[1] - quad[0])), np.hypot(*(quad[3] - quad[2])))
    avail = min(short_px / max(spec.height_cm, 1e-9), long_px / max(spec.width_cm, 1e-9))
    px_per_cm = max(3, min(px_per_cm, int(avail)))

    w = int(round(spec.width_cm * px_per_cm))
    h = int(round(spec.height_cm * px_per_cm))
    if w < 8 or h < 4:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(gray, M, (w, h))


def _checkerboard_score(patch: np.ndarray, spec: CardSpec) -> float:
    """How well a rectified patch matches the card's printed pattern.

    Returns 0..1. Each row is reduced to one mean value per expected cell and
    correlated against both phases of an alternating run; the better phase wins,
    since which cell is black is arbitrary.

    Scoring runs per row rather than over the whole grid because rows may differ
    in cell size, so there is no single lattice to correlate against. That also
    drops the requirement that neighbouring rows be offset from one another --
    on this card they are not, and demanding it rejected the genuine article.
    Ten alternating cells in each of several rows remains far beyond what a rock
    or a shadow produces.

    The layout is tried both ways up: nothing constrains which end of the card
    faces the camera, and the corner ordering is chosen from image geometry.
    """
    if patch is None or patch.size == 0:
        return 0.0
    h, w = patch.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    if float(patch.max()) - float(patch.min()) < 18.0:
        return 0.0      # a flat region cannot be a checkerboard

    layout = spec.row_layout
    total_cm = sum(height for height, _ in layout)
    if total_cm <= 0:
        return 0.0

    def score(rows: Tuple[Tuple[float, int], ...]) -> float:
        edges, acc = [0.0], 0.0
        for height, _ in rows:
            acc += height
            edges.append(acc / total_cm)

        weighted, weight = 0.0, 0.0
        for r, (_, n_cells) in enumerate(rows):
            y0, y1 = int(edges[r] * h), int(edges[r + 1] * h)
            if y1 - y0 < 2 or n_cells < 2:
                continue
            xs = np.linspace(0, w, n_cells + 1).astype(int)

            values = np.zeros(n_cells, float)
            for c in range(n_cells):
                # Trim a margin so a slightly misplaced corner does not bleed
                # neighbouring cells into the sample.
                x0, x1 = xs[c], xs[c + 1]
                my, mx = max(1, (y1 - y0) // 4), max(1, (x1 - x0) // 4)
                block = patch[y0 + my:max(y0 + my + 1, y1 - my),
                              x0 + mx:max(x0 + mx + 1, x1 - mx)]
                values[c] = float(block.mean()) if block.size else 0.0

            if values.max() - values.min() < 18.0:
                continue        # a uniform row carries no evidence either way
            norm = (values - values.mean()) / (values.std() + 1e-9)

            # Score on sign agreement rather than correlation magnitude.
            # Foreshortening and blur compress the contrast between cells
            # without reordering them, so a magnitude-based correlation reads an
            # oblique but perfectly good card as a failure -- and every
            # photograph here is oblique. Sign agreement survives that, while
            # still collapsing to chance on anything that is not a checkerboard.
            ideal = np.where(np.arange(n_cells) % 2 == 0, 1.0, -1.0)
            best = 0.0
            for phase in (ideal, -ideal):
                agree = float(np.mean(np.sign(norm) == np.sign(phase)))
                best = max(best, agree)
            weighted += n_cells * float(np.clip(2.0 * (best - 0.5), 0.0, 1.0))
            weight += n_cells

        return weighted / weight if weight else 0.0

    return max(score(layout), score(tuple(reversed(layout))))


def _candidate_quads(gray: np.ndarray, min_area: float, max_area: float) -> List[Tuple[np.ndarray, str]]:
    """Quadrilaterals from several thresholding strategies, pooled."""
    out: List[Tuple[np.ndarray, str]] = []
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    strategies = []
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strategies.append(("otsu", otsu))
    strategies.append(("otsu_inv", 255 - otsu))
    strategies.append(("adaptive", cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)))
    edges = cv2.Canny(blur, 50, 150)
    strategies.append(("canny", cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)))

    for name, binary in strategies:
        # Closing merges the card's black and white squares into one blob, so the
        # contour follows the card outline instead of each individual square.
        k = np.ones((7, 7), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.02, 0.04, 0.06):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    out.append((approx.reshape(4, 2).astype(float), name))
                    break
    return out


def detect_card(image, spec: CardSpec = CardSpec(), work_width: int = 1600,
                min_area_frac: float = 2e-4, max_area_frac: float = 0.25,
                refine: bool = True, min_cell_px: float = 6.0) -> Optional[CardDetection]:
    """Locate the scale card in one photograph.

    ``image`` may be a path or a grayscale/BGR array. Detection runs on a
    downscaled copy for speed -- these are 15 MB images -- and the winning
    corners are then refined at full resolution, so precision is not sacrificed
    to that speed-up.

    ``min_cell_px`` is the smallest printed cell, in working-image pixels, that
    is still worth testing; smaller candidates are discarded unverified rather
    than scored on noise.

    Returns None if nothing verified as a checkerboard. Callers are expected to
    fall back to manual calibration rather than to trust a guess.
    """
    if not HAVE_CV2:
        raise RuntimeError(
            "automatic card detection needs opencv-python "
            "(pip install opencv-python); use manual calibration otherwise"
        )
    spec.validate()

    if isinstance(image, (str, Path)):
        full = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if full is None:
            raise FileNotFoundError(f"could not read image: {image}")
    else:
        arr = np.asarray(image)
        full = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    H, W = full.shape[:2]
    scale = min(1.0, work_width / float(W))
    small = cv2.resize(full, (int(W * scale), int(H * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else full

    area = small.shape[0] * small.shape[1]
    quads = _candidate_quads(small, min_area_frac * area, max_area_frac * area)
    if not quads:
        return None

    best: Optional[CardDetection] = None
    seen: List[np.ndarray] = []
    for quad, method in quads:
        ordered = order_card_corners(quad, spec)

        # Skip near-duplicates from different strategies.
        if any(np.abs(ordered - s).max() < 3.0 for s in seen):
            continue
        seen.append(ordered)

        e0 = 0.5 * (np.hypot(*(ordered[1] - ordered[0])) + np.hypot(*(ordered[3] - ordered[2])))
        e1 = 0.5 * (np.hypot(*(ordered[2] - ordered[1])) + np.hypot(*(ordered[0] - ordered[3])))

        # Refuse candidates too small to verify. The pattern check reduces each
        # cell to a mean, and below a few pixels per cell those means are noise:
        # sign agreement then reaches the threshold by chance often enough that
        # a speck of gravel outscores the card. This is a statement about what
        # can be checked, not about how the photographer framed the shot, so it
        # holds regardless of image size.
        layout = spec.row_layout
        cell_px_x = e0 / max(max(n for _, n in layout), 1)
        cell_px_y = e1 * min(h for h, _ in layout) / max(spec.height_cm, 1e-9)
        if min(cell_px_x, cell_px_y) < min_cell_px:
            continue

        patch = _rectify_patch(small, ordered, spec)
        score = _checkerboard_score(patch, spec)
        if score <= 0.0:
            continue

        aspect = e0 / max(e1, 1e-9)
        aspect_err = abs(np.log(aspect / spec.aspect))
        # Perspective legitimately changes the observed aspect, so this only
        # gently penalises implausible shapes rather than rejecting them.
        combined = score * float(np.exp(-0.5 * aspect_err))

        if best is None or combined > best.score:
            best = CardDetection(corners_px=ordered / scale, score=combined,
                                 method=method, aspect_error=float(aspect_err))

    if best is None:
        return None

    if refine:
        corners, rms = refine_polyline_edge(full, best.corners_px)
        if np.isfinite(rms):
            best.corners_px = order_card_corners(corners, spec)
            best.subpixel = True
            best.edge_rms_px = float(rms)
        else:
            best.notes = "edge refinement rejected; using contour corners"

        pts_px, pts_cm = _interior_corners(full, best.corners_px, spec)
        if pts_px is not None and len(pts_px) >= 2:
            best.interior_px = pts_px
            best.interior_cm = pts_cm

    return best


def _interior_corners(gray: np.ndarray, corners_px: np.ndarray, spec: CardSpec,
                      max_move_frac: float = 0.35):
    """Locate the checkerboard's interior corners to sharpen the homography.

    Four outer corners exactly determine a homography, which means there are no
    residuals left over and no way to tell a good calibration from a bad one.
    The interior corners fix both problems: they add well-localised constraints
    spread along the card's long axis, which is the direction the scale depends
    on most, and they make the reprojection residual an actual quality measure.

    These are true saddle points, which is what ``cornerSubPix`` is designed
    for -- unlike the card's outer corners, where two edges are turning at once.
    Their approximate positions are known in advance from the outer corners, so
    each refinement starts close and only small corrections are accepted.
    """
    # A saddle point needs the colour to change on both sides of the boundary,
    # so an interior corner exists only where a cell edge in the row above lines
    # up with one in the row below. On a uniform grid that is every edge; where a
    # row of 1 cm cells meets a row of 2 cm cells it is only every second one.
    layout = spec.row_layout
    if len(layout) < 2:
        return None, None

    model_pts = []
    y_cm = 0.0
    for above, below in zip(layout, layout[1:]):
        y_cm += above[0]
        n_a, n_b = above[1], below[1]
        step_a, step_b = spec.width_cm / n_a, spec.width_cm / n_b
        for i in range(1, n_a):
            x_cm = i * step_a
            k = x_cm / step_b
            if abs(k - round(k)) < 1e-6 and 0 < round(k) < n_b:
                model_pts.append([x_cm, y_cm])
    if len(model_pts) < 2:
        return None, None
    model = np.array(model_pts, np.float32)

    src = np.array([[0.0, 0.0], [spec.width_cm, 0.0],
                    [spec.width_cm, spec.height_cm], [0.0, spec.height_cm]], np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, corners_px.astype(np.float32))
    except cv2.error:
        return None, None
    guess = cv2.perspectiveTransform(model.reshape(-1, 1, 2), M).reshape(-1, 2)

    # Search window scaled to one square, so it cannot reach a neighbouring corner.
    edge = np.hypot(*(corners_px[1] - corners_px[0]))
    square_px = edge / max(max(n for _, n in layout), 1)
    win = int(max(3, min(round(square_px * 0.35), 25)))
    try:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        refined = cv2.cornerSubPix(gray, guess.astype(np.float32).reshape(-1, 1, 2),
                                   (win, win), (-1, -1), crit).reshape(-1, 2)
    except cv2.error:
        return None, None

    moved = np.hypot(*(refined - guess).T)
    keep = moved < max_move_frac * square_px
    if keep.sum() < 2:
        return None, None
    return refined[keep].astype(float), model[keep].astype(float)
