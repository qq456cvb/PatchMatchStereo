"""
PatchMatch Stereo -- Stereo Matching with Slanted Support Windows
(M. Bleyer, C. Rhemann, C. Rother, BMVC 2011), implemented in Python + Numba.

Every pixel p carries a slanted disparity plane f_p = (a, b, c), i.e. the
disparity of any pixel q under f_p is  d = a * q_x + b * q_y + c.

Pipeline
  1. Random initialisation of one plane per pixel, in both views.
  2. `iters` iterations.  In each one the left and then the right view is swept
     (even iterations: top-left -> bottom-right, odd: bottom-right -> top-left).
     For every pixel p:
       a. spatial propagation -- try the planes of the left / upper (or right /
          lower) neighbour,
       b. view propagation    -- try the planes of the pixels in the other view
          whose current plane maps them onto p (transformed into p's view),
       c. plane refinement    -- random perturbation of disparity and normal
          with exponentially shrinking search ranges.
     A candidate plane is adopted whenever it lowers the aggregated cost m(p, f).
  3. Post-processing: left/right consistency check, filling of invalid pixels
     with the background plane of the nearest valid neighbour on the scanline,
     and a weighted median filter on the filled pixels.

Speed: the sweep is parallelised over anti-diagonals.  Spatial propagation only
looks at the left and upper (right and lower) neighbours, so all pixels on one
anti-diagonal are independent.  Each pixel also has its own deterministic
random stream, so the result is exactly what a sequential raster scan would
produce, and it is the same for every thread count.
"""
import argparse
import os
import time

import cv2
import numpy as np
from numba import njit, prange

INVALID_COST = 1e30

# Disparity range allowed for any plane, in pixels: [0, max_disp].
# The left view matches x -> x - d (sign = +1), the right view x -> x + d (sign = -1).
LEFT, RIGHT = 1, -1


# -----------------------------------------------------------------------------
# Per-pixel random numbers (SplitMix64)
# -----------------------------------------------------------------------------
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S11, _S27, _S30, _S31 = np.uint64(11), np.uint64(27), np.uint64(30), np.uint64(31)
_INV_2_53 = 1.0 / 9007199254740992.0


@njit(cache=True)
def _mix64(z):
    z = (z ^ (z >> _S30)) * _MIX1
    z = (z ^ (z >> _S27)) * _MIX2
    return z ^ (z >> _S31)


@njit(cache=True)
def _rand_sym(state):
    """Advance the RNG state and return (state, uniform sample in [-1, 1))."""
    state = state + _GOLDEN
    u = (_mix64(state) >> _S11) * _INV_2_53
    return state, 2.0 * u - 1.0


# -----------------------------------------------------------------------------
# Matching cost (Eqs. 3-5 of the paper)
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _support_weights(x, y, feat, half, wlut, wbuf):
    """w(p, q) = exp(-||I_p - I_q||_1 / gamma) for the window around p = (x, y)."""
    H, W = feat.shape[0], feat.shape[1]
    r, g, b = feat[y, x, 0], feat[y, x, 1], feat[y, x, 2]
    for qy in range(max(y - half, 0), min(y + half, H - 1) + 1):
        for qx in range(max(x - half, 0), min(x + half, W - 1) + 1):
            dist = abs(r - feat[qy, qx, 0]) + abs(g - feat[qy, qx, 1]) + abs(b - feat[qy, qx, 2])
            wbuf[qy - y + half, qx - x + half] = wlut[int(dist + 0.5)]


@njit(cache=True, fastmath=True)
def _plane_cost(x, y, a, b, c, wbuf, feat_s, feat_o, sign, max_disp, half,
                alpha, tau_col, tau_grad, bound):
    """Aggregated cost m(p, f) of plane f = (a, b, c) at pixel p = (x, y).

    feat_s / feat_o hold colour (3 channels) and gray-value gradient (2 channels)
    of the own / other view.  Matching points are sampled with linear
    interpolation.  The sum stops early once it reaches `bound` (the candidate
    then cannot win anyway).
    """
    d_p = a * x + b * y + c
    if d_p < 0.0 or d_p > max_disp:
        return INVALID_COST

    H, W = feat_s.shape[0], feat_s.shape[1]
    w_col = 1.0 - alpha
    max_rho = w_col * tau_col + alpha * tau_grad
    x_last = W - 1.0
    total = 0.0
    for qy in range(max(y - half, 0), min(y + half, H - 1) + 1):
        wy = qy - y + half
        row = b * qy + c
        for qx in range(max(x - half, 0), min(x + half, W - 1) + 1):
            w = wbuf[wy, qx - x + half]
            xm = qx - sign * (a * qx + row)          # matching point q' in the other view
            if xm < 0.0 or xm > x_last:
                total += w * max_rho
                continue
            xi = int(xm)
            if xi > W - 2:
                xi = W - 2
            f = xm - xi
            g = 1.0 - f
            d_col = (abs(feat_s[qy, qx, 0] - (g * feat_o[qy, xi, 0] + f * feat_o[qy, xi + 1, 0])) +
                     abs(feat_s[qy, qx, 1] - (g * feat_o[qy, xi, 1] + f * feat_o[qy, xi + 1, 1])) +
                     abs(feat_s[qy, qx, 2] - (g * feat_o[qy, xi, 2] + f * feat_o[qy, xi + 1, 2])))
            d_grad = (abs(feat_s[qy, qx, 3] - (g * feat_o[qy, xi, 3] + f * feat_o[qy, xi + 1, 3])) +
                      abs(feat_s[qy, qx, 4] - (g * feat_o[qy, xi, 4] + f * feat_o[qy, xi + 1, 4])))
            total += w * (w_col * min(d_col, tau_col) + alpha * min(d_grad, tau_grad))
        if total >= bound:
            break
    return total


@njit(cache=True, fastmath=True, parallel=True)
def _init_costs(planes, feat_s, feat_o, sign, max_disp, half, alpha, tau_col, tau_grad, wlut):
    H, W = feat_s.shape[0], feat_s.shape[1]
    cost = np.empty((H, W))
    for y in prange(H):
        wbuf = np.empty((2 * half + 1, 2 * half + 1), np.float32)
        for x in range(W):
            _support_weights(x, y, feat_s, half, wlut, wbuf)
            cost[y, x] = _plane_cost(x, y, planes[y, x, 0], planes[y, x, 1], planes[y, x, 2], wbuf,
                                     feat_s, feat_o, sign, max_disp, half, alpha, tau_col, tau_grad,
                                     INVALID_COST)
    return cost


# -----------------------------------------------------------------------------
# One pixel: spatial propagation, view propagation, plane refinement
# -----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _process_pixel(x, y, step, planes_s, cost_s, planes_o, feat_s, feat_o, sign, max_disp, half,
                   alpha, tau_col, tau_grad, wlut, wbuf, state):
    H, W = cost_s.shape
    _support_weights(x, y, feat_s, half, wlut, wbuf)
    a, b, c = planes_s[y, x, 0], planes_s[y, x, 1], planes_s[y, x, 2]
    best = cost_s[y, x]

    # (a) Spatial propagation: left & upper neighbours (right & lower on odd iterations).
    for k in range(2):
        nx = x - step if k == 0 else x
        ny = y if k == 0 else y - step
        if 0 <= nx < W and 0 <= ny < H:
            ca, cb, cc = planes_s[ny, nx, 0], planes_s[ny, nx, 1], planes_s[ny, nx, 2]
            cost = _plane_cost(x, y, ca, cb, cc, wbuf, feat_s, feat_o, sign, max_disp, half,
                               alpha, tau_col, tau_grad, best)
            if cost < best:
                a, b, c, best = ca, cb, cc, cost

    # (b) View propagation: every pixel p' of the other view (same scanline) whose
    # plane maps it onto p.  The other view matches x' -> x' + sign * d', and with
    # valid disparities p' lies within max_disp pixels of p.
    reach = int(max_disp) + 2
    lo = max(x - reach, 0) if sign > 0 else max(x - 1, 0)
    hi = min(x + 1, W - 1) if sign > 0 else min(x + reach, W - 1)
    for xo in range(lo, hi + 1):
        ao, bo, co = planes_o[y, xo, 0], planes_o[y, xo, 1], planes_o[y, xo, 2]
        d_o = ao * xo + bo * y + co
        if d_o < 0.0 or d_o > max_disp or int(np.floor(xo + sign * d_o + 0.5)) != x:
            continue
        # Express the other view's plane in this view.  With x = x' + sign * d:
        #   d = a' (x - sign * d) + b' y + c'  =>  d = (a' x + b' y + c') / (1 + sign * a')
        den = 1.0 + sign * ao
        if abs(den) < 1e-6:
            continue
        ca, cb, cc = ao / den, bo / den, co / den
        cost = _plane_cost(x, y, ca, cb, cc, wbuf, feat_s, feat_o, sign, max_disp, half,
                           alpha, tau_col, tau_grad, best)
        if cost < best:
            a, b, c, best = ca, cb, cc, cost

    # (c) Plane refinement, in point-normal form: z0 is the disparity at p and
    # n = (-a, -b, 1) / ||.|| the plane normal.  Both are perturbed randomly.
    # The disparity range starts at max_disp / 2, the normal range at 1, and
    # both are halved until the disparity range drops below 0.1.
    z0 = a * x + b * y + c
    inv = 1.0 / np.sqrt(a * a + b * b + 1.0)
    n0, n1, n2 = -a * inv, -b * inv, inv
    dz, dn = max_disp / 2.0, 1.0
    while dz >= 0.1:
        state, r = _rand_sym(state)
        z = z0 + r * dz
        state, r0 = _rand_sym(state)
        state, r1 = _rand_sym(state)
        state, r2 = _rand_sym(state)
        m0, m1, m2 = n0 + r0 * dn, n1 + r1 * dn, n2 + r2 * dn
        norm = np.sqrt(m0 * m0 + m1 * m1 + m2 * m2)
        if 0.0 <= z <= max_disp and norm > 1e-12 and abs(m2) > 1e-6 * norm:
            m0, m1, m2 = m0 / norm, m1 / norm, m2 / norm
            ca, cb = -m0 / m2, -m1 / m2
            cc = (m0 * x + m1 * y + m2 * z) / m2
            cost = _plane_cost(x, y, ca, cb, cc, wbuf, feat_s, feat_o, sign, max_disp, half,
                               alpha, tau_col, tau_grad, best)
            if cost < best:
                a, b, c, best = ca, cb, cc, cost
                z0, n0, n1, n2 = z, m0, m1, m2
        dz *= 0.5
        dn *= 0.5

    planes_s[y, x, 0], planes_s[y, x, 1], planes_s[y, x, 2] = a, b, c
    cost_s[y, x] = best


@njit(cache=True, fastmath=True, parallel=True)
def _sweep(planes_s, cost_s, planes_o, feat_s, feat_o, sign, forward, max_disp, half,
           alpha, tau_col, tau_grad, wlut, seed):
    """One propagation pass over a view, parallelised over anti-diagonals."""
    H, W = cost_s.shape
    step = 1 if forward else -1
    for k in range(H + W - 1):
        y_lo = max(0, k - (W - 1))
        y_hi = min(H - 1, k)
        for i in prange(y_hi - y_lo + 1):
            y = y_lo + i
            x = k - y
            if not forward:
                y = H - 1 - y
                x = W - 1 - x
            wbuf = np.empty((2 * half + 1, 2 * half + 1), np.float32)
            state = _mix64(seed ^ (np.uint64(y * W + x) * _GOLDEN))
            _process_pixel(x, y, step, planes_s, cost_s, planes_o, feat_s, feat_o, sign, max_disp,
                           half, alpha, tau_col, tau_grad, wlut, wbuf, state)


# -----------------------------------------------------------------------------
# Post-processing
# -----------------------------------------------------------------------------
def _plane_to_disparity(planes):
    H, W = planes.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W]
    return planes[..., 0] * xs + planes[..., 1] * ys + planes[..., 2]


def _consistency_check(disp_s, disp_o, sign, threshold=1.0):
    """Valid where the disparity agrees with the other view's at the matching point."""
    H, W = disp_s.shape
    xm = np.rint(np.arange(W)[None, :] - sign * disp_s).astype(np.int64)
    inside = (xm >= 0) & (xm < W)
    d_other = np.take_along_axis(disp_o, np.clip(xm, 0, W - 1), axis=1)
    return inside & (np.abs(disp_s - d_other) <= threshold)


@njit(cache=True)
def _fill_invalid(planes, disp, valid, max_disp):
    """Fill each invalid pixel from the nearest valid pixels to its left and right.

    Both neighbours' planes are extrapolated to the pixel, and the lower
    disparity (the background) wins, since occlusions belong to the background.
    Extrapolations outside [0, max_disp] are ignored.  If neither one is usable,
    the lower of the two neighbours' own disparities is taken.
    """
    H, W = disp.shape
    out = disp.copy()
    left = np.empty(W, np.int64)
    right = np.empty(W, np.int64)
    for y in range(H):
        last = -1
        for x in range(W):
            if valid[y, x]:
                last = x
            left[x] = last
        last = -1
        for x in range(W - 1, -1, -1):
            if valid[y, x]:
                last = x
            right[x] = last
        for x in range(W):
            if valid[y, x]:
                continue
            best = np.inf
            fallback = np.inf
            for xn in (left[x], right[x]):
                if xn < 0:
                    continue
                d = planes[y, xn, 0] * x + planes[y, xn, 1] * y + planes[y, xn, 2]
                if 0.0 <= d <= max_disp and d < best:
                    best = d
                fallback = min(fallback, disp[y, xn])
            if best < np.inf:
                out[y, x] = best
            elif fallback < np.inf:
                out[y, x] = fallback
    return out


@njit(cache=True, parallel=True)
def _weighted_median(disp, valid, feat, half, wlut):
    """Weighted median filter (weights from Eq. 4), applied to invalid pixels only."""
    H, W = disp.shape
    out = disp.copy()
    n_max = (2 * half + 1) ** 2
    for y in prange(H):
        vals = np.empty(n_max)
        wts = np.empty(n_max)
        for x in range(W):
            if valid[y, x]:
                continue
            n = 0
            total = 0.0
            for qy in range(max(y - half, 0), min(y + half, H - 1) + 1):
                for qx in range(max(x - half, 0), min(x + half, W - 1) + 1):
                    dist = (abs(feat[y, x, 0] - feat[qy, qx, 0]) + abs(feat[y, x, 1] - feat[qy, qx, 1]) +
                            abs(feat[y, x, 2] - feat[qy, qx, 2]))
                    w = wlut[int(dist + 0.5)]
                    vals[n] = disp[qy, qx]
                    wts[n] = w
                    total += w
                    n += 1
            order = np.argsort(vals[:n])
            acc = 0.0
            for k in range(n):
                acc += wts[order[k]]
                if acc >= 0.5 * total:
                    out[y, x] = vals[order[k]]
                    break
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def _features(img):
    """H x W x 5 float32: colour (3) + gray-value gradient in x and y (2)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.ascontiguousarray(np.dstack([img.astype(np.float32), gx, gy]))


def _random_planes(H, W, max_disp, rng):
    """Random disparity in [0, max_disp] and random unit normal for every pixel."""
    ys, xs = np.mgrid[0:H, 0:W]
    z0 = rng.uniform(0.0, max_disp, (H, W))
    nz = rng.uniform(1e-3, 1.0, (H, W))          # uniform on the upper hemisphere
    phi = rng.uniform(0.0, 2.0 * np.pi, (H, W))
    r = np.sqrt(1.0 - nz * nz)
    nx, ny = r * np.cos(phi), r * np.sin(phi)
    return np.ascontiguousarray(np.stack([-nx / nz, -ny / nz, (nx * xs + ny * ys + nz * z0) / nz], -1))


def patchmatch_stereo(img_left, img_right, max_disp, iters=3, window=35, gamma=10.0, alpha=0.9,
                      tau_col=10.0, tau_grad=2.0, post_process=True, seed=0, verbose=True):
    """Run PatchMatch Stereo on a rectified image pair (8-bit BGR or gray).

    Returns a dict with the final left / right disparity maps ('disp_left',
    'disp_right'), the maps before post-processing ('raw_left', 'raw_right'),
    the left/right consistency masks ('valid_left', 'valid_right') and the
    per-pixel planes ('planes_left', 'planes_right', each H x W x 3 holding
    (a, b, c) with d = a*x + b*y + c).
    """
    if img_left.shape[:2] != img_right.shape[:2]:
        raise ValueError('left and right images must have the same size')
    max_disp = float(max_disp)
    half = window // 2
    feat_l, feat_r = _features(img_left), _features(img_right)
    H, W = feat_l.shape[:2]
    wlut = np.exp(-np.arange(3 * 255 + 1) / gamma).astype(np.float32)
    params = (max_disp, half, alpha, tau_col, tau_grad)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    rng = np.random.default_rng(seed)
    t0 = time.time()
    planes = {LEFT: _random_planes(H, W, max_disp, rng), RIGHT: _random_planes(H, W, max_disp, rng)}
    feats = {LEFT: (feat_l, feat_r), RIGHT: (feat_r, feat_l)}
    costs = {v: _init_costs(planes[v], *feats[v], v, *params, wlut) for v in (LEFT, RIGHT)}
    log('random initialisation: %.1fs' % (time.time() - t0))

    for it in range(iters):
        forward = it % 2 == 0
        for v in (LEFT, RIGHT):
            t0 = time.time()
            seed64 = np.uint64(rng.integers(0, 2 ** 63))
            _sweep(planes[v], costs[v], planes[-v], *feats[v], v, forward, *params, wlut, seed64)
            log('iteration %d/%d, %s view (%s): %.1fs' % (
                it + 1, iters, 'left' if v == LEFT else 'right',
                'top-left -> bottom-right' if forward else 'bottom-right -> top-left', time.time() - t0))

    raw_l, raw_r = _plane_to_disparity(planes[LEFT]), _plane_to_disparity(planes[RIGHT])
    result = dict(raw_left=raw_l, raw_right=raw_r, planes_left=planes[LEFT], planes_right=planes[RIGHT])
    if not post_process:
        result.update(disp_left=raw_l, disp_right=raw_r,
                      valid_left=np.ones((H, W), bool), valid_right=np.ones((H, W), bool))
        return result

    t0 = time.time()
    valid_l = _consistency_check(raw_l, raw_r, LEFT)
    valid_r = _consistency_check(raw_r, raw_l, RIGHT)
    disp_l = _fill_invalid(planes[LEFT], raw_l, valid_l, max_disp)
    disp_r = _fill_invalid(planes[RIGHT], raw_r, valid_r, max_disp)
    disp_l = _weighted_median(disp_l, valid_l, feat_l, half, wlut)
    disp_r = _weighted_median(disp_r, valid_r, feat_r, half, wlut)
    log('post-processing: %.1fs (%.1f%% / %.1f%% of left / right pixels failed the consistency check)'
        % (time.time() - t0, 100 * (1 - valid_l.mean()), 100 * (1 - valid_r.mean())))
    result.update(disp_left=disp_l, disp_right=disp_r, valid_left=valid_l, valid_right=valid_r)
    return result


def evaluate(disp, gt, threshold=1.0):
    """Percentage of bad pixels (|d - gt| > threshold) among pixels with known ground truth."""
    known = gt > 0
    err = np.abs(disp - gt)[known]
    return 100.0 * np.mean(err > threshold), float(np.mean(err))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description='PatchMatch Stereo (Bleyer et al., BMVC 2011)')
    ap.add_argument('left', nargs='?', default=os.path.join(here, 'im2.png'), help='rectified left image')
    ap.add_argument('right', nargs='?', default=os.path.join(here, 'im6.png'), help='rectified right image')
    ap.add_argument('--max-disp', type=float, default=60, help='maximum disparity in pixels (default: 60)')
    ap.add_argument('--iters', type=int, default=3, help='number of iterations (default: 3)')
    ap.add_argument('--window', type=int, default=35, help='support window size (default: 35)')
    ap.add_argument('--gamma', type=float, default=10.0, help='adaptive-weight bandwidth (default: 10)')
    ap.add_argument('--alpha', type=float, default=0.9, help='colour / gradient balance (default: 0.9)')
    ap.add_argument('--tau-col', type=float, default=10.0, help='colour truncation (default: 10)')
    ap.add_argument('--tau-grad', type=float, default=2.0, help='gradient truncation (default: 2)')
    ap.add_argument('--no-post', action='store_true', help='skip post-processing')
    ap.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    ap.add_argument('--out', default='results', help='output directory (default: results)')
    ap.add_argument('--gt', help='ground-truth disparity image for evaluation, e.g. Middlebury disp2.png')
    ap.add_argument('--gt-scale', type=float, default=4.0,
                    help='ground-truth pixel value per unit disparity (default: 4, Middlebury 2003)')
    ap.add_argument('--show', action='store_true', help='display the result in a window')
    args = ap.parse_args()

    img_l, img_r = cv2.imread(args.left), cv2.imread(args.right)
    if img_l is None or img_r is None:
        raise SystemExit('could not read %s / %s' % (args.left, args.right))
    print('%s + %s: %dx%d, max disparity %g' % (args.left, args.right, img_l.shape[1], img_l.shape[0],
                                                args.max_disp))

    t0 = time.time()
    res = patchmatch_stereo(img_l, img_r, args.max_disp, iters=args.iters, window=args.window,
                            gamma=args.gamma, alpha=args.alpha, tau_col=args.tau_col,
                            tau_grad=args.tau_grad, post_process=not args.no_post, seed=args.seed)
    print('total: %.1fs' % (time.time() - t0))

    os.makedirs(args.out, exist_ok=True)

    def to_u8(d):
        return np.clip(d * (255.0 / args.max_disp), 0, 255).astype(np.uint8)

    for name in ('disp_left', 'disp_right', 'raw_left', 'raw_right'):
        cv2.imwrite(os.path.join(args.out, name + '.png'), to_u8(res[name]))
    np.save(os.path.join(args.out, 'disp_left.npy'), res['disp_left'].astype(np.float32))
    np.save(os.path.join(args.out, 'disp_right.npy'), res['disp_right'].astype(np.float32))
    print('saved disparity maps to %s/' % args.out)

    if args.gt:
        gt_img = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            raise SystemExit('could not read %s' % args.gt)
        gt = gt_img.astype(np.float64) / args.gt_scale
        for name in ('raw_left', 'disp_left'):
            bad, mae = evaluate(res[name], gt)
            print('%-9s bad pixels (error > 1): %5.2f%%   mean abs error: %.3f' % (name, bad, mae))

    if args.show:
        try:
            cv2.imshow('left disparity', to_u8(res['disp_left']))
            cv2.imshow('right disparity', to_u8(res['disp_right']))
            cv2.waitKey(0)
        except cv2.error as e:
            print('cannot display (%s)' % e.err)


if __name__ == '__main__':
    main()
