import cv2
import numpy as np
import math
from tqdm import tqdm


def mloss(pt, fp, color_l, color_r, grad_l, grad_r, alpha=0.9, gamma=10, window_size=35, tau_col=10, tau_grad=2):
    y, x = np.meshgrid(np.arange(pt[1] - window_size // 2, pt[1] + window_size // 2 + 1),
                             np.arange(pt[0] - window_size // 2, pt[0] + window_size // 2 + 1))
    x = x.reshape(-1)
    y = y.reshape(-1)

    xv = (x.astype(np.float) - color_l.shape[1] / 2)
    yv = (y.astype(np.float) - color_l.shape[0] / 2)

    d = (fp[0] * xv + fp[1] * yv + fp[2] + 0.5).astype(np.int)

    weights = np.exp(-np.linalg.norm(color_l[y, x] - color_l[pt[1], pt[0]], ord=1, axis=1) / gamma)

    if np.min(x - d) < 0 or np.max(x - d) >= color_l.shape[1]:
        return np.inf
    rou = (1 - alpha) * np.minimum(np.linalg.norm(color_l[y, x] - color_r[y, x-d], ord=1, axis=1), tau_col) + \
        alpha * np.minimum(np.linalg.norm(grad_l[y, x] - grad_r[y, x-d], ord=1, axis=1), tau_grad)
    loss = np.dot(weights, rou)
    return loss


def PatchMatch(img_l, img_r, dmax):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    grad_l = np.stack([cv2.Sobel(gray_l, cv2.CV_32F, 1, 0, None, 3, 1, 0, cv2.BORDER_DEFAULT),
                       cv2.Sobel(gray_l, cv2.CV_32F, 0, 1, None, 3, 1, 0, cv2.BORDER_DEFAULT)], -1) / 8.
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    grad_r = np.stack([cv2.Sobel(gray_r, cv2.CV_32F, 1, 0, None, 3, 1, 0, cv2.BORDER_DEFAULT),
                       cv2.Sobel(gray_r, cv2.CV_32F, 0, 1, None, 3, 1, 0, cv2.BORDER_DEFAULT)], -1) / 8.
    # cv2.imshow('grad', grad_l[:, :, 1])
    # cv2.waitKey()

    img_l = cv2.copyMakeBorder(img_l, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    img_r = cv2.copyMakeBorder(img_r, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    gray_l = cv2.copyMakeBorder(gray_l, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    gray_r = cv2.copyMakeBorder(gray_r, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    grad_l = cv2.copyMakeBorder(grad_l, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    grad_r = cv2.copyMakeBorder(grad_r, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)

    xv, yv = np.meshgrid(np.arange(img_l.shape[1]), np.arange(img_l.shape[0]))
    xv = xv.astype(np.float) - img_l.shape[1] / 2
    yv = yv.astype(np.float) - img_l.shape[0] / 2
    dp = np.random.rand(img_l.shape[0], img_l.shape[1]) * dmax

    rand1 = np.random.rand(img_l.shape[0], img_l.shape[1])
    rand2 = np.random.rand(img_l.shape[0], img_l.shape[1])

    nx = np.cos(2 * math.pi * rand2) * np.sqrt(1 - rand1 * rand1)
    ny = np.sin(2 * math.pi * rand2) * np.sqrt(1 - rand1 * rand1)
    nz = rand1
    n = np.stack([nx, ny, nz], 2)

    fp = n * (dp / (xv * nx + yv * ny + nz))[..., None]

    cv2.imshow('disp', (fp[:, :, 0] * xv + fp[:, :, 1] * yv + fp[:, :, 2]) / dmax)
    cv2.waitKey(30)

    loss = np.zeros([img_l.shape[0], img_l.shape[1]], dtype=np.float32)
    for x in tqdm(range(dmax, img_l.shape[1] - dmax)):
        for y in range(dmax, img_l.shape[0] - dmax):
            loss[y, x] = mloss((x, y), fp[y, x], img_l, img_r, grad_l, grad_r)

    # print(loss[loss != np.inf].max())
    # cv2.imshow('loss', loss / loss[loss != np.inf].max())
    # cv2.waitKey()
    dnmax = 1
    dzmax = dmax / 2

    def iteration(x, y, inc):
        # spatial propogation
        loss1 = mloss((x, y), fp[y + inc, x], img_l, img_r, grad_l, grad_r)
        if loss[y, x] > loss1:
            fp[y, x] = fp[y + inc, x]
            dp[y, x] = np.dot((x - img_l.shape[1] / 2, y - img_l.shape[0] / 2, 1), fp[y, x])
            loss[y, x] = loss1
        loss2 = mloss((x, y), fp[y, x + inc], img_l, img_r, grad_l, grad_r)
        if loss[y, x] > loss2:
            fp[y, x] = fp[y, x + inc]
            dp[y, x] = np.dot((x - img_l.shape[1] / 2, y - img_l.shape[0] / 2, 1), fp[y, x])
            loss[y, x] = loss2

        # random refinement
        dn = dnmax
        dz = dzmax
        while dz > 0.1:
            normal = fp[y, x] / np.linalg.norm(fp[y, x]) + (np.random.rand(3) * 2 - 1) * dn
            normal = normal / np.linalg.norm(normal)
            disp = dp[y, x] + (np.random.rand() * 2 - 1) * dz
            plane = normal * disp / np.dot((x - img_l.shape[1] / 2, y - img_l.shape[0] / 2, 1), normal)
            potential = mloss((x, y), plane, img_l, img_r, grad_l, grad_r)
            if loss[y, x] > potential:
                fp[y, x] = plane
                dp[y, x] = np.dot((x - img_l.shape[1] / 2, y - img_l.shape[0] / 2, 1), fp[y, x])
                loss[y, x] = potential
            dz /= 2
            dn /= 2

    for it in range(3):
        if it % 2 == 0:
            for y in tqdm(range(dmax + 1, img_l.shape[0] - dmax)):
                for x in range(dmax + 1, img_l.shape[1] - dmax):
                    iteration(x, y, -1)
                cv2.imshow('disp', dp[dmax:-dmax, dmax:-dmax] / dmax)
                cv2.waitKey(1)
        else:
            for y in tqdm(reversed(range(dmax, img_l.shape[0] - dmax - 1))):
                for x in reversed(range(dmax, img_l.shape[1] - dmax - 1)):
                    iteration(x, y, 1)
                cv2.imshow('disp', dp[dmax:-dmax, dmax:-dmax] / dmax)
                cv2.waitKey(1)

        cv2.imshow('disp', dp[dmax:-dmax, dmax:-dmax] / dmax)
        cv2.waitKey(0)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_l = cv2.imread('im2.png')
    img_r = cv2.imread('im6.png')
    PatchMatch(img_l, img_r, 50)
