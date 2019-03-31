import cv2
import numpy as np
import math
from tqdm import tqdm


def mloss(pt, fp, color_l, color_r, grad_l, grad_r, alpha=0.9, gamma=10, window_size=15, tau_col=10, tau_grad=2):
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
        alpha * np.minimum(np.abs(grad_l[y, x] - grad_r[y, x-d]), tau_grad)
    loss = np.dot(weights, rou) / weights.size
    return loss


def PatchMatch(img_l, img_r, dmax):
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    lap_l = cv2.Laplacian(gray_l, cv2.CV_32F)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    lap_r = cv2.Laplacian(gray_r, cv2.CV_32F)

    xv, yv = np.meshgrid(np.arange(img_l.shape[1]), np.arange(img_l.shape[0]))
    xv = xv.astype(np.float) - img_l.shape[1] / 2
    yv = yv.astype(np.float) - img_l.shape[0] / 2
    dp = np.random.rand(img_l.shape[0], img_l.shape[1]) * dmax

    rand1 = np.random.rand(img_l.shape[0], img_l.shape[1])
    rand2 = np.random.rand(img_l.shape[0], img_l.shape[1])

    nx = np.cos(2 * math.pi * rand2) * np.sqrt(1 - rand1 * rand1)
    ny = np.sin(2 * math.pi * rand2) * np.sqrt(1 - rand1 * rand1)
    nz = rand1

    k = (xv * nx + yv * ny + nz) / dp
    a = nx / k
    b = ny / k
    c = nz / k

    fp = np.stack([a, b, c], -1)
    cv2.imshow('disp', (fp[:, :, 0] * xv + fp[:, :, 1] * yv + fp[:, :, 2]) / dmax)
    cv2.waitKey(30)

    img_l = cv2.copyMakeBorder(img_l, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    img_r = cv2.copyMakeBorder(img_r, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    gray_l = cv2.copyMakeBorder(gray_l, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    gray_r = cv2.copyMakeBorder(gray_r, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    lap_l = cv2.copyMakeBorder(lap_l, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    lap_r = cv2.copyMakeBorder(lap_r, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)
    fp = cv2.copyMakeBorder(fp, dmax, dmax, dmax, dmax, cv2.BORDER_CONSTANT, value=0)

    loss = np.zeros([img_l.shape[0], img_l.shape[1]], dtype=np.float32)
    for x in tqdm(range(dmax, img_l.shape[1] - dmax)):
        for y in range(dmax, img_l.shape[0] - dmax):
            loss[y, x] = mloss((x, y), fp[y, x], img_l, img_r, lap_l, lap_r)

    # print(loss[loss != np.inf].max())
    # cv2.imshow('loss', loss / loss[loss != np.inf].max())
    # cv2.waitKey()
    # fp_display = fp.copy()
    # fp_display[loss == np.inf] = 0
    # cv2.imshow('disp', (
    #             fp_display[dmax:-dmax, dmax:-dmax, 0] * xv + fp_display[dmax:-dmax, dmax:-dmax, 1] * yv + fp_display[
    #                                                                                                       dmax:-dmax,
    #                                                                                                       dmax:-dmax,
    #                                                                                                       2]) / dmax)
    # cv2.waitKey(0)
    for it in range(5):
        # spatial propogation
        for x in tqdm(range(dmax + 1, img_l.shape[1] - dmax)):
            for y in range(dmax + 1, img_l.shape[0] - dmax):
                loss1 = mloss((x, y), fp[y - 1, x], img_l, img_r, lap_l, lap_r)
                if loss[y, x] > loss1:
                    fp[y, x] = fp[y - 1, x]
                    loss[y, x] = loss1
                loss2 = mloss((x, y), fp[y, x - 1], img_l, img_r, lap_l, lap_r)
                if loss[y, x] > loss2:
                    fp[y, x] = fp[y, x - 1]
                    loss[y, x] = loss2

        fp_display = fp.copy()
        fp_display[loss == np.inf] = 0
        cv2.imshow('disp', (fp_display[dmax:-dmax, dmax:-dmax, 0] * xv + fp_display[dmax:-dmax, dmax:-dmax, 1] * yv + fp_display[dmax:-dmax, dmax:-dmax, 2]) / dmax)
        cv2.waitKey(0)

        for x in tqdm(reversed(range(dmax, img_l.shape[1] - dmax - 1))):
            for y in reversed(range(dmax, img_l.shape[0] - dmax - 1)):
                loss1 = mloss((x, y), fp[y + 1, x], img_l, img_r, lap_l, lap_r)
                if loss[y, x] > loss1:
                    fp[y, x] = fp[y + 1, x]
                    loss[y, x] = loss1
                loss2 = mloss((x, y), fp[y, x + 1], img_l, img_r, lap_l, lap_r)
                if loss[y, x] > loss2:
                    fp[y, x] = fp[y, x + 1]
                    loss[y, x] = loss2

        fp_display = fp.copy()
        fp_display[loss == np.inf] = 0
        cv2.imshow('disp', (fp_display[dmax:-dmax, dmax:-dmax, 0] * xv + fp_display[dmax:-dmax, dmax:-dmax,
                                                                         1] * yv + fp_display[dmax:-dmax, dmax:-dmax,
                                                                                   2]) / dmax)
        cv2.waitKey(0)
        # random improvement
        # for x in range(img_l.shape[1]):
        #     for y in range(img_l.shape[0]):
        #         dz = dmax / 2
        #         dn = 1
        #         while dz > 0.1:
        #             dz_rand = (np.random.rand() * 2 - 1) * dz
        #             z_bk = fp[y, x, -1]
        #             fp[y, x, -1] += dz_rand
        #             potential = mloss((x, y), fp[y, x], img_l, img_r, lap_l, lap_r)
        #             if loss[y, x] < potential:
        #                 fp[y, x, -1] = z_bk
        #             else:
        #                 loss[y, x] = potential
        #             dz /= 2
        # cv2.imshow('improve%d' % it, ((fp[:, :, 0] * xv + fp[:, :, 1] * yv + fp[:, :, 2]).astype(np.float32)) / dmax)
        # cv2.waitKey(0)


if __name__ == '__main__':
    img_l = cv2.imread('im2.png')
    img_r = cv2.imread('im6.png')
    PatchMatch(img_l, img_r, 50)
