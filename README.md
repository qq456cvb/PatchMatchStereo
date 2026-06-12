# PatchMatchStereo

A compact Python/NumPy implementation of [PatchMatch Stereo — Stereo Matching with Slanted Support Windows](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/01/PatchMatchStereo_BMVC2011_6MB.pdf) (Bleyer, Rhemann & Rother, BMVC 2011).

Instead of assigning each pixel a fronto-parallel disparity, PatchMatch Stereo estimates a **slanted disparity plane** per pixel, which models tilted surfaces correctly and yields sub-pixel disparities. Good planes are found surprisingly fast with PatchMatch-style random search: start from random planes, then alternately propagate good planes from neighbors and refine them with a shrinking random search.

## What's Implemented

- **Per-pixel slanted planes**: each pixel carries a 3-parameter plane; disparity is evaluated as a plane equation over the support window
- **Adaptive support-weight matching cost**: truncated color + gradient dissimilarity, weighted by color similarity to the window center (`mloss`, Eq. 3–5 of the paper)
- **Random initialization**: plane normals sampled uniformly on the hemisphere, random disparities within `[0, dmax]`
- **Spatial propagation**: alternating raster / reverse-raster passes adopting a neighbor's plane when it lowers the cost
- **Plane refinement**: random perturbation of normal and disparity with exponentially shrinking search radii

Not implemented (contributions welcome): view propagation, post-processing (left-right consistency + weighted median filling), and a CUDA port — the pure-NumPy inner loop is slow, so expect minutes per iteration.

## Run

```bash
pip install opencv-python numpy tqdm
python main.py
```

The demo runs three propagation/refinement iterations on the bundled Middlebury *Cones* pair (`im2.png` / `im6.png`, `dmax = 50`) and displays the evolving disparity map after each row (press a key after each iteration to continue).

## Result (after 1 iteration)

![result](https://github.com/qq456cvb/PatchMatchStereo/raw/master/result1.PNG)

## Reference

```bibtex
@inproceedings{bleyer2011patchmatch,
  title={PatchMatch Stereo - Stereo Matching with Slanted Support Windows},
  author={Bleyer, Michael and Rhemann, Christoph and Rother, Carsten},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2011}
}
```
