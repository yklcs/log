---
date: 2024-05-07
---

# Feedforward Gaussian Splatting

3D Gaussian Splatting[^3dgs] comprises the 3D Gaussian representation, the rendering pipeline, and the optimization method.
3D Gaussians are an unstructured explicit representation similar to point clouds, yet they can represent volumetric geometry and view-dependent appearances.
The GPU-accelerated differentiable rendering pipeline renders the 3D Gaussians to an image.
This rendering pipeline allows for direct optimization of the Gaussian parameters based on image-wise losses.

[^3dgs]: _3D Gaussian Splatting for Real-Time Radiance Field Rendering_ (Kerbl et al., SIGGRAPH 2023)

Combined, 3D Gaussians, the rendering pipeline, and the optimization step results in a method to represent 3D scenes from 2D image inputs.
Taking away the optimization step from 3D Gaussian Splatting, we are left with a flexible representation for 3D scenes and a way to render them while allowing for backpropagation.

This means we can use various models to regress 3D Gaussian parameters from image inputs.
In particular, we can use a neural network as the model, which allows us to predict 3D Gaussians in a single forward pass.
This is known as _Feedforward Gaussian Splatting_.

<img src="./diagram.svg" width="150" />

The motivation for Feedforward Gaussian Splatting follows from the usual advantages that deep learning has over classical machine learning.
Using a neural network lets the model to learn priors from large-scale datasets.
This allows Feedforward Gaussian Splatting models to work on sparse views and generalize between scenes.
The original 3D Gaussian Splatting approach, in contrast, must be optimized per scene.

## Methods

Splatter Image[^splatter-image] directly outputs 3D Gaussians from sparse object-centric input views.
A CNN maps a $H \times W \times C$ input image to a $H \times W \times D$ image, where the $D$ output channels represent 3D Gaussian parameters.
This means the output image contains $HW$ 3D Gaussians.

[^splatter-image]: _Splatter Image: Ultra-Fast Single-View 3D Reconstruction_ (Szymanowicz et al., CVPR 2024)

![](./splatter-image.png)

pixelSplat[^pixelsplat] focuses on stereoscopic reconstruction. An encoder based on cross-view- and self- attention maps input views to a feature map.
Each feature of the feature map is used as an input to a variational model which outputs 3D Gaussian parameters.

[^pixelsplat]: _pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction_ (Charatan et al., CVPR 2024)

![](./pixelsplat.png)

Both Splatter Image and pixelSplat are pixel-aligned. Each $(u,v)$ of the input image represent a point in 3D space.
This aspect is used to preserve locality.

MVSplat[^mvsplat] uses a multi-view Transformer to extract features from input views, which are then matched into cost volumes via plane-sweeping.
A U-Net predicts depth maps and 3D Gaussian opacities, covariances, and colors from the cost volumes.
The depth maps are then unprojected to yield 3D Gaussian centers.

[^mvsplat]: _MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images_ (Chen et al., arXiv)

![](./mvsplat.png)
