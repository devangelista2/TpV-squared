# TpV-squared

## Installation
To replicate the proposed experiments, just clone the Github repository by

```
git clone https://github.com/devangelista2/TpV_RISING.git
cd TpV_RISING
```

Please note that to run the experiments, the following packages are required:
- `pytorch`
- `astra-toolbox`
- `numpy`
- `matplotlib`

Since our experiments make large use of `astra-toolbox`, it is also required to have access to a cuda-compatible graphics card. 

## Dataset
Our experiments have been performed on the COULE dataset (available on Kaggle at https://www.kaggle.com/datasets/loiboresearchgroup/coule-dataset). The dataset consists of 430 grey-scale images of dimension $256 \times 256$ representing ellipses of different contrast levels and small, high-contrasted dots, that imitates the human body structure, divided in 400 training images and 30 test images. More informations about the structure of the dataset is available at the link above.

<p align="center">
<img src="./data/COULE/test/gt/0.png">
</p>

## Methods
We consider $x \in \mathbb{R}^n$ to be a grey-scale image representing the (unknown) interior scan we want to reconstruct. Also, given an angular range $\Gamma \subseteq [0, 2\pi]$ discretized into $N_\alpha$ uniformly distributed angles, we define the CT forward projector (with fan-beam geometry) $K \in \mathbb{R}^{m \times n}$ where $m = N_\alpha \cdot N_d$, $N_d$ being the number of pixel of the detector, which in the experiments is equal to $2\sqrt{n}$. The forward problem reads

```math
y^\delta = Kx + e \quad e \sim \mathcal{N}(0,\sigma^2I).
```

where $\sigma^2$ is selected such that $||e|| \leq \delta$ with high probability. 

Given $p \in (0, 1]$, we consider the regularized variational functional

```math
    \min_{x \geq 0} \frac{1}{2} || Kx - y^\delta ||_2^2 + \frac{\lambda}{p} || \nabla x ||_{2, p}^p
```

where 

```math
    || \nabla x ||_{2, p}^p = \sum_{i=1}^n \Bigl( \sqrt{(\nabla_h x)_i^2 + (\nabla_v x)_i^2} \Bigr)^p
```

is the isotropic Total Variation operator of order $p>0$.

Thid problem can be solved by using the Chambolle-Pock (CP) algorithm with reweighting. The resulting algorithm is called CP-TpV.

Given $k \in \mathbb{N}$, we define the operator $\phi_k^p: \mathbb{R}^m \times \mathbb{R}^n \to \mathbb{R}^n$ such that $\phi_k^p(y^\delta, x^{(0)})$ models the application of $k$ iteration of the CP-TpV algorithm over the sinogram $y^\delta$, starting from the initial guess $x^{(0)}$. When the algorithm is executed until convergence (to a local minima), then the associated operator will be named $\hat{\phi}^p$. Finally, in the case where $x^{(0)} = 0$ and there is no confusion, we will redefine it from the input of $\phi_k^p(y^\delta) = \phi_k^p(y^\delta, x^{(0)})$ and $\hat{\phi}^p(y^\delta) = \hat{\phi}^p(y^\delta, x^{(0)})$. 

With the notation above, we define:

- $x^{IS} = \hat{\phi}^p(y^\delta)$;
- $x^{RIS} = \phi_k^p(y^\delta)$,

where *IS* stands for *Iterative Solution*, while *RIS* stands for *Rapid Iterative Solution*.

Given a set of ground-truth images $` \{ x^{GT}_{i} \}_{i=1}^N \subseteq \mathbb{R}^n `$, we consider, for any $` i = 1, \dots, N$, $x^{RIS}_{i} = \phi^{p}_{k}(y^{\delta}_{i}) `$ and $` x^{IS}_{i} = \hat{\phi}^p(y^{\delta}_{i}) `$, where $` y^{\delta}_{i} = Kx^{GT}_{i} + e_{i} `$. A neural network $\psi_{\theta}$ trained to map $x^{RIS}$ to $x^{IS}$ is called **TpV-Net** in the following.

Furthermore, to obtain a solution which is provably a stationary point for the TpT-regularized inverse problem, we consider an algorithm where the TpV-Net solution is employed as starting iteration of the Chambolle-Pock algorithm. The rationale for this choice is that if TpV-Net is able to generate high quality images, we can assume its output is close to the global minima of the objective function. Thus, executing an iterative algorithm with is as starting point, will be able to reach a good stationary point in fewer iterations than the usual method that starts from $x^{(0)} = 0$. We call this method **TpV-squared**.

The main objective of this work is to explore how the TpV-Net and TpV-squared approached behave on non-convex TpV optimization problem, compared to the classical TpV-CP iterative algorithm. In particular, we want to understand:

- If the output of TpV-Net is close to the local minima $x^{IS}$;
- If the local minima computed by TpV-squared is close to the prediction of TpV-Net;
- If $x^{IS}$ and the output of TpV-squared represents the same local minima;
- If TpV-squared requires fewer iterations than TpV-CP to converge.

In particular, we tested the above problem on a SparseCT inverse problem, where the operator $K$ describes the fanbeam Radon transforms, with angular range $\Gamma = [0, 180]$ and number of discretization angles $N_\alpha = 180$. We refer to this setup as **Mild Sparse**. We plan to extend those methods to a **Severely Sparse** problem, where the number of projection angles is reduced to $N_\alpha = 60$. We test each method with $p \in \{ 0.1, 0.5, 1 \}$ in the following. We remark that when $p=1$, the resulting optimization problem is convex.

## Usage

> **NOTE:** To replicate the experiments, it is required to either train the models again by running the command: 
> ```
> python train.py --batch_size BATCHSIZE --n_epochs N_EPOCHS --p P --device DEVICE
> ```
> with your choice of BATCHSIZE, N_EPOCHS, P and DEVICE, or by downloading the pretrained models from the link below in the corresponding section. Note that the weights are supposed to be placed in a folder named `model_weights` that has to created, whose structure follows the folder structure of the downloadable model weights from Google Drive. 

The results present in the paper, can be simply obtained by running the `test.py` file with the desired parameters. For example, the command

```
python test.py --p 1 --mode test --model unet --device cuda
```

will run the test with $p = 1$ on the test set, by computing the TpV-Net solution with a UNet model on GPU. To get the list of available settings with the documentation, just type:

```
python test.py --h
```

## Pre-trained models
The weights for the pre-trained models is available for the download on Google Drive at the following URL: https://drive.google.com/drive/folders/1GoVA3jZKafdlzoOK4lX0SblntKUM8PNE?usp=sharing

## Cite Us
The paper associated with this repository has been submitted to the NUMTA2023 conference. The BibTex to cite the paper will be available as soon as the paper will be published. 