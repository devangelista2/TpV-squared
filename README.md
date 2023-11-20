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

\begin{align}
    y^\delta = Kx + e \quad e \sim \mathcal{N}(0,\sigma^2I)
\end{align}

where $\sigma^2$ is selected such that $||e|| \leq \delta$ with high probability. 

Given $p \in (0, 1]$, we consider the regularized variational functional

\begin{align}
    \min_{x \geq 0} \frac{1}{2} || Kx - y^\delta ||_2^2 + \frac{\lambda}{p} || \nabla x ||_{2, p}^p
\end{align}

where 

\begin{align}
    || \nabla x ||_{2, p}^p = \sum_{i=1}^n \Bigl( \sqrt{(\nabla_h x)_i^2 + (\nabla_v x)_i^2} \Bigr)^p
\end{align}

is the isotropic Total Variation operator of order $p>0$.

Thid problem can be solved by using the Chambolle-Pock (CP) algorithm with reweighting. The resulting algorithm is called CP-T$_p$V.

Given $k \in \mathbb{N}$, we define the operator $\phi_k^p: \mathbb{R}^m \times \mathbb{R}^n \to \mathbb{R}^n$ such that $\phi_k^p(y^\delta, x^{(0)})$ models the application of $k$ iteration of the CP-T$_p$V algorithm over the sinogram $y^\delta$, starting from the initial guess $x^{(0)}$. When the algorithm is executed until convergence (to a local minima), then the associated operator will be named $\hat{\phi}^p$. Finally, in the case where $x^{(0)} = 0$ and there is no confusion, we will redefine it from the input of $\phi_k^p(y^\delta) = \phi_k^p(y^\delta, x^{(0)})$ and $\hat{\phi}^p(y^\delta) = \hat{\phi}^p(y^\delta, x^{(0)})$. 

With the notation above, we define:

- $x^{IS} = \hat{\phi}^p(y^\delta)$;
- $x^{RIS} = \phi_k^p(y^\delta)$.

Given a set of ground-truth images $\{ x_i^{GT} \}_{i=1}^N \subseteq \mathbb{R}^n$, we consider, for any $i = 1, \dots, N$, $x^{RIS}_i = \phi_k^p(y^\delta_i)$ and $x^{IS}_i = \hat{\phi}^p(y^\delta_i)$, where $y^\delta_i = Kx^{GT}_i + e_i$. A neural network $\psi_\theta$ trained to map $x^{RIS}$ to $x^{IS}$ is called T$_p$V-RISING in the following. \\

The main objective of this work is to explore how T$_p$V-RISING behaves on non-convex optimization problems. In particular, we remark that if $p<1$, then \eqref{eq:inverse_problem} is non-convex, thus $\hat{\phi}^p(y^\delta)$ is, in general, just a local minima. In particular, named $x^{RISING} = \psi_\theta(y^\delta)$, we want to understand:

- If $x^{RISING}$ is close to the local minima $x^{IS}$;
- If the local minima closest to $x^{RISING}$, namely $\hat{x}^{IS} = \hat{\phi}^p(y^\delta, x^{RISING})$, is far from $x^{RISING}$;
    \item If $x^{IS}$ and $\hat{x}^{IS}$ represents the same local minima.
\end{itemize}

This will be done by computing the distance between each couple of points, measured as euclidean distance and SSIM. In particular, we considered the following scenarios:

\begin{itemize}
    \item \textbf{Mild Sparse}: $\Gamma = [0, 180]$ with $N_\alpha = 180$;
    \item \textbf{Severe Sparse}:  $\Gamma = [0, 180]$ with $N_\alpha = 60$;
\end{itemize}

each of them with $p \in \{ 0.01, 0.1, 0.5, 1 \}$. Remember that when $p=1$, the resulting optimization problem is convex.

## Pre-trained models
The weights for the pre-trained models is available for the download on Google Drive at the following URL: https://drive.google.com/drive/folders/1GoVA3jZKafdlzoOK4lX0SblntKUM8PNE?usp=sharing
