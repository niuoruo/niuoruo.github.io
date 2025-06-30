---
layout: post
title: "State of the Art on Monocular 3D Face Reconstruction, Tracking, and Applications 论文阅读（三）"
date: 2024-06-26 21:20:00 +0800
categories: Avator
---

# State of the Art on Monocular 3D Face Reconstruction, Tracking, and Applications 论文阅读（三）

上一篇讲解了一个人脸模型的构成，那么有了这个模型，如何根据我们的输入估计参数呢？本篇将详细介绍。原论文中的公式大多数是直接给出，我会更加详细地推导这些公式的由来。

![image-20250627102040834](https://niuoruo.github.io/assets/images/image-20250627102040834.png)

## 参数估计方法

### 最小化损失函数

给定单目视频作为输入，参数估计通常被表述为一个一般的非线性优化问题。为此，首先将所有未知量堆叠到参数向量$P$中。最近的最先进方法通常针对以下全部或部分参数进行优化：全局头部**旋转$R$**和**平移$t$**，**面部形状${\alpha_i}_{i=1}^{m_s}$**，**反射率${\beta_i}_{i=1}^{m_r}$**，**表情${\sigma_i}_{i=1}^{m_e}$**和**光照${l_i}_{i=1}^{m_i}$**。

通过最小化重建 / 跟踪目标函数$E$，找到最优参数$P^*$：

$$\mathcal{P}^{*}=\underset{\mathcal{P}}{\operatorname{argmin}}E(\mathcal{P})\;$$

一般来说，跟踪的目标在未知参数方面具有高度非线性，并且由以下组件组成：

$$E(\mathcal{P})=\underbrace{wdenseEdense(\mathcal{P})+wsparseEsparse(\mathcal{P})}_{data}+\underbrace{wregEreg(\mathcal{P})}_{prior}$$

**稀疏特征对齐项**$Esparse(\mathcal{P})$，将模型的少量特征点与输入中检测到的对应特征点匹配；**密集对齐项**$Esparse(\mathcal{P})$，在顶点或像素级别施加约束；**正则化项**$Ereg(\mathcal{P})$，约束求解。下面会一一解释具体做法。

- 稀疏特征对齐

由于人脸包含许多视觉显著的结构，最常用且最重要的数据项之一是**稀疏特征对齐**：

$$E_{\mathrm{lan}}(\mathcal{P})=\frac{1}{|\mathcal{F}|}\sum_{\mathbf{f}_j\in\mathcal{F}}w_{\mathrm{conf},j}\left\|\mathbf{f}_j-\mathbf{K}\Pi(\mathbf{R}\cdot\mathbf{v}_j(\mathcal{P})+\mathbf{t})\right\|_2^2$$

这一项约束使得面部特征点$v_j(\mathcal{P})$接近输入2D图像中检测出的特征点$f_j\in F\in\mathbb{R}^2$。每每次检测通常都带有一个置信度$w_{conf, j}$，可用于对相应的对齐约束进行加权。

关于面部特征和标志点的检测与跟踪方法，已有大量文献。一些方法是**整体式**的，利用整个面部区域来定位面部标志点，如主动外观模型（AAM）。另一些是**局部式**的，通常使用约束条件来稳定估计值。这些受限局部模型（CLM）方法快速且稳健。因此，它们通常被用作密集面部跟踪和重建的面部特征跟踪器。

鉴于能量最小化问题具有高度非线性，$E_{lan}$是一个非常重要的能量项。面部特征点不仅对于在第一帧中初始化跟踪器很重要，而且能使优化更接近密集对齐约束的收敛域。没有这个数据项，快速运动和强烈变形就无法得到可靠的跟踪。可以看作是一种全局配准过程，没有这个配准，后续局部上的梯度优化可能会陷入局部最优解，无法找到全局最优。

![image-20250627173349035](https://niuoruo.github.io/assets/images/image-20250627173349035.png)

- 密集光度对齐

稀疏数据项使模型大致与输入对齐，这样优化过程就**处于密集数据的收敛域**内。而光度对齐密集地衡量当前拟合的渲染版本对输入数据的匹配程度，可以更细化的优化模型：

$$E_{\mathrm{col}}(\mathcal{P})=\frac{1}{|\mathcal{V}|}\sum_{\mathcal{p}\in\mathcal{V}}\|C_\mathcal{S}(\mathcal{P,p})-C_\mathcal{I}(\mathcal{p})\|_2^2$$

这一项约束使得渲染的$C_S$图像接近输入的$C_I$图像，$p\in V$表示在$C_S$中用于比较的所有可见像素位置。出于性能考虑，此约束通常在顶点级别而不是像素或片段级别定义。使用不同的范数来衡量接近程度

![image-20250627173405641](https://niuoruo.github.io/assets/images/image-20250627173405641.png)

- 密集几何对齐

上一部分提到的密集对齐是颜色上的对齐，而这一部分将在深度方面对齐，因此这里应用于使用RGB-D相机输入的算法。最常用的对齐项测量模型表面点与输入深度之间的点对点距离。

$$Epoint(\mathcal{P})=\frac{1}{|\mathcal{V}|}\sum_{p\in\mathcal{V}}\|X_\mathcal{S}(\mathcal{P,p})-X_\mathcal{I}(\mathcal{p})\|_{2}^{2}$$

其中$XS(\mathcal{P, p})-XI(\mathcal{p})$是RGB-D测量的三维位置与相应三维模型之间的差异。$V$ 表示用于比较的所有可见像素位置。一般情况下，将这个计算简化为每个顶点计算而非对每个像素计算。

为了提高鲁棒性，通常会使用一阶表面近似距离，这种方法在很多激光SLAM中也有使用，如FAST-LIO，将原本的点与点匹配变更为点与面匹配，点优化的梯度方向为模型面的法向，能够显著提高鲁棒性

$$E_{\mathrm{plane}}(\mathcal{P})=\frac{1}{|\mathcal{V}|}\sum_{\mathcal{p}\in\mathcal{V}}\left[N_{\mathcal{S}}(\mathcal{P,p})^\top\cdot(X_{\mathcal{S}}(\mathcal{P},\mathcal{p})-X_{\mathcal{I}}(\mathcal{p}))\right]^2$$

其中$N_S$表示表面的法线向量，一般使用模型的表面法线，因为RGB-D输入的深度很可能不是平滑的。一些方法建议使用一种对称的点到平面距离，既从输入到模型，也从模型到输入。

- 正则化

利用人脸先验中可用的统计信息，重建问题可以进一步加以约束，以解决模糊性问题：

$$E_{\mathrm{reg}}(\mathcal{P})=\sum_{i=1}^{m_s}\left(\frac{\alpha_i}{\sigma_{s,i}}\right)^2+\sum_{i=1}^{m_r}\left(\frac{\beta_i}{\sigma_{r,i}}\right)^2+\sum_{i=1}^{m_e}\left(\frac{\delta_i}{\sigma_{e,i}}\right)^2$$

该约束条件强制模型参数在统计意义上接近其均值。这是一种常用的正则化策略，可防止面部几何形状和面部反射率的退化。**正则化在单目 RGB 重建场景中尤为重要**，因为单目相机无法获取深度。

### 模型初始化

虽然对所有参数进行联合估计通常是可行的，但几乎所有方法都会**在算法的某个阶段固定脸型（shape）**，随后仅对其余参数进行优化。许多方法在输入序列的起始阶段使用多幅图像，与从单帧重建相比，这能够实现更好的脸型估计。以下介绍一些不同采用策略的示例。Weise等人基于刚性对齐融合多个深度帧，以获得中性面部的 3D 模型。Thies等人对具有不同头部姿态的 3 帧进行基于模型的非刚性光束平差法。Ichim等人基于 iPhone 拍摄的多视图数据执行静态运动恢复结构重建。

### 处理遮挡

遮挡处理对于稳健的人脸跟踪至关重要。胡须、手或其他遮挡物很容易使前面描述的人脸重建和跟踪方法性能下降。因此使用分割掩码，只提取图像中感兴趣的人脸区域至关重要。当时还没有很多分割模型，目前一些分割算法有很强的zero shot能力，如facebook research划时代的*Segment Anything*

![image-20250627183707697](https://niuoruo.github.io/assets/images/image-20250627183707697.png)

### 优化策略

合成分析方法的主要组成部分之一是估计面部参数，以使输入的面部图像与合成的面部图像之间的差异最小化。通常，为了找到这个参数估计值，必须解决一个优化问题。合成分析方法从使用旧参数$P_i$生成的合成图像开始，迭代计算一组新的参数$P_{i+1}$。选择参数$P_{i+1}$，以便使衡量观测图像与合成图像之间差异的能量最小化。

下面举一个例子方便解释优化策略，为简单起见，我们假设一个目标函数$E(\mathcal{P})$，该函数由单个数据拟合组成。

$$E(\mathcal{P})=\frac{1}{|\mathcal{V}|}\sum_{\mathbf{p}\in\mathcal{V}}\Psi(\mathbf{r}(\mathcal{P},\mathbf{p}))$$

其中$\Psi:\mathbb{R}^n\to\mathbb{R}_{\geq0}$代表一个**将观测量与模型的差异$\mathbf{r}(\mathcal{P},\mathbf{p})=C_{\mathcal{S}}(\mathcal{P},\mathbf{p})-C_{\mathcal{I}}(\mathbf{p})$映射为标量**的函数，联系一下上面所说的，这可以是稀疏特征点在法向距离上的距离和，可以是顶点的颜色误差和或三维位置误差和，也可以是正则化项——到平均脸型的距离和。

最常见的度量是$\ell_2\mathrm{-norm}(\Psi(\mathbf{x})=||\mathbf{x}||_2^2)$。在这种情况下，优化问题归结为一个最小二乘问题。因此，上式可以改写为：

$$E(\mathcal{P})=\frac{1}{|\mathcal{V}|}\sum_{\mathbf{p}\in\mathcal{V}}||\mathbf{r}(\mathcal{P},\mathbf{p})||_2^2=||F(\mathcal{P})||_2^2$$

- **线性**问题

  如果$F(\mathcal{P})$是模型参数$\mathcal{P}$中的线性函数，那么优化问题就简化为线性最小二乘问题$(\|A\mathcal{P}+b\|^{2}\to min)$，可以用解析的方法，求距离原问题最近的解，乘$A^{\top}$使其半正定：$A^{\top}A\mathcal{P}=-A^{\top}\mathbf{b}$

  但在很多目前最优秀的重建工作中，大部分$F(\mathcal{P})$都是非线性的，回忆一下之前的相机针孔模型，这个投影过程本身就是非线性的，由于噪声的存在，当我们把图像输入的观测直接带入方程中时，它们并不会完美的成立。这时候怎么办呢？我们把状态的估计值进行微调，使得整体的误差下降一些。当然这个下降也有限度，它一般会到达一个极小值。这就是一个典型非线性优化的过程。

- **解析**方法求**非线性**问题

  如果问题$F(\mathcal{P})$是个**很简单的函数**，那问题也许可以用解析形式来求。**令目标函数的导数为零，然后求解$\mathcal{P}$的最优值**，就和一个求二元函数的极值一样：

  $$\frac{\mathrm{d}F(\mathcal{P})}{\mathrm{d}\mathcal{P}}=\mathbf{0}$$

  解此方程，就得到了导数为零处的极值。它们可能是极大、极小或鞍点处的值，只要挨个比较它们的函数值大小即可。但是，这个导数等于0的方程是否容易求解呢？这取决于$F(\mathcal{P})$导数的形式。人脸本身的顶点众多，再加上之前提到的诸多优化项，因此我们**不太能够顺利求解这样一个复杂的非线性方程**。

- **迭代**方法求**非线性**问题

  对于不方便直接求解的最小二乘问题，我们可以用迭代的方式，**从一个初始值出发，不断地更新当前的优化变量**，使目标函数下降。具体步骤可列写如下:

  1. 给定某个初始值$\mathcal{P}$，这里可以使用由稀疏特征点PNP或者使用learning-based方法获得的初值。
  2. 对于第 k 次迭代，寻找一个增量$\Delta\mathcal{P}$，使得$\left\|F\left(\mathcal{P}_k+\Delta\mathcal{P}_k\right)\right\|_2^2$达到极小值。
  3. 若$\Delta \mathcal{P}$足够小，则停止。
  4. 否则，令$\mathcal{P}_{k+1}=\mathcal{P}_k+\Delta \mathcal{P}_k$，返回2。

  这让求解导函数为零的问题，变成了一个**不断寻找梯度并下降**的过程。直到某个时刻增量非常小，无法再使函数下降。此时算法收敛，目标达到了一个极小，我们完成了寻找极小值的过程。在这个过程中，我们只要找到迭代点的梯度方向即可，而无需寻找全局导函数为零的情况。

  接下来的问题是：增量$\Delta \mathcal{P}_k$如何确定？接下来将介绍几种方法：

  - 一阶和二阶梯度法

    求解增量最直观的方式是将目标函数在$\mathcal{P}$附近**泰勒展开**，这可以获得一个近似于原式的表达式，我们**暂时将其视为求解目标**：

    $$\|F(\mathcal{P}+\Delta\mathcal{P})\|_2^2\approx\|F(\mathcal{P})\|_2^2+\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}+\frac{1}{2}\Delta\mathcal{P}^T\boldsymbol{H}(\mathcal{P})\Delta\mathcal{P}$$

    其中$\boldsymbol{J}$是$\|F(\mathcal{P})\|_2^2$关于$\mathcal{P}$的导数（雅可比矩阵），而$\boldsymbol{H}$是二阶导数（海塞矩阵），我们可以选择保留泰勒展开的一阶或二阶项，对应的求解方法则为一阶梯度或二阶梯度法。如果**保留一阶梯度**，那么增量的方向：

    $$\Delta \mathcal{P}^*=-\boldsymbol{J}^T(\mathcal{P}).$$

    它的直观意义非常简单，只要我们沿着反向梯度方向前进即可。当然，我们还需要该方向上取一个步长$\lambda$，表示我们沿这个方向下降多少，这种方法称为**最速下降法**。

    如果保留**二阶梯度**，经过了二阶泰勒展开，求解目标的形式会变得更加简单，此时我们相当于**使用解析的方法求这个二阶泰勒展开的最小值**——对其求关于$\Delta \mathcal{P}$的导数并使令它为0，就得到了增量的解：

    $$\Delta\mathcal{P}^*=\arg\min\|F\left(\mathcal{P}\right)\|_2^2+\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}+\frac{1}{2}\Delta\mathcal{P}^T\boldsymbol{H}(\mathcal{P})\Delta\mathcal{P}$$

    $$\boldsymbol{H}(\mathcal{P})\Delta\mathcal{P}=-\boldsymbol{J}(\mathcal{P})^T$$

    因此**论文中给出的方程其实是最后的更新方程**：

    $$\mathcal{P}_{k+1}=\mathcal{P}_k-\boldsymbol{H}(\mathcal{P}_k)^{-1}\boldsymbol{J}(\mathcal{P}_k)^T$$

    该方法称又为**牛顿法**。我们看到，一阶和二阶梯度法都十分直观，只要把函数在迭代点附近进行泰勒展开，并针对更新量作最小化即可。由于泰勒展开之后函数变成了多项式，所以求解增量时只需解线性方程即可，避免了直接求导函数为零这样的非线性方程的困难。
    不过，这两种方法也存在它们自身的问题。**最速下降法过于贪心，容易走出锯齿路线**，反而增加了迭代次数。而**牛顿法则需要计算目标函数的$\boldsymbol{H}$矩阵，这在问题规模较大时非常困难**，我们通常倾向于避免$\boldsymbol{H}$的计算。所以，接下来我们详细地介绍两类更加实用的方法：高斯牛顿法和列文伯格——马夸尔特方法。

  - Gauss-Newton

    Gauss Newton 是最优化算法里面最简单的方法之一。它的思想是将$F(\mathcal{P})$进行一阶的泰勒展开（请注意不是目标函数$\|F(\mathcal{P})\|_2^2$）:

    $$F(\mathcal{P}+\Delta\mathcal{P})\approx F(\mathcal{P})+\boldsymbol{J}(\mathcal{P})\Delta\mathcal{P}$$

    根据前面的框架，当前的目标是为了寻找下降矢量$\Delta\mathcal{P}$，使得$\|F\left(\mathcal{P}+\Delta\mathcal{P}\right)\|_2^2$达到最小。为了求$\Delta\mathcal{P}$，我们需要解一个线性的最小二乘问题：

    $$\Delta\mathcal{P}^*=\arg\min_{\Delta\mathcal{P}}\frac{1}{2}\left\|F\left(\mathcal{P}\right)+\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}\right\|^2$$

    展开平方项：

    $$\begin{aligned}\frac{1}{2}\|F\left(\mathcal{P}\right)+\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}\|^2&=\frac{1}{2}\left(F\left(\mathcal{P}\right)+\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}\right)^T\left(F\left(\mathcal{P}\right)+\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}\right)\\&=\frac{1}{2}\left(\|F(\mathcal{P})\|_2^2+2F\left(\mathcal{P}\right)^T\boldsymbol{J}(\mathcal{P})\Delta\mathcal{P}+\Delta\mathcal{P}^T\boldsymbol{J}(\mathcal{P})^T\boldsymbol{J}(\mathcal{P})\Delta\mathcal{P}\right)\end{aligned}$$

    使用解析的方法求解平方项的最小值——求导并使其为零：

    $$2\boldsymbol{J}\left(\mathcal{P}\right)^TF\left(\mathcal{P}\right)+2\boldsymbol{J}\left(\mathcal{P}\right)^T\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}=\boldsymbol{0}$$

    可得到如下方程组：

    $$\boldsymbol{J}(\mathcal{P})^T\boldsymbol{J}(\mathcal{P})\Delta\mathcal{P}=-\boldsymbol{J}(\mathcal{P})^TF(\mathcal{P})$$

    注意，我们要求解的变量是$\Delta\mathcal{P}$，因此这是一个线性方程组，我们称它为**增量方程**，也可以称为**高斯牛顿方程 (Gauss Newton equations)** 或**者正规方程 (Normal equations)**。我们把左边的系数定义为$\boldsymbol{H}$，右边定义为$\boldsymbol{g}$，那么上式变为：

    $$\boldsymbol{H}(\mathcal{P})\Delta\mathcal{P}=\boldsymbol{g}$$

    这里把左侧记作$\boldsymbol{H}$是有意义的。对比牛顿法可见，Gauss-Newton用$\boldsymbol{J}^T\boldsymbol{J}$作为牛顿法中二阶 Hessian 矩阵的近似，从而省略了计算$\boldsymbol{H}$的过程。**求解增量方程是整个优化问题的核心所在**。

    将$\Delta\mathcal{P}$求解，便是**原论文的更新方程**：
  
    $$\mathcal{P}_{k+1}=\mathcal{P}_k-(\boldsymbol{J}(\mathcal{P})^T\boldsymbol{J}(\mathcal{P}))^{-1}\boldsymbol{J}(\mathcal{P})^TF(\mathcal{P})$$
  
  如果我们能够顺利解出该方程，那么 Gauss-Newton 的算法步骤可以写成：
  
  1. 给定某个初始值$\mathcal{P}_0$，这里可以使用由稀疏特征点PNP或者使用learning-based方法获得的初值。
    2. 对于第 k 次迭代，求出当前的雅可比举证$\boldsymbol{J}(\mathcal{P}_k)$和误差$F(\mathcal{P})$。
  3. 求解增量方程：$\boldsymbol{H}(\mathcal{P})\Delta\mathcal{P}=\boldsymbol{g}$。
    4. 若$\Delta \mathcal{P}$足够小，则停止。否则，令$\mathcal{P}_{k+1}=\mathcal{P}_k+\Delta \mathcal{P}_k$，返回2。

    从算法步骤中可以看到，增量方程的求解占据着主要地位。原则上，它要求我们所用的近似$\boldsymbol{H}$矩阵是可逆的（而且是正定的），但实际数据中计算得到的$\boldsymbol{J}^T\boldsymbol{J}$却**只有半正定性**。也就是说，在使用 Gauss Newton 方法时，可能出现$\boldsymbol{J}^T\boldsymbol{J}$为奇异矩阵或者病态 (ill-condition) 的情况，此时增量的稳定性较差，导致算法不收敛。更严重的是，就算我们假设 H 非奇异也非病态，如果我们求出来的步长$\Delta\mathcal{P}$太大，也会导致我们采用的局部近似不够准确，这样一来我们甚至都无法保证它的迭代收敛，哪怕是让目标函数变得更大都是有可能的。

    尽管 Gauss Newton 法有这些缺点，但是它依然值得我们去学习，因为在非线性优化里，相当多的算法都可以归结为 Gauss Newton 法的变种。这些算法都借助了 Gauss Newton 法的思想并且通过自己的改进修正 Gauss Newton 法的缺点。例如一些**线搜索方法 (line search method)**，这类改进就是加入了一个标量$\alpha$，在确定了$\Delta\mathcal{P}$进一步找到$\alpha$使得$\|F(\mathcal{P}+\alpha\Delta\mathcal{P})\|^2$达到最小，而不是像 Gauss Newton 法那样简单地令$\alpha=1$，接下来说的方法就属于这一类。

  - Levenberg-Marquadt

    由于 Gauss-Newton 方法中采用的近似二阶泰勒展开只能在展开点附近有较好的近似效果，所以我们很自然地想到应该给$\mathcal{P}$添加一个**信赖区域（Trust Region）**，不能让它太大而使得近似不准确。非线性优化中有一系列这类方法，这类方法也被称之为信赖区域方法 (Trust Region Method)。在信赖区域里边，我们认为近似是有效的；出了这个区域，近似可能会出问题。

    那么如何确定这个信赖区域的范围呢？一个比较好的方法是**根据我们的近似模型跟实际函数之间的差异**，也就是计算一阶泰勒展开式$F\left(\mathcal{P}\right)+\boldsymbol{J}\left(\mathcal{P}\right)\Delta\mathcal{P}$和原式$F(\mathcal{P}+\Delta\mathcal{P})$的差异来确定这个范围：如果差异小，我们就让范围尽可能大；如果差异大，我们就缩小这个近似范围。因此，考虑使用：

    $$\rho=\frac{F\left(\mathcal{P}+\Delta\mathcal{P}\right)-F\left(\mathcal{P}\right)}{\boldsymbol{J}\left(\mathcal{P}\right)\Delta \mathcal{P}}$$

    来判断泰勒近似是否够好。$\rho$的**分子是实际函数下降的值**，**分母是近似模型下降的值**。如果$\rho$接近于 1，则近似是好的。如果$\rho$太小，说明实际减小的值远少于近似减小的值，则认为近似比较差，需要缩小近似范围。反之，如果$\rho$比较大，则说明实际下降的比预计的更大，我们可以放大近似范围。

    于是，我们构建一个改良版的非线性优化框架，该框架会比 Gauss Newton 有更好的效果：

    1. 给定某个初始值$\mathcal{P}_0$，以及初始优化半径$\mu$。

    2. 对于第 k 次迭代，求解：

       $$\min_{\Delta\mathcal{P}_k}\frac{1}{2}\|F\left(\mathcal{P}_k\right)+\boldsymbol{J}\left(\mathcal{P}_k\right)\Delta\mathcal{P}_k\|^2,\quad s.t.\|\boldsymbol{D}\Delta\mathcal{P}_k\|^2\leq\mu$$

       这$\mu$是信赖区域的半径，$\boldsymbol{D}$将在后文说明

    3. 计算$\rho$

    4. 若$\rho>\frac{3}{4}$，则$\mu=2\mu$
  
    5. 若$\rho<\frac{1}{4}$，则$\mu=0.5\mu$
  
    6. 如果$\rho$大于某阈值，认为近似可行。令$\mathcal{P}_{k+1}=\mathcal{P}_k+\Delta\mathcal{P}_k$
  
    7. 若$\Delta \mathcal{P}$足够小，则停止。否则返回2。
    
    这里近似范围扩大的倍数和阈值都是经验值，可以替换成别的数值。在$\|\boldsymbol{D}\Delta\mathcal{P}_k\|^2\leq\mu$中，我们**把增量限定于一个半径为$\mu$的球中**，认为只在这个球内才是有效的。带上$\boldsymbol{D}$之后，这个球可以看成一个椭球。在 Levenberg 提出的优化方法中，把$\boldsymbol{D}$取成单位阵$\boldsymbol{I}$，相当于直接把$\Delta\mathcal{P}$约束在一个球中。随后，Marqaurdt 提出将$\boldsymbol{D}$取成非负数对角阵——实际中通常用$\boldsymbol{J}^T\boldsymbol{J}$的对角元素平方根，使得在梯度小的维度上约束范围更大一些——相当于使用一个椭球来约束增量，椭球的大小由梯度决定（这有点像3DGS，只不过协方差改成了梯度$\boldsymbol{J}^T\boldsymbol{J}$）。不论如何，在 L-M 优化中，我们都需要解$\|\boldsymbol{D}\Delta\mathcal{P}_k\|^2\leq\mu$来获得梯度。这个子问题是**带不等式约束的优化问题，这类问题会遇到解空间不连续的问题，难以优化**，常用的做法是将这种$\leq$**必须满足**的**硬约束**问题转化为$\min$**使其最小**这种**软约束**问题，我们用 Lagrange 乘子将它转化：
    
    $$\min_{\Delta\mathcal{P}_k}\frac{1}{2}\|F\left(\mathcal{P}_k\right)+\boldsymbol{J}\left(\mathcal{P}_k\right)\Delta\mathcal{P}_k\|^2+\frac{\lambda}{2}\left\|\boldsymbol{D}\Delta\mathcal{P}_k\right\|^2.$$
    
    这里$\lambda$为 Lagrange 乘子。类似于 Gauss-Newton 中的做法——展开后求导，令导数为零，得到：
    
    $$\left(\boldsymbol{H}(\mathcal{P})+\lambda\boldsymbol{D}^T\boldsymbol{D}\right)\Delta \mathcal{P}=\boldsymbol{g}$$
    
    可以看到，增量方程相比于 Gauss-Newton，多了一项$\lambda\boldsymbol{D}^T\boldsymbol{D}$。
    
    我们看到，当参数$\lambda$比较小时，$\boldsymbol{H}$占主要地位，这说明二次近似模型在该范围内是比较好的，L-M 方法更接近于 G-N 法。另一方面，当$\lambda$比较大时，$\lambda\boldsymbol{D}^T\boldsymbol{D}$占据主要地位，L-M更接近于一阶梯度下降法（即最速下降），这说明附近的二次近似不够好。L-M 的求解方式，可**在一定程度上避免线性方程组的系数矩阵的非奇异和病态问题，提供更稳定更准确的增量$\Delta\mathcal{P}$**。
    
    将$\Delta\mathcal{P}$求解并带入，可得到原论文中给出的更新方程：
    
    $$\mathcal{P}_{k+1}=\mathcal{P}_k-(\boldsymbol{J}(\mathcal{P}_k)^T\boldsymbol{J}(\mathcal{P}_k)+\lambda\boldsymbol{D}^T\boldsymbol{D})^{-1}\boldsymbol{J}(\mathcal{P}_k)^TF(\mathcal{P}_k)$$

另一种度量使用$\ell_{2,1}\mathrm{-norm~}(\Psi(\mathbf{x})=||\mathbf{x}||_2^1)$，这种方法可以看做先按照$L_2$范数对每个元素求平方，再将每行所有元素加起来，得到一个列向量，这样**将每一行的元素都关联起来**，优化的目标更**倾向于将一整行的数值都变为0，也就是行稀疏**：

$$||\mathbf{r}(\mathcal{P},\mathbf{p})||_2^1=\underbrace{(||\mathbf{r}(\mathcal{P},\mathbf{p})||_2^1)^{-1}}_{constant}\cdot\underbrace{||\mathbf{r}(\mathcal{P},\mathbf{p})||_2^2}_{least-squares}$$

其中第一部分在迭代过程中视为常数，可以视作第二部分的权重，第二部分就是一个非线性最小二乘问题，用上面提到的方法求解。这种方法称为**Iteratively Reweighted Least Squares (IRLS) method**，翻译过来也很好理解，就是在每次迭代中**使用$L_{2,1}$范数对$L_{2,2}$范数进行赋权**。

### 基于机器学习的重建稠密人脸

对比基于优化的方法能够拟合出人脸稀疏顶点的参数（位置，颜色），基于学习的方法能够直接重建出稠密人脸。有工作通过学习视频序列，也有通过单张图片获得人脸。大多数方法使用**随机森林与卷积神经网络**来学习人脸模型参数或直接学习人脸结构。但作者表示本篇论文主要注重于基于优化的方法，基于学习的方法得到的结果非常鼓舞人心，我们也希望未来能有更惊艳的基于学习的工作。基于学习的数据集通常采用重建出的人脸模型、完全基于生成模型的图片或是这两种的混合，有些也会直接使用采集的图片而不作特殊处理；表面细化也被作为一种学习任务，从粗糙的表面和图片回归出更加精细的细节；还有方法采用端到端的方式直接估计稠密人脸表面，反射率和场景中的光照；还有不估计模型参数，而是估计每个像素的深度，这种方法不用像之前的人脸模型一样注意解的约束空间，可以得到更细节的表面；

尽管基于学习、回归的方法有更快地速度，但基于优化的方法能获得更高的重建质量，并且不受数据集偏置的影响。在未来，我们希望能看到更多学习和优化结合的方法。