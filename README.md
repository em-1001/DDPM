# Stable Diffusion
## VAE
### Maximum Likelihood
VAE(Variational Autoencoders)는 Generative model로 Autoencoders와는 반대로 Decoder부분을 학습시키기 위해 만들어졌다. 
MLE(Maximum Likelihood Estimation)관점에서의 모델의 학습에 대해 먼저 설명하면 input $z$와 target $x$가 있을 때, $f_{\theta}(\cdot)$ 은 모델의 종류가 되고, 최종 목표는 정해진 확률분포에서 target이 나올 확률인 $p(x | f_{\theta}(z))$가 최대가 되도록 하는 것이다. 따라서 MLE에서는 학습전에 확률분포(가우시안, 베르누이.. )를 먼저 정하게 되고, 모델의 출력은 이 확률 분포를 정하기 위한 파라미터(가우시안의 경우 $\mu, \sigma^2$)라고 해석할 수 있다. 결과적으로 target을 잘 생성하는 모델 파라미터 $\theta$는 $\theta^* = \underset{\theta}{\arg\min} [-\log(p(x | f_{\theta}(z)))]$가 된다. 이렇게 찾은 $\theta^*$는 확률분포를 찾은 것이므로 결과에 대한 sampling이 가능하고, 이 sampling에 따라 다양한 이미지가 생성될 수 있는 것이다.

VAE의 Decoder도 위와 비슷하다. Encoder를 통해 sampling된 데이터 $z$ (Latent Variable)가 있고 Generator $g_{\theta}(\cdot)$와 Target $x$가 있을 때, training data에 있는 $x$가 나올 확률을 구하는 것을 목적으로 한다. 이때 $z$는 controller로서 생성될 이미지를 조정하는 역할을 할 수 있다. 예를 들면 고양이의 귀여움을 조정하여 더 귀여운 고양이 이미지를 생성하는 것이다.

다시 돌아와서 결과적으로 VAE의 목적은 모든 training data $x$에 대해 $x$가 나올 확률 $p(x)$를 구하는 것이 목적이다. 이때 training data에 있는 sample과 유사한 sample을 생성하기 위해서 prior 값을 이용하는데, 이 값이 Latent Variable인 $z$가 나올 확률 $p(z)$이고, $p(x)$는 $\int p(x | g_{\theta}(z))p(z) dz = p(x)$로 구해진다. 


### Prior Distribution 

<p align="center"><img src="https://github.com/em-1001/Stable-Diffusion/assets/80628552/2a1ac824-6398-43df-b68a-504785def59d"></p>

앞서 말했듯이 $z$는 controller 역할을 하기 때문에 $z$를 잘 조정할 수 있어야 한다. 이때 $z$는 고차원 input에 대한 manifold 상에서의 값들인데, generator의 input으로 들어가기 위해 sampling된 값이 이 manifold 공간을 잘 대표하는가? 라는 질문이 나온다. 이에 대한 답은 잘 대표한다는 것이다. 위 이미지의 예시처럼 normally-distributed 된 왼쪽에 $g(z) = \frac{z}{10} + \frac{z}{||z||}$를 적용하면 오른쪽의 ring 형태가 나오는걸 확인할 수 있다. 이처럼 간단한 변형으로 manifold를 대표할 수 있기 때문에 모델이 DNN 이라면, 학습해야 하는 manifold가 복잡하다 하더라도, DNN의 한 두개의 layer가 manifold를 찾기위한 역할로 사용될 수 있다. 따라서 Prior Distribution을 normal distribution과 같은 간단한 distribution으로 해도 상관없다.  


### Variational Inference
$p(x | g_{\theta}(z))$의 likelihood가 최대가 되는 것이 목표라면 Maximum Likelihood Estimation를 직접적으로 사용해서 구하면 될거 같은데 실제론 그렇지 않다. 그 이유는 가우시안 분포라 가정했을 때, $p(x | g_{\theta}(z))$의 log loss인 $-\log(p(x | g_{\theta}(z)))$는 Mean Squared Error와 같아진다. 즉, MSE의 관점에서 loss가 작은 것이 $p(x)$에 더 크게 관여하는데, MSE loss가 작은 이미지가 실제 의미적으로 더 가까운 이미지가 아닌 경우가 많기 때문에 올바른 방향으로 학습할 수가 없다. 

$$||x - z_{bad}||^2 < ||x - z_{good}||^2 \to p(x | g_{\theta}(z_{bad})) > p(x | g_{\theta}(z_{good}))$$

예를 들면 원래 고양이 이미지에서 일부분이 조금 잘린 이미지 $a$와 한 pixel씩 옆으로 이동한 이미지 $b$가 있다고 하면 $b$는 pixel만 옆으로 밀렸을 뿐 고양이 그대로 이지만 $a$는 이미지가 잘렸기 때문에 의미적으론 $b$가 $a$보다 고양이에 가까운데, MSE 관점에서는 $b$의 loss가 더 크게 나오게 된다. 

이러한 문제를 해결하기 위해 Variational Inference가 나오게 된다. 기존 prior에서 sampling을 하니 학습이 잘 안되니까 $z$를 prior에서 sampling하지 말고 target인 $x$와 유사한 sample이 나올 수 있는 이상적인 확률분포 $p(z|x)$로 부터 sampling한다. 이때 우리는 
$p(z|x)$가 무엇인지 알지 못하기 때문에, 이미 알고 있는 확률 분포(가우시안..) $q_{\phi}(z|x)$를 임의로 택하고 그것의 파라미터 $\phi$를 조정하여 $p(z|x)$와 유사하게 되도록 하는 것이다. 그렇게 이상적인 확률분포에 근사된 $q_{\phi}$를 통해서 $z$를 sampling하게 된다. $p(z|x) \approx q_{\phi}(z|x) \sim z$


### ELBO
지금까지의 내용을 정리하면 우리가 구하고자 하는 것은 $p(x)$였고, 이를 위해 Prior Distribution을 사용했으며, 그냥 prior에서 sampling하려니 잘 학습이 안되서 이상적인 확률분포 $p(z|x)$ 를 근사한 $q_{\phi}$를 사용하게 됐다. 이 4개간의 관계식에서 loss를 유도하는 과정에 우리가 찾아야 하는 ELBO(Evidence LowerBOund)가 나오게 된다. 

우선 $\log(p(x))$에서 시작해서 ELBO를 유도하는 과정을 정리하면 아래와 같다. 
 
$$\begin{aligned}
\log(p(x)) &= \int \log(p(x))q_{\phi}(z|x)dz 　 \leftarrow \int q_{\phi}(z|x)dz = 1 \\ 
&=\int \log\left(\frac{p(x, z)}{p(z|x)}\right)q_{\phi}(z|x)dz 　 \leftarrow p(x) = \frac{p(x, z)}{p(z|x)} \\
&=\int \log\left(\frac{p(x, z)}{q_{\phi}(z|x)}\cdot\frac{q_{\phi}(z|x)}{p(z|x)}\right)q_{\phi}(z|x)dz \\ 
&=\int \log\left(\frac{p(x, z)}{q_{\phi}(z|x)}\right)q_{\phi}(z|x)dz + \int \log\left(\frac{q_{\phi}(z|x)}{p(z|x)}\right)q_{\phi}(z|x)dz \\ 
\\ 
&　　　 　 　\color{red}ELBO(\phi)　　　　 　　\color{yellowgreen}KL\left(q_{\phi}(z|x) \ || \ p(z|x)\right)
\end{aligned}$$

여기서 $KL\left(q_{\phi}(z|x) \ || \ p(z|x)\right)$ term은 Kullback–Leibler divergence로 두 확률분포 간의 거리($\ge 0$)를 구한다.
우리가 원하는 건 $q_{\phi}(z|x)$가 $p(z|x)$에 최대한 가까워 져야 하므로 $KL$을 최소화 하는 $q_{\phi}(z|x)$의 $\phi$를 찾아야 하는데 $p(z|x)$를 모르기 때문에 KL을 최소화 하는 대신 $ELBO$를 최대화 하는 $\phi$를 찾으면 된다. 

$$\log(p(x)) = ELBO(\phi) + KL(\left(q_{\phi}(z|x) \ || \ p(z|x)\right)$$ 

$$q_{\phi^*}(z|x) = \underset{\phi}{\arg\max} \ ELBO(\phi)$$

$ELBO$를 최대화 하기 위해 $ELBO$ term을 다시 전개하면 다음과 같다. 

$$\begin{aligned}
ELBO(\phi) &= \int \log \left(\frac{p(x, z)}{q_{\phi}(z|x)}\right)q_{\phi}(z|x)dz \\
&= \int \log \left(\frac{p(x|z)p(z)}{q_{\phi}(z|x)}\right)q_{\phi}(z|x)dz \\  
&= \int \log \left(p(x|z)\right)q_{\phi}(z|x)dz - \int \log \left(\frac{q_{\phi}(z|x)}{p(z)}\right)q_{\phi}(z|x)dz \\ 
&= \mathbb{E}_ {q_{\phi}(z|x)} \left[\log\left(p(x|z)\right)\right] - KL\left(q_{\phi}(z|x) \ || \ p(z)\right)
\end{aligned}$$

그래서 정리하면 $ELBO$를 최대화 하는데 있어서 $\phi$, $\theta$로 총 2개에 대한 Optimization Problem을 해결해야 하고 이는 아래와 같다. 

**Optimization Problem 1 on $\phi$: Variational Inference**

$$\log(p(x)) \ge \mathbb{E}_ {q_{\phi}(z|x)} \left[\log\left(p(x|z)\right)\right] - KL\left(q_{\phi}(z|x) \ || \ p(z)\right) = ELBO(\phi)$$

첫 번째로 $ELBO$를 maximize하는 $\phi$를 찾는데, 이 과정이 이상적인 sampling function을 찾는 것이다.    
$\mathbb{E}_ {q_{\phi}(z|x)} \left[\log\left(p(x|z)\right)\right]$는 $q_{\phi}$에서 sampling한 $z$에 대한 $\log\left(p(x|z)\right)$를 의미한다.  

**Optimization Problem 2 on $\theta$: Maximum likelihood**

$$-\sum_i \log(p(x_i)) \le -\sum_i \lbrace\mathbb{E}_ {q_{\phi}(z|x_i)} \left[\log\left( p \left(x_i|g_{\theta}(z)\right)\right)\right] - KL\left(q_{\phi}(z|x_i) \ || \ p(z)\right)\rbrace$$

두 번째는 이상적인 sampling function을 찾았으므로 여기서 $z$를 sampling해서 $z$로 부터 target $x$가 나오게 하는 Conditional probability $p(x | g_{\theta}(z))$가 최대가 되도록 하는 확률분포를 찾는 것이다.    

**Final Optimization Problem**

$$\underset{\phi, \theta}{\arg\min} \sum_i \mathbb{E}_ {q_{\phi}(z|x_i)} \left[\log\left( p \left(x_i|g_{\theta}(z)\right)\right)\right] - KL\left(q_{\phi}(z|x_i) \ || \ p(z)\right)$$

결국 위 두 Optimization Problem을 종합하는 식이 위와 같고, 이 식이 $ELBO$를 최대화하는 것과 같게 된다. 

정리해서 이상적으로 sampling을 해주는 $q_{\phi}(z|x)$를 Encoder, Posterior, Inference Network 등으로 부르고, sampling된 $z$로 부터 이미지를 generate 해주는 $g_{\theta}(x|z)$를 Decoder, Generator, Generation Network 등으로 부른다. 

### Loss Function 
결과적으로 ELBO를 최대화 하기 위한 Loss는 아래와 같이 표현된다. 

$$L_i(\phi, \theta, x_i) = -\mathbb{E}_ {q_{\phi}(z|x_i)} \left[\log\left( p \left(x_i|g_{\theta}(z)\right)\right)\right] + KL\left(q_{\phi}(z|x_i) \ || \ p(z)\right)$$ 

Reconstruction  Error : $-\mathbb{E}_ {q_{\phi}(z|x_i)} \left[\log\left( p \left(x_i|g_{\theta}(z)\right)\right)\right]$  
Regularization : $KL\left(q_{\phi}(z|x_i) \ || \ p(z)\right)$  

Reconstruction  Error term은 앞서 말했던 MSE(가우시안의 경우)로 계산되는 부분이다. 해당 term을 결론적으로 보면 $x$를 넣었을 때 $x$가 나올 확률에 대한 것이기 때문에 Reconstruction  Error라고 한다. 만약 가정을 베르누이 분포라고 하면 MSE가 아닌 Cross Entropy가 된다. 

Regularization는 같은 Reconstruction  Error를 갖는 $q_{\phi}$가 여럿 있다면, 그 중에서도 prior $p(z)$와 가까운 $q_{\phi}$를 고르라는 것으로, 생성 데이터에 대한 통제 조건을 prior에 부여하고, 이와 유사해야 한다는 조건을 부여한 것이다. 

그럼 이제 $ELBO$를 실제 어떻게 계산하는지 알아보기 전에 $q_{\phi}$를 gaussian distribution $q_{\phi} \sim N(\mu_i, \sigma_i^2I)$, $p(z)$를 normal distribution $p(z) \sim N(0, 1)$으로 가정한다고 하자. 

우선 Regularization term의 경우 2개의 가우시안 분포 간의 KL divergence가 아래와 같이 계산된다고 수학적으로 알려져있다. 

$$D_{KL}(\mathcal{N}_0 \ || \ \mathcal{N}_1) = \frac{1}{2}\left[ tr\left(\sum_1^{-1}\sum_0\right) + (\mu_1 - \mu_0)^T \sum_1^{-1}(\mu_1 - \mu_0) - k + \ln\frac{|\sum_1|}{|\sum_0|}\right]$$

이에 따라 앞서 가정한대로 KL term을 계산하면 아래와 같이 된다.

$$KL\left(q_{\phi}(z|x_i) \ || \ p(z)\right) = \frac{1}{2}\sum_{j=1}^J\left(\mu_{i,j}^2 + \sigma_{i,j}^2 - \ln(\sigma_{i,j}^2) - 1\right)$$

$J$ : dimension  
$\mu, \sigma$ : $q_{\phi} \sim N(\mu_i, \sigma_i^2I)$

Reconstruction  Error term의 경우 원래라면 아래처럼 기댓값을 구할 때 적분을 해야하지만, 대신 Monte Carlo method로 $L$개를 sampling하여 구한다. 

$$\begin{aligned}
\mathbb{E}_ {q_{\phi}(z|x_i)} \left[\log\left( p \left(x_i|g_{\theta}(z)\right)\right)\right] &= \int \log(p_{\theta}(x_i|z))q_{\phi}(z|x_i)dz \\ 
&\approx \frac{1}{L}\sum_{z^{i,l}} \log\left(p_{\theta}(x_i|z^{i,l})\right)
\end{aligned}$$

이때 문제는 $q_{\phi}$에서 random으로 sampling을 하기 때문에 backpropagation에서의 편미분이 불가능하다는 것이다. 그래서 이를 해결하기 위해 reparameterization trick을 사용하는데, 이는 gaussian distribution이 gaussian distribution대신 normal distribution에서 sampling한 $\epsilon$에 대해 아래처럼 표현될 수 있다는 점을 이용하여 backpropagation이 가능하도록 한 것이다. 

$$z^{i,l} \sim \mathcal{N}(\mu_i, \sigma_i^2I) 　\to　 z^{i,l} = \mu_i + \sigma_i^2 \odot \epsilon 　　\epsilon \sim \mathcal{N}(0,1)$$

## DDPM

### 통계학
https://angeloyeo.github.io/2020/01/09/Bayes_rule.html

https://angeloyeo.github.io/2020/09/14/normal_distribution_derivation.html -> exponential 내부의 식 유도전까지 공부함

몬테카를로 시뮬레이션으로 배우는 확률통계 with 파이썬 : https://www.yes24.com/Product/Goods/117709828

#### MLE
https://modernflow.tistory.com/67  
https://everyday-tech.tistory.com/entry/%EC%B5%9C%EB%8C%80-%EC%9A%B0%EB%8F%84-%EC%B6%94%EC%A0%95%EB%B2%95Maximum-Likelihood-Estimation

#### 나이브 베이즈 분류기 
https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98

## Stable-Diffusion 영상  
ELBO 식 : https://yonghyuc.wordpress.com/2019/09/26/elbo-evidence-of-lower-bound/  
ELBO 식에서 $\log p(x)$가 빠져나오는 이유는 $\int q(z) = 1$이기 때문인듯. 

VAE : https://huidea.tistory.com/296  , https://deepinsight.tistory.com/127 이걸로 낼 공부 ㄱㄱ

레전드 강의2 -> 33:10


## 목표 
stable-diffusion from scratch 영상을 따라 코딩해보고(직접 구현할 수 있는건 직접 구현), 원하는 이미지를 구해서 커스텀 학습시켜보기 
VAE -> CLIP -> U-Net -> DDPM 순으로 직접 구현해보기 

### VAE 구현
https://avandekleut.github.io/vae/

언데드 언럭(ㅋㄹ ㅈㅌㅍ

# Reference
## Paper
Tutorial on Variational Autoencoders : https://arxiv.org/pdf/1606.05908.pdf  
