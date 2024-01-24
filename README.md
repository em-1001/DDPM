# Stable Diffusion
## VAE
VAE(Variational Autoencoders)는 Generative model로 Autoencoders와는 반대로 Decoder부분을 학습시키기 위해 만들어졌다. 
MLE(Maximum Likelihood Estimation)관점에서의 모델의 학습에 대해 먼저 설명하면 input $z$와 target $x$가 있을 때, $f_{\theta}(\cdot)$ 은 모델의 종류가 되고, 최종 목표는 정해진 확률분포에서 target이 나올 확률인 $p(x | f_{\theta}(z))$가 최대가 되도록 하는 것이다. 따라서 MLE에서는 학습전에 확률분포(가우시안, 베르누이 등)를 먼저 정하게 되고, 모델의 출력은 이 확률 분포를 정하기 위한 파라미터(가우시안의 경우 $\mu, \sigma$)라고 해석할 수 있다. 

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
