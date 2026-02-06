# Code For Reproducing Paper "Adaptive Momentum and Nonlinear Damping in Neural Network Training".

This repository contains code for the paper [Adaptive Momentum and Nonlinear Damping in Neural Network Training](https://arxiv.org/abs/2602.00334). 



**Adaptive Momentum and Nonlinear Damping for Neural Network Training**

> Aikaterini Karoni, Rajit Rajpal, Benedict Leimkuhler, Gabriel Stoltz<br>
>**Abstract:** We propose a continuous-time scheme for large-scale optimization that introduces individual, adaptive momentum coefficients regulated by the kinetic energy of each model parameter. This approach automatically adjusts to local landscape curvature to maintain stability without sacrificing convergence speed. We demonstrate that our adaptive friction can be related to cubic damping, a suppression mechanism from structural dynamics. Furthermore, we introduce two specific optimization schemes by augmenting the continuous dynamics of mSGD and Adam with a cubic damping term. Empirically, our methods demonstrate robustness and match or outperform Adam on training ViT, BERT, and GPT2 tasks where mSGD typically struggles. We further provide theoretical results establishing the exponential convergence of the proposed schemes.

Each folder contains one of six experiments shown in the paper. `optimizers/` contains implementations of our optimizers CD, iKFAD, and CADAM in addition to our custom implementation of LDHD. `scripts/` contains scripts for some experiments to run the best hyperparameter configuration experiments over a few seeds. It has scripts for all except `nano/` and `owt/` which have scripts within those folders for sweeping and running best hyperparameter configurations. Within each folder, there is an environment.yml file to set up conda environments for the associated experiment.

### Cite
```
@misc{karoni2026adaptivemomentumnonlineardamping,
      title={Adaptive Momentum and Nonlinear Damping for Neural Network Training}, 
      author={Aikaterini Karoni and Rajit Rajpal and Benedict Leimkuhler and Gabriel Stoltz},
      year={2026},
      eprint={2602.00334},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.00334}, 
}
```

### MIT License

```
Copyright (c) 2026 Rajit Rajpal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
