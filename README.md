# Code For Reproducing Paper "Adaptive Momentum and Nonlinear Damping in Neural Network Training".

This repository contains code for the paper [Adaptive Momentum and Nonlinear Damping in Neural Network Training](link.com). Each folder contains one of six experiments shown in the paper. `optimizers/` contains implementations of our optimizers CD, iKFAD, and CADAM in addition to our custom implementation of LDHD. `scripts/` contains scripts for some experiments to run the best hyperparameter configuration experiments over a few seeds. It has scripts for all except `nano/` and `owt/` which have scripts within those folders for sweeping and running best hyperparameter configurations. Within each folder, there is an environment.yml file to set up conda environments for the associated experiment.


### Cite
```
@misc{smallbatch,
  title={Adaptive Momentum and Nonlinear Damping for Neural Network Training}, 
  author={Katerina Karoni and Rajit Rajpal and Benedict Leimkuhler and Gabriel Stoltz},
  year={2026},
  eprint={...},
  archivePrefix={arXiv},
  primaryClass={...}
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
