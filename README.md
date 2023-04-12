### Investigating and Mitigating Failure Modes in Physics-informed Neural Networks(PINNs)


### Abstract:
This paper explores the difficulties in solving partial differential equations (PDEs) using physics-informed neural networks (PINNs). PINNs use physics as a regularization term in the objective function. However, a drawback of this approach is the requirement for manual hyperparameter tuning, making it impractical in the absence of validation data or prior knowledge of the solution. Our investigations of the loss landscapes and backpropagated gradients in the presence of physics reveal that existing methods produce non-convex loss landscapes that are hard to navigate. Our findings demonstrate that high-order PDEs contaminate backpropagated gradients and hinder convergence. To address these challenges, we introduce a novel method that bypasses the calculation of high-order derivative operators and mitigates the contamination of backpropagated gradients. Consequently, we reduce the dimension of the search space and make learning PDEs with non-smooth solutions feasible. Our method also provides a mechanism to focus on complex regions of the domain. Besides, we present a dual unconstrained formulation based on Lagrange multiplier method to enforce equality constraints on the model's prediction, with adaptive and independent learning rates inspired by adaptive subgradient methods. We apply our approach to solve various linear and non-linear PDEs. 


#### Highlight: For our forward problems, we aim to minimize the loss on the PDE, subject to various equality constraints such as clean/noiseless boundary or initial conditions, flux, or any other constraint deemed necessary. By incorporating these constraints into our optimization framework, we obtain a more accurate solution that satisfies the physical laws governing the system. We solve the constrained optimization problem by formulating the unconstrained dual problem, which is equivalent to the original problem. This approach allows us to leverage the dual variables to enforce the equality constraints while minimizing the loss function.


## Citation
Please cite us if you find our work useful for your research:
##### 1) [Investigating and Mitigating Failure Modes in Physics-informed Neural Networks (PINNs)](https://arxiv.org/abs/2209.09988)
```
@article{basir2022investigating,
  title={Investigating and Mitigating Failure Modes in Physics-informed Neural Networks (PINNs)},
  author={Basir, Shamsulhaq},
  journal={arXiv preprint arXiv:2209.09988},
  year={2022}
}
```

##### 2) [Physics and Equality Constrained Artificial Neural Networks: Application to Forward and Inverse Problems with Multi-fidelity Data Fusion](https://doi.org/10.1016/j.jcp.2022.111301)
```
@article{PECANN_2022,
title = {Physics and Equality Constrained Artificial Neural Networks: Application to Forward and Inverse Problems with Multi-fidelity Data Fusion},
journal = {J. Comput. Phys.},
pages = {111301},
year = {2022},
issn = {0021-9991},
doi = {https://doi.org/10.1016/j.jcp.2022.111301},
url = {https://www.sciencedirect.com/science/article/pii/S0021999122003631},
author = {Shamsulhaq Basir and Inanc Senocak}
}
```

##### 3) [Critical Investigation of Failure Modes in Physics-informed Neural Networks](https://doi.org/10.2514/6.2022-2353)
```
@inbook{doi:10.2514/6.2022-2353,
author = {Shamsulhaq Basir and Inanc Senocak},
title = {Critical Investigation of Failure Modes in Physics-informed Neural Networks},
booktitle = {AIAA SCITECH 2022 Forum},
chapter = {},
pages = {},
doi = {10.2514/6.2022-2353},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2022-2353},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2022-2353},
}
```

##### Note: I am thinking of making a video explaining the codes for those who are new to the field.
##### The codes are in Jupyter notebook and self-containing. You can run them on google colab or on your own machine if you have Pytorch installed. I would like to mention that inputs to the models are normalized as follows:

For example, you have a square domain with bottom left corner (-1,-1) and top right corner = (1,1) :
``` Generating collocation points from that domain will give you a mean of (0,0) and std (0.5773, 0.5773) that  you can use to normalize your inputs
x_max = 1
x_min = -1
x_ = torch.rand(100000) * (x_max - x_min) + x_min
x_mean = x_.mean()
x_std  = x_.std()
----
domain   = np.array([[-1,-1.0],[1.,1.]])
kwargs   = {"mean":torch.tensor([[0.0, 0.0]]), "stdev":torch.tensor([[0.5773, 0.5773]])}  

```

### Funding Acknowledgment
This material is based upon work supported by the National Science Foundation under Grant No. 1953204 and in part in part by the University of Pittsburgh Center for Research Computing through the resources provided.

### Questions and feedback?
For questions or feedback feel free to reach us at [Shams Basir](mailto:shamsbasir@gmail.com) or [Linkedin](https://www.linkedin.com/in/shamsulhaqbasir/)
