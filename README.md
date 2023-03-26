### Investigating and Mitigating Failure Modes in Physics-informed Neural Networks(PINNs)


### Abstract:
This paper explores the difficulties in solving partial differential equations(PDEs) using physics-informed neural networks (PINNs). PINNs use physics as a reg-ularization term in the objective function. However, a drawback of this approach is therequirement for manual hyperparameter tuning, making it impractical in the absenceof validation data or prior knowledge of the solution. Our investigations of the losslandscapes and backpropagated gradients in the presence of physics reveal that exist-ing methods produce non-convex loss landscapes that are hard to navigate. Our find-ings demonstrate that high-order PDEs contaminate backpropagated gradients andhinder convergence. To address these challenges, we introduce a novel method thatbypasses the calculation of high-order derivative operators and mitigates the contam-ination of backpropagated gradients. Consequently, we reduce the dimension of thesearch space and make learning PDEs with non-smooth solutions feasible. Our methodalso provides a mechanism to focus on complex regions of the domain. Besides, wepresent a dual unconstrained formulation based on Lagrange multiplier method to en-force equality constraints on the model’s prediction, with adaptive and independentlearning rates inspired by adaptive subgradient methods. We apply our approach tosolve various linear and non-linear PDEs. 



##### Note: All the codes will be published after the peer-review process


## Citation
Please cite us if you find our work useful for your research:
```
@article{basir2022investigating,
  title={Investigating and Mitigating Failure Modes in Physics-informed Neural Networks (PINNs)},
  author={Basir, Shamsulhaq},
  journal={arXiv preprint arXiv:2209.09988},
  year={2022}
}
```
