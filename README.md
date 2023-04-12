### Investigating and Mitigating Failure Modes in Physics-informed Neural Networks(PINNs)


### Abstract:
This paper explores the difficulties in solving partial differential equations (PDEs) using physics-informed neural networks (PINNs). PINNs use physics as a regularization term in the objective function. However, a drawback of this approach is the requirement for manual hyperparameter tuning, making it impractical in the absence of validation data or prior knowledge of the solution. Our investigations of the loss landscapes and backpropagated gradients in the presence of physics reveal that existing methods produce non-convex loss landscapes that are hard to navigate. Our findings demonstrate that high-order PDEs contaminate backpropagated gradients and hinder convergence. To address these challenges, we introduce a novel method that bypasses the calculation of high-order derivative operators and mitigates the contamination of backpropagated gradients. Consequently, we reduce the dimension of the search space and make learning PDEs with non-smooth solutions feasible. Our method also provides a mechanism to focus on complex regions of the domain. Besides, we present a dual unconstrained formulation based on Lagrange multiplier method to enforce equality constraints on the model's prediction, with adaptive and independent learning rates inspired by adaptive subgradient methods. We apply our approach to solve various linear and non-linear PDEs. 




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
##### Note: I am thinking of making a video explaining the codes for those who are new to the field.
##### The codes are in Jupyter notebook and self-containing. You can run them on google colab or on your own machine if you have Pytorch installed. I would like to mention that inputs to the models are normalized as below:
For example, you have a square domain with bottom left corner (-1,-1) and top right corner = (1,1) :
``` domain   = np.array([[-1,-1.0],[1.,1.]]). Generating collocation points from that domain will give you a mean of (0,0) and std (0.5773, 0.5773) that you can use to normalize your inputs. 
kwargs ={"mean":torch.tensor([[0.0, 0.0]]), "stdev":torch.tensor([[0.5773, 0.5773]])} 
You can use x_ = np.rand(1000000) * (x_max - x_min) + x_min and then x_.mean() and x_.std() to calculate those numbers 
```
