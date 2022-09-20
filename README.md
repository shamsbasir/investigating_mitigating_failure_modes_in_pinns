### Investigating and Mitigating Failure Modes in Physics-informed Neural Networks(PINNs)


### Abstract:
In this paper, we demonstrate and investigate several challenges that stand in the way of tackling complex problems using physics-informed neural networks. In particular, we visualize the loss landscapes of trained models and perform sensitivity analysis of backpropagated gradients in the presence of physics. Our findings suggest that existing methods produce highly non-convex loss landscapes that are difficult to navigate. Furthermore, high-order PDEs contaminate the backpropagated gradients that may impede or prevent convergence. We then propose a novel method that bypasses the calculation of high-order PDE operators and mitigates the contamination of backpropagating gradients. In doing so, we reduce the dimension of the search space of our solution and facilitate learning problems with non-smooth solutions. Our formulation also provides a feedback mechanism that helps our model adaptively focus on complex regions of the domain that are difficult to learn. We then formulate an unconstrained dual problem by adapting the Lagrange multiplier method. We apply our method to solve several challenging benchmark problems governed by linear and non-linear PDEs. 


##### Note: All the codes will be published after the peer-review process
