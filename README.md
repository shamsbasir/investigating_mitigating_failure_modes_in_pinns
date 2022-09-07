### Investigating and Mitigating Failures in Physics-informed Neural Networks


### Abstract:
Physics-informed neural networks (PINNs) are trained on a weighted sum of the residuals of a governing partial differential equation (PDE) and its boundary conditions. These weights that are not known a priori are problem specific and significantly impact model predictions. Recent developments gave rise to physics-and-equality-constrained artificial neural networks (PECANNs) that properly constrain the boundary conditions using the augmented Lagrangian method (ALM). While the above approaches offer great promise, several challenges stand in the way of complex applications. First, the strong form of PDEs might be too stringent for problems with non-smooth solutions. Second, high-order PDEs pollute backpropagated gradients and produce loss landscapes that are complex to navigate. Third, finding an optimal update strategy as well as a proper maximum value for the penalty parameter in ALM is challenging. Fourth, the above approaches may require many collocation points to converge. Because models do not have any feedback mechanism to attend to difficult regions of the domain during training. To characterize these challenges, we analyze the loss landscapes of our trained models and investigate the impact of differential operators on polluting the backpropagated gradients. We then propose a method that avoids taking high-order derivatives and mitigates backpropagated gradient contamination. In doing so, we reduce the dimension of the search space of the solution and facilitate learning problems with non-smooth solutions. Our formulation also provides a feedback mechanism that helps our model adaptively focus on complex regions of the domain that are difficult to learn. Finally, we formulate a dual optimization problem by adapting the Lagrange multiplier method to accelerate the convergence of Lagrange multipliers. Our numerical experiments demonstrate that our approach obtains state-of-the-art results on several challenging benchmark problems.


##### Note: All the codes will be published after the peer-review process
