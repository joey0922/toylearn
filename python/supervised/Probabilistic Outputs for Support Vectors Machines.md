# Probabilistic Outputs for Support Vectors Machines

Platt(2000) proposes approximating the posterior class possibility Pr(y=1|x) by a sigmoid function:
$$
Pr(y=1|x) \approx P_{A,B}(f) \equiv \frac{1}{1+exp(Af + B)}
$$
where $f = f(x)$, that the binary Support Vector Machine(SVM) computes $sign(f(x))$ to predict the label of any example x. Let each $f_i$ be an estimate of $f(x_i)$. The best parameters settings $z^* = (A^*, B^*)$ is determined by solving the following regularized maximum likelihood problem(with $N_+$ of $y_i$ 's positive, and $N_-$ negative):
$$
\min_{z=(A, B)} F(z) = -\Sigma_{i=1}^l [t_ilogp_i + (1-t_i)log(1-p_i)]
$$
for $p_i=P_{A,B}(f_i)$ , and $t_i = \{ \begin{array} \\ \Large\frac{N_++1}{N_++2} & \mbox{if} & y_i=+1\\ \Large\frac{1}{N_-+2} & \mbox{if} & y_i=-1 \end{array}, i = 1, \cdots, l.$ 

# Reasoning

Obviously, the likelihood formula is similar to logistic regression's likelihood function(cross-entropy loss).  It is a convex problem to get parameters A, B of F(z). Define objective function as below:
$$
L(A, B) = -\Sigma_{i=1}^l [t_ilogp_i + (1-t_i)log(1-p_i)]
$$
We can deform the formula above:
$$
L(A, B) = \Sigma_{i=1}^l [(t_i-1)(Af_i+B)+log(1+exp(Af_i+B))]
$$
It is easily to use gradient descent to solve this optimization problem:
$$
\begin{align} \\ \nabla_A L(A, B) &= \Sigma_{i=1}^l[(t_i-1)f_i+\frac{1}{1+exp(Af_i+B)}*exp(Af_i+B)*f_i] \\ &=\Sigma_{i=1}^l[((t_i-1)f_i)+(1-p_i)f_i] \\ &= \Sigma_{i=1}^l(t_i-p_i)f_i \\ \end{align}
$$
Similarly, we can calculate$\nabla_B$​ :
$$
\nabla_BL(A,B)=\Sigma_{i=1}^l(t_i-p_i)
$$
Thus, update $z^*=(A^*,B^*)$ by gradient decent during iterations:
$$
A = A - \nabla_A \\ B=B-\nabla_B
$$
The two papers below proposes two different ways to optimize parameters A and B. Will talk about them in the future.

# References

- Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods
- A Note on Platt’s Probabilistic Outputs for Support Vector Machines
