# Linear-Regression-Analytic-and-Numerical-Methods-vs-Sklearn

# Linear Regression: Analytic and Numerical Methods vs Sklearn

Author: jvachier <br>
Creation date: May 2022  <br>
Publication date: May 2022 <br>

Goal: Salary prediction as function of the number of years of experience using univariate linear regression, from analytic and numerical methods (built using Python classes) compared to Sklearn (in-built). 


## Analytic Method <br>

The univariate (single feature) linear regression is given by<br>
$$
f(X) = \beta_0 + X \beta_1 \text{ or } \mathbf{\hat{y}} = \mathbf{X}_b \mathbf{\beta},
$$<br>
where $X$ is the feature (the number of years of experience), $\mathbf{X}_b$ the matrix $Nx2$ with a $1$ in the first position ($N$ the size of the output $\mathbf{y}$), and $\mathbf{\beta} = (\beta_0,\beta_1)^T$ are the parameters of the model that are learned.<br>
A popular method to determine the parameters $\mathbf{\beta}$ is to use the method of least squares. The parameters $\mathbf{\beta}$ are chosen such that they minimize the residual sum of squares
$$
RSS(\mathbf{\beta}) = \sum\limits_{i=1}^{N}(y_i - f(x_i))^2 \text{ or } RSS(\mathbf{\beta}) = (\mathbf{y}-\mathbf{X}_b \mathbf{\beta})^T(\mathbf{y}-\mathbf{X}_b \mathbf{\beta}).
$$
The first and second derivatives are given by
$$
\frac{\partial}{\partial \mathbf{\beta}} RSS = - 2 \mathbf{X}_b^T (\mathbf{y}-\mathbf{X}_b \mathbf{\beta}),
$$
and
$$
\frac{\partial^2}{\partial \mathbf{\beta}\partial \mathbf{\beta}^T} RSS = 2 \mathbf{X}_b^T \mathbf{X}_b \text{ with } \mathbf{X}_b^T \mathbf{X}_b > 0.
$$
Setting the first derivative to zero
$$
\mathbf{X}_b^T (\mathbf{y}-\mathbf{X}_b \mathbf{\beta}) = 0,
$$
gives the solution
$$
\mathbf{\beta} = (\mathbf{X}_b^T\mathbf{X}_b)^{-1}\mathbf{X}_b^T\mathbf{y}.
$$
The predicted values are 
$$
\mathbf{\hat{y}} = \mathbf{X}_b \mathbf{\beta}.
$$
These latest will be compared to the ones obtained using a linear regression model from Sklearn.

## Numerical Method <br>

Cost function
$$
cost = \frac{1}{N}\sum\limits_{i=1}^{N}(y_i - \hat{y}_i)^2,
$$
with $ \hat{y}_i = w x_i + b $ the prediction, $ w $ the weight, $ b $ the bias and $ x_i $ the data. <br>

### Gradient Descent 
In order to find $w$ and $b$ at the minimum
$$ 
w = w - \alpha \frac{\partial}{\partial w} cost \\
b = b - \alpha \frac{\partial}{\partial b} cost,
$$
with $\alpha$ the learning rate and
$$
\frac{\partial}{\partial w} cost = -\frac{2}{N}\sum\limits_{i=1}^{N} x_i(y_i - \hat{y}_i)\\
\frac{\partial}{\partial b} cost = -\frac{2}{N}\sum\limits_{i=1}^{N} (y_i - \hat{y}_i).
$$


References: 
- https://www.kaggle.com/code/jvachier/linear-regression
