Machine Learning Coursera - Andrew Ng
---

---

- [Week #1](#week-1)
  - [Introduction](#introduction)
    - [Machine Learning](#machine-learning)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
  - [Model and Cost Function](#model-and-cost-function)
    - [Model Representation](#model-representation)
    - [Cost Function](#cost-function)
    - [Cost Function - Intuition I](#cost-function---intuition-i)
    - [Cost Function - Intuition II](#cost-function---intuition-ii)
  - [Parameter Learning](#parameter-learning)
    - [Gradient Descent](#gradient-descent)
    - [Gradient Descent Intuition](#gradient-descent-intuition)
    - [Gradient Descent For Linear Regression](#gradient-descent-for-linear-regression)
  - [Linear Algebra Review](#linear-algebra-review)
    - [Matrices and Vectors](#matrices-and-vectors)
    - [Addition and Scalar Multiplication](#addition-and-scalar-multiplication)
    - [Matrix Vector Multiplication](#matrix-vector-multiplication)
    - [Matrix Matrix Multiplication](#matrix-matrix-multiplication)
    - [Matrix Multiplication Properties](#matrix-multiplication-properties)
    - [Inverse and Transpose](#inverse-and-transpose)
- [Week #2](#week-2)
  - [Multivariate Linear Regression](#multivariate-linear-regression)
    - [Multiple Features](#multiple-features)
    - [Gradient Descent for Multiple Variables](#gradient-descent-for-multiple-variables)
    - [Gradient Descent in Practice I - Feature Scaling](#gradient-descent-in-practice-i---feature-scaling)
    - [Gradient Descent in Practice II - Learning Rate](#gradient-descent-in-practice-ii---learning-rate)
    - [Features and Polynomial Regression](#features-and-polynomial-regression)
      - [Polynomial Regression](#polynomial-regression)
  - [Computing Parameters Analytically](#computing-parameters-analytically)
    - [Normal Equation](#normal-equation)
    - [Normal Equation Noninvertibility](#normal-equation-noninvertibility)
- [Week #3](#week-3)
  - [Classification and Representation](#classification-and-representation)
    - [Classification](#classification)
    - [Hypothesis Representation](#hypothesis-representation)
    - [Decision Boundary](#decision-boundary)
  - [Logistic Regression Model](#logistic-regression-model)
    - [Cost Function](#cost-function-1)
    - [Simplified Cost Function and Gradient Descent](#simplified-cost-function-and-gradient-descent)
      - [Gradient Descent](#gradient-descent-1)
    - [Advanced Optimization](#advanced-optimization)
  - [Multiclass Classification](#multiclass-classification)
    - [Multiclass Classification: One-vs-all](#multiclass-classification-one-vs-all)
  - [Solving the Problem of Overfitting](#solving-the-problem-of-overfitting)
    - [The Problem of Overfitting](#the-problem-of-overfitting)
    - [Cost Function](#cost-function-2)
    - [Regularized Linear Regression](#regularized-linear-regression)
      - [Gradient Descent](#gradient-descent-2)
      - [Normal Equation](#normal-equation-1)
    - [Regularized Logistic Regression](#regularized-logistic-regression)
      - [Cost Function](#cost-function-3)
- [Week #4](#week-4)
  - [Motivations](#motivations)
    - [Non-linear Hypotheses](#non-linear-hypotheses)
    - [Neurons and the Brain](#neurons-and-the-brain)
  - [Neural Networks](#neural-networks)
    - [Model Representation I](#model-representation-i)
    - [Model Representation II](#model-representation-ii)
  - [Applications](#applications)
    - [Examples and Intuitions I](#examples-and-intuitions-i)
    - [Examples and Intuitions II](#examples-and-intuitions-ii)
    - [Multiclass Classification](#multiclass-classification-1)
- [Week #5](#week-5)
  - [Cost Function and Backpropagation](#cost-function-and-backpropagation)
    - [Cost Function](#cost-function-4)
    - [Backpropagation Algorithm](#backpropagation-algorithm)
    - [Backpropagation Intuition](#backpropagation-intuition)
  - [Backpropagation in Practice](#backpropagation-in-practice)
    - [Implementation Note: Unrolling Parameters](#implementation-note-unrolling-parameters)
    - [Gradient Checking](#gradient-checking)
    - [Random Initialization](#random-initialization)
    - [Putting It Together](#putting-it-together)
  - [Application of Neural Networks](#application-of-neural-networks)
    - [Autonomous Driving](#autonomous-driving)
- [Week #6](#week-6)
  - [Evaluating a Learning Algorithm](#evaluating-a-learning-algorithm)
    - [Evaluating a Hypothesis](#evaluating-a-hypothesis)
    - [Model Selection and Train/Validation/Test Sets](#model-selection-and-trainvalidationtest-sets)
  - [Bias vs. Variance](#bias-vs-variance)
    - [Diagnosing Bias vs. Variance](#diagnosing-bias-vs-variance)
    - [Regularization and Bias/Variance](#regularization-and-biasvariance)
    - [Learning Curves](#learning-curves)
    - [Deciding What to Do Next Revisited](#deciding-what-to-do-next-revisited)
  - [Building a Spam Classifier](#building-a-spam-classifier)
    - [Prioritizing What to Work On](#prioritizing-what-to-work-on)
    - [Error Analysis](#error-analysis)
  - [Handling Skewed Data](#handling-skewed-data)
    - [Error Metrics for Skewed Classes](#error-metrics-for-skewed-classes)
    - [Trading Off Precision and Recall](#trading-off-precision-and-recall)
  - [Using Large Data Sets](#using-large-data-sets)
    - [Data For Machine Learning](#data-for-machine-learning)


# Week #1

---

## Introduction

### Machine Learning

Two definitions of Machine Learning are offered. Arthur Samuel described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.

Tom Mitchell provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

In general, any machine learning problem can be assigned to one of two broad classifications:
- Supervised learning
- Unsupervised learning

### Supervised Learning

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into "regression" and "classification" problems. 
- Regression problem - In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function.
- Classigifcation problem - In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

### Unsupervised Learning

Unsupervised learning allows us to approach problems with little or no idea what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables.

We can derive this structure by clustering the data based on relationships among the variables in the data.

With unsupervised learning there is no feedback based on the prediction results.

## Model and Cost Function

### Model Representation
- $m$ - number of training examples
- $x$ - "input" variable / features
- $y$ - "output" variable / "target" variable
- $X$ - space of input values
- $Y$ - space of output values
- $(x,y)$ - one training example
- $(x^{(i)}, y^{(i)})$ - $i^{th}$ training example

A list of training examples is called a **training set**.

The way the supervised learning algorithm works is by feeding the trainign set to the learning algorithm. The job of the learning algorithm is to output a function which by convention is denoted by $h$ and $h$ stands for **hypothesis**. The job of the hypothesis is a function that takes input and tries to output the estimated value for the corresponding input. $h$ is a function that maps x to y.

![Model Representation](/Resources/modelRepresentation.png)

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $h : X \to Y$ so that $h(x)$ is a “good” predictor for the corresponding value of y.

The way to represent $h$ is by:

$$ h_{\theta} (x) = \theta_{0} + \theta_{1}x$$

Shorthand: $h(x)$ is equivalent to $h_{\theta}x$ 

This way, the model will predict that y is a linear function of x. It is called **linear regression**, or actually **linear regression with one variable**, or **univariate linear regression**.

### Cost Function

$\theta_{0}$ and $\theta_{1}$ are called the parameters of this model.

In linear regression, we want to come up with values for the parameters such that the line best fits to the data given.

Idea: Choose $\theta_{0}$ and $\theta_{1}$, so that $h_{\theta} (x)$ is close to $y$ for our training examples $(x,y)$

Therefore, a minimization problem has to be solved.

Minimize over $\theta_{0}$ and $\theta_{1}$ the sum of the square differences between the predicted value and the real output for each training example.

$$ \text{Minimize over } \theta_{0},\theta_{1}: \hskip{0.1em} \frac{1}{2m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^{2} $$

$$ h_{\theta}(x^{(i)}) = \theta_{0} + \theta_{1} x^{i} $$

Note: Minimizing one half of something should give the same result as minimizing the whole.

Therefore, the idea is to find the values of $\theta_{0}$ and $\theta_{1}$ that cause the above expression to be minimized.

The cost function is:

$$ J(\theta_{0}, \theta_{1}) = \frac{1}{2m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^{2} $$

$$ \text{Minimize over } \theta_{0},\theta_{1} \hskip{0.1em} J(\theta_{0}, \theta_{1}) $$

This cost function is also called the **squared error function** or sometimes called the **squared error cost function** or **mean squared error**.

The squared cost function is probably the most commonly used one as it works well in most cases.

We can measure the accuracy of our hypothesis function by using a **cost function**.

$$ J(\theta_{0}, \theta_{1}) = \frac{1}{2m} \sum_{i = 1}^{m} (\hat{y}^{(i)} - y^{(i)})^{2} $$

The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term.

![Cost Function](/Resources/costFunction.png)

### Cost Function - Intuition I

Working on a simplified hypothesis dependent only one parameter with the same cost function and minimizing only on one parameter $\theta_{1})$:

$$ h_{\theta} (x) = \theta_{1}x $$

Therefore, the fit line will pass through the origin and only depend on the slope, ($\theta_{1}$) parameter.

If we try to think of it in visual terms, our training data set is scattered on the $x-y$ plane. We are trying to make a straight line (defined by $h_{\theta} (x)$) which passes through these scattered data points.

Our objective is to get the best possible line. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. Ideally, the line should pass through all the points of our training data set. In such a case, the value of $J(\theta_{0}, \theta_{1})$ will be 0. The following example shows the ideal situation where we have a cost function of 0.

![Cost Function Intuition I](/Resources/costFunctionIntuition1.png)

When $\theta_{1} = 1$, we get a slope of 1 which goes through every single data point in our model. Conversely, when $\theta_{1} = 0.5$, we see the vertical distance from our fit to the data points increase.

![Cost Function Intuition II](/Resources/costFunctionIntuition2.png)

This increases our cost function to 0.58. Plotting several other points yields the following graph:

![Cost Function Intuition III](/Resources/costFunctionIntuition3.png)

Thus as a goal, we should try to minimize the cost function. In this case, $\theta_{1} = 1$ is our global minimum.

### Cost Function - Intuition II

Trying to fit a line with 2 parameters now instead of only one as in the example above would yield a cost function $J$, similar to a 2-1 map.

Therefore, for different choices of the 2 parameters, the cost function will yield different results.

A contour plot is a graph that contains many contour lines. A contour line of a two variable function has a constant value at all points of the same line. An example of such a graph is the one to the right below.

![Cost Function IV](/Resources/costFunctionIntuition4.png)

Taking any color and going along the 'circle', one would expect to get the same value of the cost function. For example, the three green points found on the green line above have the same value for $J(\theta_{0},\theta_{1})$ and as a result, they are found along the same line. The circled $x$ displays the value of the cost function for the graph on the left when $\theta_{0} = 800$ and $\theta_{1} = -0.15$. Taking another $h(x)$ and plotting its contour plot, one gets the following graphs:

![Cost Function V](/Resources/costFunctionIntuition5.png)

When $\theta_{0} = 360$ and $\theta_{1} = 0$, the value of $J(\theta_{0},\theta_{1})$ in the contour plot gets closer to the center thus reducing the cost function error. Now giving our hypothesis function a slightly positive slope results in a better fit of the data. 

![Cost Function VI](/Resources/costFunctionIntuition6.png)

The graph above minimizes the cost function as much as possible and consequently, the result of $\theta_{1}$ and $\theta_{0}$ tend to be around $0.12$ and $250$ respectively. Plotting those values on our graph to the right seems to put our point in the center of the inner most 'circle'.

## Parameter Learning

### Gradient Descent

Gradient Descent is the algorithm used to find the minimum of the cost function that we previously talked about, but not only.

So we have our hypothesis function and we have a way of measuring how well it fits into the data. Now we need to estimate the parameters in the hypothesis function. That's where gradient descent comes in.

Imagine that we graph our hypothesis function based on its fields $\theta_{0}$ and $\theta_{1}$ (actually we are graphing the cost function as a function of the parameter estimates). We are not graphing $x$ and $y$ itself, but the parameter range of our hypothesis function and the cost resulting from selecting a particular set of parameters.

We put $\theta_{0}$ on the $x$ axis and $\theta_{1}$ on the $y$ axis, with the cost function on the vertical $z$ axis. The points on our graph will be the result of the cost function using our hypothesis with those specific theta parameters. The graph below depicts such a setup.

![Gradient Descent Visualization](/Resources/gradientDescent1.png)

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph, i.e. when its value is the minimum. The red arrows show the minimum points in the graph.

The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $\alpha$, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter $\alpha$. A smaller $\alpha$ would result in a smaller step and a larger $\alpha$ results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(\theta_{0}, \theta_{1})$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

Gradient Descent Algorithm is defined as:

Repeat until convergence: {
    $$ \theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial\theta_{j}} J(\theta_{0}, \theta_{1}) \hskip{0.2em} (\text{for } j = 0 \text{ and } j = 1) $$
}

Parts:
- $\alpha$ - learning rate (how big of a step we take descending)
- $\frac{\partial}{\partial\theta_{j}} J(\theta_{0}, \theta_{1})$ - derivative term

Both partials are simultaneously updated,  meaning that the updated values for each parameter are updated in such a way that they did not interfere with the respective calculations of the other.

temp0 $:= \theta_{j} := \theta_{0} - \alpha \frac{\partial}{\partial\theta_{0}} J(\theta_{0}, \theta_{1}) $

temp1 $:= \theta_{j} := \theta_{1} - \alpha \frac{\partial}{\partial\theta_{1}} J(\theta_{0}, \theta_{1}) $

$\theta_{0} :=$ temp0

$\theta_{1} :=$ temp1

![Gradient Descent Simultaneous Update](/Resources/gradientDescent2.png)

At each iteration $j$, one should simultaneously update the parameters $\theta_{1}, \theta_{2}, \dots, \theta_{n}$. Updating a specific parameter prior to calculating another one on the $j^{(th)}$ iteration would yield to a wrong implementation.

### Gradient Descent Intuition

Regardless of the slope's sign for $\frac{d}{d\theta_{1}}$, $\theta_{1}$ eventually converges to its minimum value. The following graph shows that when the slope is negative, the value of $\theta_{1}$ increases and when it is positive, the value of $\theta_{1}$ decreases.

![Gradient Descent Intuition I](/Resources/gradientDescent3.png)

On a side note, we should adjust our parameter $\alpha$ to ensure that the gradient descent algorithm converges in a reasonable time. Failure to converge or too much time to obtain the minimum value imply that our step size is wrong.

![Gradient Descent Intuition II](/Resources/gradientDescent4.png)

How does gradient descent converge with a fixed step size $\alpha$?

The intuition behind the convergence is that $\frac{d}{d\theta_{1}}$ approaches 0 as we approach the bottom of our convex function. At the minimum, the derivative will always be 0 and thus we get:

$$ \theta_{1} = \theta_{1} - \alpha \cdot 0 $$

![Gradient Descent Intuition III](/Resources/gradientDescent5.png)

### Gradient Descent For Linear Regression

When specifically applied to the case of linear regression, a new form of the gradient descent equation can be derived. We can substitute our actual cost function and our actual hypothesis function and modify the equation to:

repeat until convergence: {
    $$ \theta_{0} := \theta_{0} - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x_{i}) - y_{i}) $$
    $$ \theta_{1} := \theta_{1} - \alpha \frac{1}{m} \sum_{i = 1}^{m} ((h_{\theta}(x_{i}) - y_{i})x_{i}) $$
}

where $m$ is the size of the training set, $\theta_{0}$ a constant that will be changing simultaneously with $\theta_{1}$ and $x_{i}$, $y_{i}$ are values of the given training set (data).

Note that we have separated out the two cases for $\theta_{j}$ into separate equations for $\theta_{0}$ and $\theta_{1}$; and that for $\theta_{1}$ we are multiplying $x_{i}$ at the end due to the derivative. The following is a derivation of $\frac{\partial}{\partial \theta_{j}} J(\theta)$ for a single example:

![Gradient Descent for Linear Regression Derivation](/Resources/gradientDescentLinRegression.png)

The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

So, this is simply gradient descent on the original cost function $J$. This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local minima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges (assuming the learning rate $\alpha$ is not too large) to the global minimum. Indeed, $J$ is a convex quadratic function. Here is an example of gradient descent as it is run to minimize a quadratic function.

![Gradient Descent for Linear Regression Visualization Contour Lines](/Resources/gradientDescentRun.png)

The ellipses shown above are the contours of a quadratic function. Also shown is the trajectory taken by gradient descent, which was initialized at $(48,30)$. The $x$’s in the figure (joined by straight lines) mark the successive values of $\theta$ that gradient descent went through as it converged to its minimum.

## Linear Algebra Review

### Matrices and Vectors

Matrices are 2-dimensional arrays:

$$
\begin{bmatrix}
a & b & cd & e & fg & h & ij & k & l
\end{bmatrix}
$$

The above matrix has four rows and three columns, so it is a $4 \times 3$ matrix.

A vector is a matrix with one column and many rows:

$$
\begin{bmatrix}
w \\
x \\
y \\ 
z
\end{bmatrix}
$$

So vectors are a subset of matrices. The above vector is a $4 \times 1$ matrix.

**Notation and terms:**

- $A_{ij}$ refers to the element in the $i^{th}$ row and $j^{th}$ column of matrix $A$.
- A vector with '$n$' rows is referred to as an '$n$'-dimensional vector.
- $v_{i}$ refers to the element in the $i^{th}$ row of the vector.
- In general, all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
- Matrices are usually denoted by uppercase names while vectors are lowercase.
- "Scalar" means that an object is a single value, not a vector or matrix.
- $\mathbb{R}$ refers to the set of scalar real numbers.
- $\mathbb{R}^{n}$ refers to the set of $n$-dimensional vectors of real numbers.

Octave/Matlab Code:

```matlab
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector 
v = [1;2;3] 

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v 
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)
```

### Addition and Scalar Multiplication

Addition and subtraction are **element-wise**, so you simply add or subtract each corresponding element:

$
\begin{bmatrix}
a & bc & d
\end{bmatrix}
$ + $
\begin{bmatrix}
w & xy & z
\end{bmatrix}
$ = $
\begin{bmatrix}
a + w & bc + xy & d + z
\end{bmatrix}
$

Subtracting Matrices:

$
\begin{bmatrix}
a & bc & d
\end{bmatrix}
$ - $
\begin{bmatrix}
w & xy & z
\end{bmatrix}
$ = $
\begin{bmatrix}
a - w & bc - xy & d - z
\end{bmatrix}
$

To add or subtract two matrices, their dimensions must be **the same**.

In scalar multiplication, we simply multiply every element by the scalar value:

$
\begin{bmatrix}
a & bc & d
\end{bmatrix}
$ * $
x
$ = $
\begin{bmatrix}
a * x & bc * x & d * x
\end{bmatrix}
$

In scalar division, we simply divide every element by the scalar value:

$
\begin{bmatrix}
a & bc & d
\end{bmatrix}
$ / $
x
$ = $
\begin{bmatrix}
a / x & bc / x & d / x
\end{bmatrix}
$

Octave/Matlab Code:

```matlab
% Initialize matrix A and B 
A = [1, 2, 4; 5, 3, 2]
B = [1, 3, 4; 1, 1, 1]

% Initialize constant s 
s = 2

% See how element-wise addition works
add_AB = A + B 

% See how element-wise subtraction works
sub_AB = A - B

% See how scalar multiplication works
mult_As = A * s

% Divide A by s
div_As = A / s

% What happens if we have a Matrix + scalar?
add_As = A + s
```

### Matrix Vector Multiplication

We map the column of the vector onto each row of the matrix, multiplying each element and summing the result.

$
\begin{bmatrix}
a & b\\
c & d\\
e & f
\end{bmatrix}
$ * $
\begin{bmatrix}
x \\
y
\end{bmatrix}
$ = $
\begin{bmatrix}
a * x + b * y\\
c * x + d * y\\
e * x + f * y
\end{bmatrix}
$

The result is a **vector**. The number of **columns** of the matrix must equal the number of **rows** of the vector.

An **m $\boldsymbol\times$ n** matrix multiplied by an **n $\boldsymbol\times$ 1** vector results in an **m $\boldsymbol\times$ 1** vector.

Octave/Matlab Code:

```matlab
% Initialize matrix A 
A = [1, 2, 3; 4, 5, 6;7, 8, 9] 

% Initialize vector v 
v = [1; 1; 1] 

% Multiply A * v
Av = A * v
```

### Matrix Matrix Multiplication

We multiply two matrices by breaking it into several vector multiplications and concatenating the result.

$
\begin{bmatrix}
a & b\\
c & d\\
e & f
\end{bmatrix}
$ * $
\begin{bmatrix}
w & x\\
y & z
\end{bmatrix}
$ = $
\begin{bmatrix}
a * w + b * y & a * x + b * z\\
c * w + d * y & c * x + d * z\\
e * w + f * y & e * x + f * z
\end{bmatrix}
$

An **m $\boldsymbol\times$ n** matrix multiplied by an **n $\boldsymbol\times$ o** matrix results in an **m $\boldsymbol\times$ o** matrix. In the above example, a 3 $\times$ 2 matrix times a 2 $\times$ 2 matrix resulted in a 3 $\times$ 2 matrix.

To multiply two matrices, the number of **columns** of the first matrix must equal the number of **rows** of the second matrix.

Octave/Matlab Code:

```matlab
% Initialize a 3 by 2 matrix 
A = [1, 2; 3, 4;5, 6]

% Initialize a 2 by 1 matrix 
B = [1; 2] 

% We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1) 
mult_AB = A*B

% Make sure you understand why we got that result
```

### Matrix Multiplication Properties

- Matrices are not commutative: $A * B \neq B * A$
- Matrices are associative: $(A * B) * C = A * (B * C)$

The **identity matrix**, when multiplied by any matrix of the same dimensions, results in the original matrix. It's just like multiplying numbers by 1. The identity matrix simply has 1's on the diagonal (upper left to lower right diagonal) and 0's elsewhere.

$$
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
$$

When multiplying the identity matrix after some matrix (A∗I), the square identity matrix's dimension should match the other matrix's **columns**. When multiplying the identity matrix before some other matrix $(I*A)$, the square identity matrix's dimension should match the other matrix's **rows**.

Octave/Matlab Code:

```matlab
% Initialize random matrices A and B 
A = [1,2;4,5]
B = [1,1;0,2]

% Initialize a 2 by 2 identity matrix
I = eye(2)

% The above notation is the same as I = [1,0;0,1]

% What happens when we multiply I*A ? 
IA = I*A 

% How about A*I ? 
AI = A*I 

% Compute A*B 
AB = A*B 

% Is it equal to B*A? 
BA = B*A 

% Note that IA = AI but AB != BA
```

### Inverse and Transpose

The **inverse** of a matrix $A$ is denoted $A^{-1}$. Multiplying by the inverse results in the identity matrix.

A non square matrix does not have an inverse matrix. We can compute inverses of matrices in Octave with the $pinv(A)$ function and in Matlab with the $inv(A)$ function. Matrices that don't have an inverse are *singular* or *degenerate*.

The **transposition** of a matrix is like rotating the matrix 90° in clockwise direction and then reversing it. We can compute transposition of matrices in Matlab with the $transpose(A)$ function or $A'$:

$$
A =
\begin{bmatrix}
a & b\\
c & d\\
e & f
\end{bmatrix}
$$

$$
A^{T} =
\begin{bmatrix}
a & c & e\\
b & d & f
\end{bmatrix}
$$

In other words:

$A_{ij} = A_{ji}^{T}$

Octave/Matlab Code:

```matlab
% Initialize matrix A 
A = [1,2,0;0,5,6;7,0,9]

% Transpose A 
A_trans = A' 

% Take the inverse of A 
A_inv = inv(A)

% What is A^(-1)*A? 
A_invA = inv(A)*A
```

# Week #2

---

## Multivariate Linear Regression

### Multiple Features

Linear regression with multiple variables is also known as **multivariate linear regression**.

We now introduce notation for equations where we can have any number of input variables.

- $x_{j}^{(i)}$ = value of feature $j$ in the $i^{th}$ training example
- $x^{(i)}$ = the input (features) of the $i^{th}$ training example
- $m$ = the number of training examples
- $n$ = the number of features

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

$$ h_{\theta} (x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + \theta_{3}x_{3} + \dots + \theta_{n}x_{n} $$

In order to develop intuition about this function, we can think about $\theta_{0}$ as the basic price of a house, $\theta_{1}$ as the price per square meter, $\theta_{2}$ as the price per floor, etc. $x_{1}$ will be the number of square meters in the house, $x_{2}$ the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:

$
h_{\theta} (x) = 
\begin{bmatrix}
\theta_{0} & \theta_{1} & \cdots & \theta_{n}
\end{bmatrix}
$$
\begin{bmatrix}
x_{0}\\
x_{1}\\
\vdots\\
x_{n}
\end{bmatrix}
$ = $
\theta^{T}x
$

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

**Remark**: Note that for convenience reasons in this course we assume $x_{0}^{(i)} = 1 \text{ for } (i\in { 1,\dots, m })$. This allows us to do matrix operations with $\theta$ and $x$. Hence making the two vectors '$\theta$' and $x^{(i)}$ match each other element-wise (that is, have the same number of elements: $n+1$).

### Gradient Descent for Multiple Variables

The gradient descent equation itself is generally the same form; we just have to repeat it for our $n$ features:

repeat until convergence: {
    $$ \theta_{0} := \theta_{0} - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_{0}^{(i)} $$
    $$ \theta_{1} := \theta_{1} - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_{1}^{(i)} $$
    $$ \theta_{2} := \theta_{2} - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_{2}^{(i)} $$
    $$\vdots$$
}

In other words:

repeat until convergence: {
    $$ \theta_{j} := \theta_{j} - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_{j}^{(i)} \hskip{0.2em} \text{for } j := 0, \dots, n $$
}

The following image compares gradient descent with one variable to gradient descent with multiple variables:

![Multivariate Gradient Descent Algorithm](/Resources/multivariateGradientDescentAlgo.png)

### Gradient Descent in Practice I - Feature Scaling

We can speed up gradient descent by having each of our input values in roughly the same range. This is because $\theta$ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:

$$ -1 \leq x_{(i)} \leq 1 $$

or

$$ -0.5 \leq x_{(i)} \leq 0.5 $$

These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are **feature scaling** and **mean normalization**. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just $1$. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:

$$ x_{i} := \frac{x_{i} - \mu_{i}}{s_{i}} $$

Where $\mu_{i}$ is the **average** of all the values for feature ($i$) and $s_{i}$ is the range of values (max - min), or $s_{i}$ is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if $x_{i}$ represents housing prices with a range of 100 to 2000 and a mean value of 1000, then, $x_{i} := \frac{price - 1000}{1900}$.

### Gradient Descent in Practice II - Learning Rate

**Debugging gradient descent.** Make a plot with number of iterations on the $x$-axis. Now plot the cost function, $J(\theta)$ over the number of iterations of gradient descent. If $J(\theta)$ ever increases, then you probably need to decrease $\alpha$.

**Automatic convergence test.** Declare convergence if $J(\theta)$ decreases by less than $E$ in one iteration, where $E$ is some small value such as $10^{-3}$. However in practice it's difficult to choose this threshold value.

![Gradient Descent Convergence Test](/Resources/gradientDescentLearningRate1.png)

It has been proven that if learning rate $\alpha$ is sufficiently small, then $J(\theta)$ will decrease on every iteration.

![Gradient Descent Learning Rate](/Resources/gradientDescentLearningRate2.png)

To summarize:
- If $\alpha$ is too small: slow convergence.
- If $\alpha$ is too large: $J(\theta)$ may not decrease on every iteration and thus may not converge.

### Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can combine multiple features into one. For example, we can combine $x_{1}$ and $x_{2}$ into a new feature $x_{3}$ by taking $x_{1} \cdot x_{2}$.

#### Polynomial Regression

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is $h_{\theta} (x) = \theta_{0} + \theta_{1}x_{1}$ then we can create additional features based on $x_{1}$, to get the quadratic function $h_{\theta} (x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{1}^{2}$ or the cubic function $h_{\theta} (x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{1}^{2} + \theta_{3}x_{1}^{3}$

In the cubic version, we have created new features $x_{2}$ and $x_{3}$ where $x_{2} = x_{1}^{2}$ and $x_{3} = x_{1}^{3}$.

To make it a square root function, we could do: $h_{\theta} (x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}\sqrt{x_{1}}$

One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

eg. if $x_{1}$ has range $1 - 1000$ then range of $x_{1}^{2}$ becomes $1 - 100000$ and that of $x_{1}^{3}$ becomes $1 - 10^{9}$.

## Computing Parameters Analytically

### Normal Equation

Gradient descent gives one way of minimizing $J$. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize $J$ by explicitly taking its derivatives with respect to the $\theta_{j}$’s, and setting them to zero. This allows us to find the optimum $\theta$ without iteration. The normal equation formula is given below:

$$ \theta = (X^{T}X)^{-1}X^{T}y $$

![Normal Equation Example](/Resources/normalEquation.png)

There is **no need** to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent              | Normal Equation                               |
|-------------------------------|-----------------------------------------------|
| Need to choose $\alpha$       | No need to choose $\alpha$                    |
| Needs many iterations         | No need to iterate                            |
| $O(kn^{2})$                   | $O(n^{3})$, need to calculate $(X^{T}X)^{-1}$ |
| Works well when $n$ is large  | Slow if $n$ is very large                     |

With the normal equation, computing the inversion has complexity $O(n^{3})$. So if we have a very large number of features, the normal equation will be slow. In practice, when $n$ exceeds $10,000$ it might be a good time to go from a normal solution to an iterative process.

### Normal Equation Noninvertibility

When implementing the normal equation in Octave we want to use the '$pinv$' function rather than '$inv$.' The '$pinv$' function will give you a value of $\theta$ even if $X^{T}X$ is not invertible.

If $X^{T}X$ is **noninvertible**, the common causes might be having:
- Redundant features, where two features are very closely related (i.e. they are linearly dependent)
- Too many features (e.g. $m \leq n$). In this case, delete some features or use "regularization" (to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

# Week #3

---

## Classification and Representation

### Classification

To attempt classification, one method is to use linear regression and map all predictions greater than $0.5$ as a $1$ and all less than $0.5$ as a $0$. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the **binary classification problem** in which $y$ can take on only two values, $0$ and $1$. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then $x^{(i)}$ may be some features of a piece of email, and $y$ may be $1$ if it is a piece of spam mail, and $0$ otherwise. Hence, $y \in \{0, 1\}$. $0$ is also called the negative class, and $1$ the positive class, and they are sometimes also denoted by the symbols "-" and "+". Given $x^{(i)}$, the corresponding $y^{(i)}$ is also called the label for the training example.

### Hypothesis Representation

We could approach the classification problem ignoring the fact that $y$ is discrete-valued, and use our old linear regression algorithm to try to predict $y$ given $x$. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for $h_{\theta} (x)$ to take values larger than $1$ or smaller than $0$ when we know that $y \in \{0, 1\}$. To fix this, let’s change the form for our hypotheses $h_{\theta} (x)$ to satisfy $0 \leq h_{\theta} (x) \leq 10$. This is accomplished by plugging $\theta^{T}x$ into the Logistic Function.

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

$$ h_{\theta} (x) = g(\theta^{T} x) $$
$$ z = \theta^{T} x $$
$$ g(z) = \frac{1}{1 + e^{-z}} $$

The following image shows us what the sigmoid function looks like:

![Sigmoid / Logistic Function](\Resources/logisticFunction.png)

The function $g(z)$, shown here, maps any real number to the $(0, 1)$ interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

$h_{\theta} (x)$ will give us the **probability** that our output is $1$. For example, $h_{\theta} (x) = 0.7$ gives us a probability of $70\%$ that our output is $1$. Our probability that our prediction is $0$ is just the complement of our probability that it is $1$ (e.g. if probability that it is $1$ is $70\%$, then the probability that it is $0$ is $30\%$).

$$ h_{\theta} (x) = P (y = 1 | x ; \theta) = 1 − P (y = 0 | x ; \theta) $$
$$ P (y = 0 | x ; \theta) + P (y = 1 | x ; \theta) = 1 $$

### Decision Boundary

In order to get our discrete $0$ or $1$ classification, we can translate te output of the hypothesis function as follows:

$$ h_{\theta} (x) \geq 0.5 \to y = 1 $$
$$ h_{\theta} (x) < 0.5 \to y = 0 $$

The way our logistic function $g$ behaves is that when its input is greater than or equal to zero, its output is greater than or equalt $0.5$:

$$ g(z) \geq 0.5 \text{ when } z \geq 0 $$

Remember:

$$ z = 0, e^{0} = 1 \Rightarrow g(z) = 1/2 $$
$$ z \to \infty, e^{-\infty} \to 0 \Rightarrow g(z) = 1 $$
$$ z \to -\infty, e^{\infty} \to \infty \Rightarrow g(z) = 0 $$

So if our input to $g$ is $\theta^{T}X$, then that means:

$$ h_{\theta} (x) = g(\theta^{T}x) \geq 0.5 \text{ when } \theta^{T}x \geq 0 $$

From these statements we can now say:

$$ \theta^{T}x \geq 0 \Rightarrow y = 1 $$
$$ \theta^{T}x < 0 \Rightarrow y = 0 $$

The **decision boundary** is the line that separates the area where $y = 0$ and where $y = 1$. It is created by our hypothesis function.

**Example:**

$$ \theta = \begin{bmatrix}5\\-1\\0\end{bmatrix} $$
$$ y = 1 \text{ if } 5 + (-1)x_{1} + 0x_{2} \geq 0 $$
$$ 5 - x_{1} \geq 0 $$
$$ -x_{1} \geq -5 $$
$$ x_{1} \leq 5 $$

In this case, our decision boundary is a straight vertical line placed on the graph where $x_{1} = 5$, and everything to the left of that denotes $y = 1$, while everything to the right denotes $y = 0$.

Again, the input to the sigmoid function $g(z)$ (e.g. $\theta^{T}X$) doesn't need to be linear, and could be a function that describes a circle (e.g. $z = \theta_{0} + \theta_{1}x_{1}^{2} + \theta_{2}x_{2}^{2}$) or any shape to fit our data.

## Logistic Regression Model

### Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be wavy, causing many local optima. In other words, it will not be a convex function.

Instead, our cost function for logistic regression looks like:

$$ J(\theta) = \frac{1}{m} \sum_{i = 1}^{m} \text{ Cost} (h_{\theta} (x^{(i)}), y^{(i)}) $$
$$ \text{Cost} (h_{\theta} (x), y) = -\log(h_{\theta} (x)) \hskip{0.2em} \text{if } y = 1 $$
$$ \text{Cost} (h_{\theta} (x), y) = -\log(1 - h_{\theta} (x)) \hskip{0.2em} \text{if } y = 0 $$

When $y = 1$, we get the following plot for $J(\theta)$ vs $h_{\theta} (x)$:

![Hypothesis Logistic Regression When y = 1](\Resources/hypothesisLogisticRegression1.png)

Similarly, when $y = 0$, we get the following plot for $J(\theta)$ vs $h_{\theta} (x)$:

![Hypothesis Logistic Regression When y = 0](\Resources/hypothesisLogisticRegression2.png)

$$ \text{Cost} (h_{\theta} (x), y) = 0 \text{ if } h_{\theta} (x) = y $$
$$ \text{Cost} (h_{\theta} (x), y) \to \infty \text{ if } y = 0 \text{ and } h_{\theta} (x) \to 1 $$
$$ \text{Cost} (h_{\theta} (x), y) \to \infty \text{ if } y = 1 \text{ and } h_{\theta} (x) \to 0 $$

If our correct answer 'y' is $0$, then the cost function will be $0$ if our hypothesis function also outputs $0$. If our hypothesis approaches $1$, then the cost function will approach infinity.

If our correct answer 'y' is $1$, then the cost function will be $0$ if our hypothesis function outputs $1$. If our hypothesis approaches $0$, then the cost function will approach infinity.

Note that writing the cost function in this way guarantess that $J(\theta)$ is convex for logistic regression.

### Simplified Cost Function and Gradient Descent

We can compress our cost function's two conditional cases into one case:

$$ \text{Cost} (h_{\theta} (x), y) = -y \log(h_{\theta} (x)) - (1 - y) \log(1 - h_{\theta} (x)) $$

Notice that when $y$ is equl to $1$, then the second term $(1 - y) \log(1 - h_{\theta} (x))$ will be zero and will not affect the result. If $y$ is equal to $0$, then the first term $-y \log(h_{\theta} (x))$ will be zero and will not affect the result.

We can fully write out our entire cost function as follows:

$$ J(\theta) = -\frac{1}{m} \sum_{i = 1}^{m} [y^{(i)} \log(h_{\theta} (x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta} (x^{(i)}))] $$

A vectorized implementation is:

$$ h = g(X\theta) $$
$$ J(\theta) = \frac{1}{m} \cdot (-y^{T}\log(h) - (1 - y)^{T}\log(1 - h)) $$

#### Gradient Descent

Remember that the general form of gradient descent is:

Repeat {
    $$ \theta_{j} := \theta_{j} - \alpha \frac{\partial}{\partial\theta_{j}} J(\theta) $$
}

We can work out the derivative part using calculus to get:

Repeat {
    $$ \theta_{j} := \theta_{j} - \frac{\alpha}{m} \sum_{i = 1}^{m} (h_{\theta} (x^{(i)}) - y^{(i)}) x_{j}^{(i)} $$
}

Notice that this algorithm is indetical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:

$$ \theta := \theta - \frac{\alpha}{m} X^{T}(g(X\theta) - \vec{y}) $$

### Advanced Optimization

"Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize $\theta$ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use the libraries instead, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value $\theta$:
- $J(\theta)$
- $\frac{\partial}{\partial \theta_{j}} J(\theta)$

We can write a single function that returns both of these:

```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Then we can use Octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()".

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
    [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

## Multiclass Classification

### Multiclass Classification: One-vs-all

Now we will approach the classification of data when we have more than two categories. Instead of $y = \{0, 1\}$ we will expand our definition so that $y = \{0, 1, \dots, n\}$.

Since $y = \{0, 1, \dots, n\}$, we divide our problem into $n + 1$ ($+1$ because our index starts at $0$) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

$$ y \in \{0, 1, \dots, n\} $$
$$ h_{\theta}^{(0)} = P(y = 0 | x ; \theta) $$
$$ h_{\theta}^{(1)} = P(y = 1 | x ; \theta) $$
$$ \vdots $$
$$ h_{\theta}^{(n)} = P(y = n | x ; \theta) $$
$$ \text{prediction } = \text{ max}_{i} (h_{\theta}^{(i)} (x)) $$

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

The following image shows how one could classify 3 classes:

![Multiclass Classification: 3 Classes with Binary Logistic Regression](\Resources/multiclassClassification.png)

**To summarize:**

Train a logistic regression classifier $h_{\theta} (x)$ for each class to predict the probability that $y = i$.

To make a prediction on a new $x$, pick the class that maximizes $h_{\theta} (x)$.

## Solving the Problem of Overfitting

### The Problem of Overfitting

Consider the problem of predicting $y$ from $x \in \mathbb{R}$. The leftmost figure below shows the result of fitting a $y = \theta_{0} + \theta_{1}x$ to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good.

![Overfitting Example](Resources/overfitting.png)

Instead, if we had added an extra feature $x^{2}$, and fit $y = \theta_{0} + \theta_{1}x + \theta_{2}x^{2}$, then we obtain a slightly better fit to the data (See middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a $5^{th}$ order polynomial $y = \sum_{j=0}^{5} \theta_{j}x^{j}$. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices ($y$) for different living areas ($x$). Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of **underfitting**-in which the data clearly shows structure not captured by the model-and the figure on the right is an example of **overfitting**.

Underfitting, or high bias, is when the form of our hypothesis function $h$ maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features:
   - Manually select which features to keep.
   - Usa a model selection algorithm (studied later in the course).
2) Regularization
   - Keep all the features, but reduce the magnitude of parameters $\theta_{j}$.
   - Regularization works well when we have a lot of slightly useful features.

### Cost Function

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:

$$ \theta_{0} + \theta_{1}x + \theta_{2}x^{2} + \theta_{3}x^{3} + \theta_{4}x^{4} $$

We'll want to eliminate the influence of $\theta_{3}x^{3}$ and $\theta_{4}x^{4}$. Without actually getting rid of those features or changing the form of our hypothesis, we can instead modify our **cost function**:

$$ \text{min}_{\theta} \frac{1}{2m} \sum_{i = 1}^{m} (h_{\theta} (x^{(i)}) - y^{(i)})^{2} + 1000 \cdot \theta_{3}^{2} + 1000 \cdot \theta_{4}^{2}) $$

We've added two extra terms at the end to inflate the cost of $\theta_{3}$ and $\theta_{4}$. Now, in order for the cost function to get close to zero, we will have to reduce the values of $\theta_{3}$ and $\theta_{4}$ to near zero. This will in turn greatly reduce the values of $\theta_{3}x^{3}$ and $\theta_{4}x^{4}$ in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms $\theta_{3}x^{3}$ and $\theta_{4}x^{4}$.

![Regularization Example](Resources/regularization.png)

We could also regularize all of our theta parameters in a single summation as:

$$ \text{min}_{\theta} \frac{1}{2m} \sum_{i = 1}^{m} (h_{\theta} (x^{(i)}) - y^{(i)})^{2} + \lambda \sum_{j = 1}^{n} \theta_{j}^{2} $$

The $\lambda$, or lambda, is the **regularization parameter**. It determines how much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if $\lambda = 0$ or is too small ?

### Regularized Linear Regression

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

#### Gradient Descent

We will modify our gradient descent function to separate out $\theta_{0}$ from the rest of the parameters because we do not want to penalize $\theta_{0}$.

Repeat {
  $$ \theta_{0} := \theta_{0} - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta} (x^{(i)}) - y^{(i)})x_{0}^{(i)} $$
  $$ \theta_{j} := \theta_{j} - \alpha \left[ \left( \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta} (x^{(i)}) - y^{(i)})x_{j}^{(i)}\right) + \frac{\lambda}{m}\theta_{j} \right] \hskip{2em} j \in \{1,2,\dots, n\} $$
}

The term $\frac{\lambda}{m}\theta_{j}$ performs our regularization. With some manipulation our update rule can be represented as:

$$ \theta_{j} := \theta_{j} (1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i = 1}^{m} (h_{\theta} (x^{(i)}) - y^{(i)})x_{j}^{(i)} $$

The first term in the above equation, $1 - \alpha \frac{\lambda}{m}$ will always be less than $1$. Intuitively you can see it as reducing the value of $\theta_{j}$ by some amount on every update. Notice that the second term is now exactly the same as it was before.

#### Normal Equation

Now let's approach regularization usign the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:

$$ \theta = (X^{T}X + \lambda \cdot L)^{-1}X^{T}y $$

$$ \text{where } L  \begin{bmatrix}
0 & & & & \\
& 1 & & & \\
& & 1 & & \\
& & & \ddots & \\
& & & & 1
\end{bmatrix}$$

$L$ is a matrix with $0$ at the top left and $1$'s down the diagonal, with $0$'s everywhere else. It should have dimension $(n + 1) \times (n + 1)$. Intuitively, this is the identity matrix (though we are not including $x_{0}$), multiplied with a single real number $\lambda$.

Recall that if $m < n$, then $X^{T}X$ is non-invertible. However, when we add the term $\lambda \cdot L$, then $X^{T}X + \lambda \cdot L$ becomes invertible.

### Regularized Logistic Regression

We can regularize logistic regression in a similar way that we regularize linear regression. As a result, we can avoid overfitting. The following image shows how the regularized function, displayed by the pink line, is less likely to overfit than the non-regularized function represented by the blue line:

![Regularized Logistic Regression](Resources/regularizedLogistic.png)

#### Cost Function

Recall that our cost function for logistic regression was:

$$ J(\theta) = -\frac{1}{m} \sum_{i = 1}^{m} [y^{(i)} \log(h_{\theta} (x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta} (x^{(i)}))] $$

We can regularize this equation by adding a term to the end:

$$ J(\theta) = -\frac{1}{m} \sum_{i = 1}^{m} [y^{(i)} \log(h_{\theta} (x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta} (x^{(i)}))] + \frac{\lambda}{2m} \sum_{j = 1}^{n} \theta_{j}^{2} $$

The second sum, $\sum_{j = 1}^{n} \theta_{j}^{2}$ **means to explicitly exclude** the bias term, $\theta_{0}$. I.e. the $\theta$ vector is indexed from $0$ to $n$ (holding $n + 1$ values, $\theta_{0}$ through $\theta_{n}$), and this sum explicitly skips $\theta_{0}$, by running from $1$ to $n$, skipping $0$. Thus, when computing the equation, we should continuously update the two following equations:

![Gradient Descent for Regularized Logistic Regression](Resources/gradientDescentRegularizedLogistic.png)

# Week #4

---

## Motivations

### Non-linear Hypotheses

When working with cases where the number of features required to work with is high, trying to add non-linear features which are derived based on the set of linear features, will make the feature space grow non-linearly. For classification problems, using simple logistic regression together with quadratic or cubic set of features is not a good way to learn complex nonlinear hypotheses.

### Neurons and the Brain

Neural Networks have their origin in the idea of creating algorithms that try to mimic the brain. They were widely used in the 80s and early 90s, and lost popularity in the late 90s. Although nowadays, they are the state-of-the-art technique for many applications.

There is a "one learning algorithm" hypothesis based on neuro-rewiring process which claims that some parts of the brain can re-learn and understand the information that gets fed to another part of the brain. Therefore it seems that there aren't many different algorithms to work on different things, rather only one, "the learning algorithm". That is the best shot on which neural networks' main idea is built upon.

## Neural Networks

### Model Representation I

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (**dendrites**) as electrical inputs (called "spikes") that are channeled to outputs (**axons**). In our model, our dendrites are like the input features $x_{1} \cdots x_{n}$, and the output is the result of our hypothesis function. In this model our $x_{0}$ input node is sometimes called the "bias unit." It is always equal to $1$. In neural networks, we use the same logistic function as in classification, $\frac{1}{1 + e^{-\theta^{T}x}}$, yet we sometimes call it a sigmoid (logistic) **activation** function. In this situation, our "theta" parameters are sometimes called "weights".

Visually, a simplistic representation looks like:

$$
\begin{bmatrix}
x_{0} \\
x_{1} \\
x_{2}
\end{bmatrix} \to
\begin{bmatrix}
a_{1}
\end{bmatrix} \to
h_{\theta} (x)
$$

Our input nodes (layer $1$), also known as the "input layer", go into another node (layer $2$), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes $a_{0}^{2} \cdots a_{n}^{2}$ and call them "activation units".

- $a_{i}^{(j)}$ = "activation" of unit $i$ in layer $j$
- $\theta^{(j)}$ = matrix of weights controlling function mapping from layer $j$ to layer $j + 1$

If we had one hidden layer, it would look like:

$$
\begin{bmatrix}
x_{0} \\
x_{1} \\
x_{2} \\
x_{3}
\end{bmatrix} \to
\begin{bmatrix}
a_{1}^{(2)} \\
a_{2}^{(2)} \\
a_{3}^{(2)}
\end{bmatrix} \to
h_{\theta} (x)
$$

The values for each of the "activation" nodes is obtained as follows:

$$ a_{1}^{(2)} = g(\theta_{10}^{(1)}x_{0} + \theta_{11}^{(1)}x_{1} + \theta_{12}^{(1)}x_{2}) + \theta_{13}^{(1)}x_{3}) $$
$$ a_{2}^{(2)} = g(\theta_{20}^{(1)}x_{0} + \theta_{21}^{(1)}x_{1} + \theta_{22}^{(1)}x_{2}) + \theta_{23}^{(1)}x_{3}) $$
$$ a_{3}^{(2)} = g(\theta_{30}^{(1)}x_{0} + \theta_{31}^{(1)}x_{1} + \theta_{32}^{(1)}x_{2}) + \theta_{33}^{(1)}x_{3}) $$
$$ h_{\theta} (x) = a_{1}^{(3)} = g(\theta_{10}^{(2)}a_{0}^{(2)} + \theta_{11}^{(2)}a_{1}^{(2)} + \theta_{12}^{(2)}a_{2}^{(2)} + \theta_{13}^{(2)}a_{3}^{(2)}) $$

This is saying that we compute our activation nodes by using a $3 \times 4$ matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix $\Theta^{(2)}$ containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, $\Theta^{(j)}$.

The dimensions of these matrices of weights is determined as follows:

If network has $s_{j}$ units in layer $j$ and $s_{j + 1}$ in layer $j + 1$, then $\Theta^{(j)}$ will be of dimension $s_{j + 1} \times (s_{j} + 1)$.

The $+1$ comes from the addition in $\Theta^{(j)}$ of the "bias nodes", $x_{0}$ and $\Theta_{0}^{(j)}$. In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:

![Neural Network Representation](\Resources/neuralNetwork.png)

Example: If layer $1$ has $2$ input nodes and layer $2$ has $4$ activation nodes. Dimension of $\Theta^{(1)}$ is going to be $4 \times 3$ where $s_{j} = 2$ and $s_{j + 1} = 4$, so $s_{j + 1} \times (s_{j} + 1) = 4 \times 3$.

### Model Representation II

To re-iterate, the following is an example of a neural network:

$$ a_{1}^{(2)} = g(\theta_{10}^{(1)}x_{0} + \theta_{11}^{(1)}x_{1} + \theta_{12}^{(1)}x_{2}) + \theta_{13}^{(1)}x_{3}) $$
$$ a_{2}^{(2)} = g(\theta_{20}^{(1)}x_{0} + \theta_{21}^{(1)}x_{1} + \theta_{22}^{(1)}x_{2}) + \theta_{23}^{(1)}x_{3}) $$
$$ a_{3}^{(2)} = g(\theta_{30}^{(1)}x_{0} + \theta_{31}^{(1)}x_{1} + \theta_{32}^{(1)}x_{2}) + \theta_{33}^{(1)}x_{3}) $$
$$ h_{\theta} (x) = a_{1}^{(3)} = g(\theta_{10}^{(2)}a_{0}^{(2)} + \theta_{11}^{(2)}a_{1}^{(2)} + \theta_{12}^{(2)}a_{2}^{(2)} + \theta_{13}^{(2)}a_{3}^{(2)}) $$

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable $z_{k}^{(j)}$ that encompasses the parameters inside our $g$ function. In our previous example if we replaced all our parameters by their respective variable $z$, we would get:

$$ a_{1}^{(2)} = g(z_{1}^{(2)}) $$
$$ a_{2}^{(2)} = g(z_{2}^{(2)}) $$
$$ a_{3}^{(2)} = g(z_{3}^{(2)}) $$

In other words, for layer $j = 2$ and node $k$, the variable $z$ will be:

$$ z_{k}^{2} = \Theta_{k,0}^{(1)}x_{0} + \Theta_{k,1}^{(1)}x_{1} + \cdots + \Theta_{k,n}^{(1)}x_{n} $$

The vector representation of $x$ and $z^{j}$ is:

$$
x =
\begin{bmatrix}
x_{0} \\
x_{1} \\
\vdots \\
x_{n}
\end{bmatrix}
\hskip{0.5em}
z^{(j)} =
\begin{bmatrix}
z_{1}^{(j)} \\
z_{2}^{(j)} \\
\vdots \\
z_{n}^{(j)}
\end{bmatrix}$$

Setting $x = a^{(1)}$, we can rewrite the equation as:

$$ z^{(j)} = \Theta^{(j - 1)}a^{(j - 1)} $$

We are multiplying our matrix $\Theta^{(j - 1)}$ with dimensions $s_{j} \times (n + 1)$ (where $s_{j}$ is the number of our activation nodes) by our vector $a^{(j - 1)}$ with height ($n + 1$). This gives us our vector $z^{(j)}$ with height $s_{j}$. Now we can get a vector of our activation nodes for layer $j$ as follows:

$$ a^{(k)} = g(z^{(j)}) $$

Where our function $g$ can be applied element-wise to out vector $z^{(j)}$.

We can then add a bias unit (equal to $1$) to layer $j$ after we have computed $a^{(j)}$. This will be element $a_{0}^{(j)}$ and will be equal to $1$. To compute our final hypothesis, let's first compute another $z$ vector:

$$ z^{(j + 1)} = \Theta^{(j)}a^{(j)} $$

We get this final $z$ vector by multiplying the next theta matrix after $\Theta^{(j - 1)}$ with the values of all the activation nodes we just got. This last theta matrix $\Theta^{(j)}$ will have only **one row** which is multiplied by one column $a^{(j)}$ so that our result is a single number. We then get our final result with:

$$ h_{\theta} (x) = a^{(j + 1)} = g(z^{(j + 1)}) $$

Notice that in this **last step**, between layer $j$ and layer $j + 1$, we are doing **exactly the same thing** as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

## Applications

### Examples and Intuitions I

A simple example of applying neural networks is by predicting $x_{1}$ AND $x_{2}$, which is the logical 'and' operator and is only true if both $x_{1}$ and $x_{2}$ are $1$.

The graph of our functions will look like:

$$ 
\begin{bmatrix}
x_{0} \\
x_{1} \\
x_{2}
\end{bmatrix} \to
\begin{bmatrix}
g(z^{(2)})
\end{bmatrix} \to
h_{\theta} (x)
$$

Remember that $x_{0}$ is our bias variable and is always $1$.

Let's set our first theta matrix as:

$$ \Theta^{(1)} = 
\begin{bmatrix}
-30 & 20 & 20
\end{bmatrix}
$$

This will cause the output of our hypothesis to only be positive if both $x_{1}$ and $x_{2}$ are $1$. In other words:

$$ h_{\theta} (x) = g(-30 + 20x_{1} + 20x_{2})$$
$$ x_{1} = 0 \text{ and } x_{2} = 0 \text{ then } g(-30) \approx 0 $$
$$ x_{1} = 0 \text{ and } x_{2} = 1 \text{ then } g(-10) \approx 0 $$
$$ x_{1} = 1 \text{ and } x_{2} = 0 \text{ then } g(-10) \approx 0 $$
$$ x_{1} = 1 \text{ and } x_{2} = 1 \text{ then } g(10) \approx 1 $$

So we have constructed one of the fundamental operations in computers by using a small neural network rather than using an actual AND gate. Neural networks can also be used to simulate all the other logical gates. The following is an example of the logical operator 'OR', meaning either $x_{1}$ is true or $x_{2}$ is true, or both:

![Neural Network OR Function](Resources/neuralNetworkOR.png)

Where $g(z)$ is the following:

![Sigmoid Function](Resources/sigmoidFunction.png)

### Examples and Intuitions II

The $\Theta^{(1)}$ matrices for AND, NOR, and OR are:

$$ \text{AND: } \Theta^{(1)} = \begin{bmatrix} -30 & 20 & 20 \end{bmatrix} $$
$$ \text{NOR: } \Theta^{(1)} = \begin{bmatrix} 10 & -20 & -20 \end{bmatrix} $$
$$ \text{OR: } \Theta^{(1)} = \begin{bmatrix} -10 & 20 & 20 \end{bmatrix} $$

We can combine these to get the XNOR logical operator (which gives $1$ if $x_{1}$ and $x_{2}$ are both $0$ or both $1$).

$$
\begin{bmatrix}
x_{0} \\
x_{1} \\
x_{2}
\end{bmatrix} \to
\begin{bmatrix}
a_{1}^{(2)} \\
a_{2}^{(2)}
\end{bmatrix} \to
\begin{bmatrix}
a^{(3)}
\end{bmatrix} \to
h_{\theta} (x)
$$

For the transition between the first and second layer, we'll use a $\Theta^{(1)}$ matrix that combines the values for AND and NOR:

$$ \Theta^{(1)} = 
\begin{bmatrix}
-30 & 20 & 20 \\
10 & -20 & -20
\end{bmatrix}
$$

For the transition between the second and third layer, we'll use a $\Theta^{(2)}$ matrix that uses the value for OR:

$$ \Theta^{(2)} =
\begin{bmatrix}
-10 & 20 & 20
\end{bmatrix}
$$

Let's write out the value for all our nodes:

$$ a^{(2)} = g(\Theta^{(1)} \cdot x) $$
$$ a^{(3)} = g(\Theta^{(2)} \cdot a^{(2)}) $$
$$ h_{\theta} (x) = a^{(3)} $$

And there we have the XNOR operator using a hidden layer with two nodes! The following summarizes the above algorithm:

![Neural Network XNOR Function](Resources/neuralNetworkXNOR.png)

### Multiclass Classification

To classify data into multiple classes, we let our hypothesis function return a vector of values. Say we wanted to classify our data into one of four categories. We will use the following example to see how this classification is done. This algorithm takes as input an image and classifies it accordingly.

![Neural Network Multiclass Classification](Resources/neuralNetworkMulticlassClassification.png)

We can define our set of resulting classes as $y$:

$$ y^{(i)} =
\begin{bmatrix}
1 \\ 0 \\ 0 \\ 0
\end{bmatrix},
\begin{bmatrix}
0 \\ 1 \\ 0 \\ 0
\end{bmatrix},
\begin{bmatrix}
0 \\ 0 \\ 1 \\ 0
\end{bmatrix},
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 1
\end{bmatrix}
$$

Each $y^{(i)}$ represents a different image corresponding to either a car, pedestrian, truck, or motorcycle. The inner layers, each provide us with some new information which leads to our final hypothesis function. The setup looks like:

$$
\begin{bmatrix}
x_{0} \\ x_{1} \\ x_{2} \\ \vdots \\ x_{n}
\end{bmatrix} \to
\begin{bmatrix}
a_{0}^{(2)} \\ a_{1}^{(2)} \\ a_{2}^{(3)} \\ \vdots
\end{bmatrix} \to
\begin{bmatrix}
a_{0}^{(3)} \\ a_{1}^{(3)} \\ a_{2}^{(3)} \\ \vdots
\end{bmatrix} \to \cdots \to
\begin{bmatrix}
h_{\theta} (x)_{1} \\
h_{\theta} (x)_{2} \\
h_{\theta} (x)_{3} \\
h_{\theta} (x)_{4}
\end{bmatrix}
$$

Our resulting hypothesis for one set of inputs may look like:

$$ h_{\theta} (x) = \begin{bmatrix} 0 & 0 & 1 & 0 \end{bmatrix} $$

In which case our resulting class is the third one down, or $h_{\theta} (x)_{3}$, which represents the motorcycle.

# Week #5

---

## Cost Function and Backpropagation

### Cost Function

Let's first define a few variables that we will need to use:
- $L$ = total number of layers in the network
- $s_{l}$ = number of units (not counting bias unit) in layer $l$
- $K$ = number of output units/classes

Recall that in neural networks, we may have many output nodes. We denote $h_{\theta} (x)_{k}$ as being a hypothesis that results in the $k^{th}$ output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the cost function for regularized logistic regression was:

$$ J(\theta) = -\frac{1}{m} \sum_{i = 1}^{m} [y^{(i)} \log(h_{\theta} (x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta} (x^{(i)}))] + \frac{\lambda}{2m} \sum_{j = 1}^{n} \theta_{j}^{2} $$

For neural networks, it is going to be slightly more complicated:

$$ J(\theta) = -\frac{1}{m} \sum_{i = 1}^{m} \sum_{k = 1}^{K} \left[ y_{k}^{(i)}\log((h_{\theta} (x^{(i)})_{k}) + (1 - y_{k}^{(i)}) \log(1 - (h_{\theta} (x^{(i)}))_{k}) \right] + \frac{\lambda}{2m} \sum_{l = 1}^{L - 1} \sum_{i = 1}^{s_{l}} \sum_{j = 1}^{s_{l} + 1} (\theta_{j, i}^{(l)})^{2} $$

We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of our nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

Note:
- the double sum simply adds up the logistic regression costs calculated for eaach cell in the output layer
- the triple sum simply adds up the squares of all the individual $\theta$s in the entire network
- the $i$ in the triple sum does **not** refer to training example $i$

### Backpropagation Algorithm

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute: 

$$ \text{min}_{\theta} J(\theta) $$

That is, we want to minimize our cost function $J$ using an optimal set of parameters in $\Theta$. In this section we'll look at the equations we use to compute the partial derivative of $J(\theta)$:

$$ \frac{\partial}{\partial \Theta_{i,j}^{(l)}} J(\theta) $$

To do so, we use the following algorithm:

![Backpropagation Algorithm](Resources/backpropagationAlgo.png)

**Backpropagation Algorithm**

Given training set $$ \{ (x^{(l)}, y^{(l)}) \cdots (x^{(m)}, y^{(m)}) \} $$

- Set $\Lambda_{i,j}^{(l)} := 0$ for all $(l, i, j)$, (hence you end up having a matrix full of zeros)

For training example $t = 1$ to $m$:

1. Set $a^{(1)} := x^{(t)}$
2. Perform forward propagation to compute $a^{(l)}$ for $l = 2, 3, \ldots, L$

![Gradient Computation of one training example](/Resources/gradCompOneExample.png)

3. Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$

Where $L$ is our total number of layers and $a^{(L)}$ is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in $y$. To get the $\delta$ values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute $\delta^{(L - 1)}, \delta^{(L - 2)}, \ldots, \delta^{(2)}$ using $\delta^{(l)} = ((\Theta^{(l)})^{T}\delta^{(l + 1)}) .\cdot a^{(l)} .\cdot (1 - a^{(l)}) $

The delta values of layer $l$ are calculated by multiplying the delta values in the next layer with the $\Theta$ matrix of layer $l$. We then element-wise multiply that with a function called $g\prime$, or g-prime, which is the derivative of the activation function $g$ evaluated with the input values given by $z^{(l)}$.

The g-prime derivative terms can also be written out as:

$$ g\prime (z^{(l)}) = a^{(l)} .\cdot (1 - a^{(l)}) $$

5. $\Delta_{i,j}^{(l)} := \Delta_{i,j}^{(l)} + a_{j}^{(l)} \delta_{i}^{(l + 1)}$ or with vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l + 1)}(a^{(l)})^{T}$

Hence we update our new $\Delta$ matrix.

- $D_{i,j}^{(l)} := \frac{1}{m} (\Delta_{i,j}^{(l)} + \lambda\Theta_{i,j}^{(l)})$, if $j \neq 0$.
- $D_{i,j}^{(l)} := \frac{1}{m} \Delta_{i,j}^{(l)}$ if $j = 0$

The capital-delta matrix $D$ is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $\frac{\partial}{\partial\Theta_{ij}^{(l)}} J(\theta) = D_{ij}^{(l)}$

### Backpropagation Intuition

Recall that the cost function for a neural network is:

$$ J(\theta) = - \frac{1}{m} \sum_{t=1}^m\sum_{k=1}^K \left[ y^{(t)}_k \ \log (h_\Theta (x^{(t)}))_k + (1 - y^{(t)}_k)\ \log (1 - h_\Theta(x^{(t)})_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_l+1} ( \Theta_{j,i}^{(l)})^{2} $$

If we consider simple non-multiclass classification ($k = 1$) and disregard regularization, the cost is computed with:

$$ cost(t) = y^{(t)}\log(h_{\theta}(x^{(t)}) + (1 - y^{(t)})\log(1-h_{\theta}(x^{(t)})) $$

Intuitively, $\delta_{j}^{(l)}$ is the "error" for $a_{j}^{(l)}$ (unit $j$ in layer $l$). More formally, the delta values are actually the derivative of the cost function:

$$ \delta_{j}^{(l)} = \frac{\partial}{\partial \delta z_{j}^{(l)}} cost(t) $$

Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are. Let us consider the following neural network below and see how we could calculate some $\delta_{j}^{(l)}$:

![Forward Propagation](Resources/forwardPropagation.png)

In the image above, to calculate $\delta_{2}^{(2)}$, we multiply the weights $\Theta_{12}^{(2)}$ and $\Theta_{22}^{(2)}$ by their respective $\delta$ values found to the right of each edge. So we get $\delta_{2}^{(2)} = \Theta_{12}^{(2)} \cdot \delta_{1}^{(3)} + \Theta_{22}^{(2)} \cdot \delta_{2}^{(3)}$. To calcualte every single possible $\delta_{j}^{(l)}$, we could start from the right of our diagram. We can think of our edges as our $\Theta_{ij}$. Going from right to left, to calculate the value of $\delta_{j}^{(l)}$, you can just take the over all sum of each weight times the $\delta$ it is coming from. Hence, another example would be $\delta_{2}^{(3)} = \Theta_{12}^{(3)} \cdot \delta_{1}^{(4)}$.

## Backpropagation in Practice

### Implementation Note: Unrolling Parameters

With neural networks, we are working with sets of matrices:

$$  \Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}, \dots \newline D^{(1)}, D^{(2)}, D^{(3)}, \dots $$

In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

```matlab
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```

If the dimensions of Theta1 is $10 \times 11$, Theta2 is $10 \times 11$ and Theta3 is $1 \times 11$, then we can get back our original matrices from the "unrolled" versions as follows:

```matlab
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

To summarize:

![Unrolling Parameters](Resources/unrollingParameters.png)

### Gradient Checking

Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

$$ \frac{\partial}{\partial\Theta} J(\theta) \approx \frac{J(\Theta + \epsilon) - J(\Theta - \epsilon)}{2\epsilon} $$

With multiple theta matrices, we can approximate the derivative **with respect to** $\mathbb\Theta_{j}$ as follows:

$$ \frac{\partial}{\partial\Theta} J(\theta) \approx \frac{J(\Theta_{1}, \ldots, \Theta_{j} + \epsilon, \ldots, \Theta_{n}) - J(\Theta_{1}, \ldots, \Theta_{j} - \epsilon, \ldots, \Theta_{n})}{2\epsilon} $$ 

A small value for $\epsilon$ (epsilon) such as $\epsilon = 10^{-4}$, guarantees that the math works out properly. If the value for $\epsilon$ is too small, we can end up with numerical problems.

Hence we are only adding or subtracting epsilon to the $\Theta_{j}$ matrix. In Octave, we can do it as follows:

```matlab
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

We previously saw how to calculate the deltaVector. So once we compute our gradApprox vector, we can check that gradApprox = deltaVector.

Once you have verified **once** that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to compute gradApprox can be very slow.

### Random Initialization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our $\Theta$ matrices using the following method:

![Random Initialization in Neural Networks](Resources/randomInit.png)

Hence, we initialize each $\Theta_{ij}^{(l)}$ to a random value between $[-\epsilon, \epsilon]$. Using the above formula gurantees that we get the desired bound. The same procedure applies to all the $\Theta$'s. Below is some working code you could use to experiment.

```matlab
If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

$rand(x, y)$ is just a function that will initialize a matrix of random real numbers between $0$ and $1$.

Note: the epsilon used above is unrelated to the epsilon from Gradient Checking

### Putting It Together

First, pick a network architechture, choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.

- Number of input units - dimension of features $x^{(i)}$
- Number of output units - number of classes
- Number of hidden units per layer - usuallly more the better (must balance with cost of computation as it increases with more hidden units)
- Defaults - 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

**Training a Neural Network**

1. Randomly initialize the weights
2. Implement forward propagation to get $h_{\theta} (x^{(i)})$ for any $x^{(i)}$
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on every training example:

```matlab
for i = 1:m,
  Perform forward propagation and backpropagation using example (x(i),y(i))
  (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

The following image gives us an intuition of what is happening as we are implementing our neural network:

![Gradient Descent and Cost Function](Resources/gradientDescentCostFunc.png)

Ideally, you want $h_{\theta} (x^{(i)}) \approx y^{(i)}$. This will minimize our cost function. However, keep in mind that $J(\theta)$ is not convex and thus we can end up in a local minimum instead.

## Application of Neural Networks

### Autonomous Driving

A fun and historically important example of neural networks learning is using a neural network for autonomous driving.

[ALVINN: An Autonomous Land Vehicle In a Neural Network](https://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network.pdf)

# Week #6

---

## Evaluating a Learning Algorithm

### Evaluating a Hypothesis

### Model Selection and Train/Validation/Test Sets

## Bias vs. Variance

### Diagnosing Bias vs. Variance

### Regularization and Bias/Variance

### Learning Curves

### Deciding What to Do Next Revisited

## Building a Spam Classifier

### Prioritizing What to Work On

### Error Analysis

## Handling Skewed Data

### Error Metrics for Skewed Classes

### Trading Off Precision and Recall

## Using Large Data Sets

### Data For Machine Learning
