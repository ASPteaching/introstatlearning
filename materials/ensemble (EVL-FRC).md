\title{
Learning with Trees
}

\section{Introduction}

Decision trees have been around for a number of years. Their recent revival is due to the discovery that ensembles of slightly different trees tend to produce much higher accuracy on previously unseen data, a phenomenon known as generalization. Ensembles of trees will be discussed but let us focus first on individual trees.

\section{Supervised Learning}

Classification and Regression are supervised learning task. Learning set is of the form:

$$
\mathcal{L}=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{n}
$$

Classification (Two classes, binary)

$$
\mathbf{x} \in \mathbb{R}^{d}, \quad y \in\{-1,+1\}
$$

Regression

$$
\mathbf{x} \in \mathbb{R}^{d}, \quad y \in \mathbb{R}
$$

\section{Decision tree basics}

\section{Decision tree}

We can see Fig.1.

1. A tree is a set of nodes and edges organized in a hierarchical fashion. In contrast to a graph, in a tree there are no loops. Internal nodes are denoted with circles and terminal nodes with squares.

2. A decision tree is a tree where each split node stores a boolean test function to be applied to the incoming data. Each leaf stores the final answer (predictor)

\section{Training and testing decision trees}

\section{Basic notation}

We can see Fig.2.

1. Input data is represented as a collection of points in the $d$-dimensional space which are labeled by their feature responses. A general tree structure

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-02.jpg?height=409&width=434&top_left_y=733&top_left_x=531)

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-02.jpg?height=667&width=550&top_left_y=480&top_left_x=1034)

Figure 1:
![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-02.jpg?height=412&width=1128&top_left_y=1750&top_left_x=496)

Figure 2: 2. A decision tree is a hierarchical structure of connected nodes.

3. Training a decision tree involves sending all training data $\{v\}$ into the tree and optimizing the parameters of the split nodes so as to optimize a chosen cost function.

4. During testing, a split (internal) node applies a test to the input data $v$ and sends it to the appropriate child. The process is repeated until a leaf (terminal) node is reached (beige path).

\section{Tree-Growing Procedure}

We can see Fig.3.

In order to grow a classification tree, we need to answer four basic questions:

1. How do we choose the Boolean conditions for splitting at each node?

2. Which criterion should we use to split a parent node into its two child nodes?

3. How do we decide when a node become a terminal node (i.e., stop splitting)?

4. How do we assign a class to a terminal node?

\section{Choosing the best split for a Variable}

\section{Splitting Strategies}

At each node, the tree-growing algorithm has to decide on which variable it is "best" to split. We need to consider every possible split over all variables present at that node, then enumerate all possible splits, evaluate each one, and decide which is best in some sense.

For a description of splitting rules, we need to make a distinction between ordinal (or continuous) and nominal (or categorical) variables. For a continuous or ordinal variable, the number of possible splits at a given node is one fewer than the number of its distinctly observed values. Suppose that a particular categorical variable is defined by $m$ distinct categories, there are $2^{m-1}-1$ distinct splits.

We first need to choose the best split for a given variable. Accordingly, we have to measure of goodness of a split. Let $C_{1}, \ldots, C_{K}$ be the $K \geq 2$ classes. For node $\tau$, we denote by

$$
P\left(X \in C_{k} \mid \tau\right)
$$

the conditional probability that an observation $x$ is in $C_{k}$ given that it falls into the node $\tau$.

\section{Node information functions}

We can see Fig.4 
![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-04.jpg?height=1498&width=854&top_left_y=599&top_left_x=603)

Figure 3: 

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-05.jpg?height=761&width=1041&top_left_y=259&top_left_x=499)

Figure 4: Node impurity functions for the two-class case. The entropy function (rescaled) is the red curve, the Gini index is the green curve, and the resubstitution estimate of the misclassification rate is the blue curve.

To choose the best split over all variables, we first need to choose the best split for a given variable. Accordingly, we define a measure of goodness of a split. For node $\tau$, the node information function $i(\tau)$ is definded by

$$
i(\tau)=\phi(p(1 \mid \tau), \ldots, p(K \mid \tau))
$$

where $p(k \mid \tau)$ is an estimate of $P\left(X \in C_{k} \mid \tau\right)$

We require $\phi$ to be a symmetric function, defined on the set of all $K$-tuples of probabilities $\left(p_{1}, \ldots, p_{k}\right)$ with unit sum, minimized at the points $(1,0, \cdots, 0),(0,1, \cdots, 0),(0,0, \cdots, 1)$, and maximized at the point $\left(\frac{1}{K}, \frac{1}{K}, \cdots, \frac{1}{K}\right)$. One such function $\phi$ is the entropy function,

$$
i(\tau)=-\sum_{i=1}^{K} p(k \mid \tau) \log p(k \mid \tau)
$$

An alternative to the the entropy is the Gini index, given by

$$
i(\tau)=1-\sum_{i=1}^{K} p(k \mid \tau)^{2}
$$

And the misclassification rate

$$
i(\tau)=\sum_{i=1}^{K} p(k \mid \tau)(1-p(k \mid \tau))
$$



![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-06.jpg?height=236&width=458&top_left_y=386&top_left_x=476)

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-06.jpg?height=206&width=209&top_left_y=287&top_left_x=1164)
![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-06.jpg?height=438&width=670&top_left_y=282&top_left_x=932)

right

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-06.jpg?height=184&width=214&top_left_y=529&top_left_x=1389)

Figure 5:

\section{Information gain}

We can see Fig.5.

Suppose, at node $\tau$, we apply split $s$ so that a proportion $p_{L}$ of the observations drops down to the left child-node $\tau_{L}$ and the remaining proportion $p_{R}$ drops down to the right child-node $\tau_{R}$

The goodness of split $s$ at node $\tau$ is given by the reduction in impurity gained by splitting the parent node $\tau$ into its child nodes, $\tau_{L}$ and $\tau_{R}$,

$$
I(s, \tau)=i(\tau)-p_{L} i\left(\tau_{L}\right)-p_{R} i\left(\tau_{R}\right)
$$

The best split for the single variable $X_{j}$ is the one that has the largest value of $I(s, \tau)$ over all $s \in \mathcal{S}_{j}$, the set of possible distinct splits for $X_{j}$.

\section{Example}

\begin{tabular}{llll} 
& +1 & -1 & Total \\
\hline$X_{j} \leq s$ & $n_{11}$ & $n_{12}$ & $n_{1+}$ \\
$X_{j}>s$ & $n_{21}$ & $n_{22}$ & $n_{2+}$ \\
\hline Total & $n_{+1}$ & $n_{+2}$ & $n_{++}$ \\
\hline
\end{tabular}

Consider first, the parent node $\tau$, estimate $P\left(\frac{+1}{\tau}\right)$ by $n_{+1} / n_{++}$and $P\left(\frac{-1}{\tau}\right)$ by $n_{+2} / n_{++}$, and then the estimated impurity function is:

$$
i(\tau)=-\left(\frac{n_{+1}}{n_{++}}\right) \log \left(\frac{n_{+1}}{n_{++}}\right)-\left(\frac{n_{+2}}{n_{++}}\right) \log \left(\frac{n_{+2}}{n_{++}}\right)
$$

Note that $i(\tau)$ is completely independent of the type of proposed split.

Now, for the child nodes, $\tau_{L}$ y $\tau_{R}$. We estimated $p_{L}=n_{1+} / n_{++}$and $p_{R}=n_{2+} / n_{++}$ For $X_{j} \leq s$, we estimate $P\left(\frac{+1}{\tau_{L}}\right)=n_{11} / n_{1+}$ and $P\left(\frac{-1}{\tau_{L}}\right)=n_{12} / n_{1+}$ For the condition $X_{j}>s$, we estimate $P\left(\frac{+1}{\tau_{R}}\right)=n_{21} / n_{2+}$ and $P\left(\frac{-1}{\tau_{R}}\right)=n_{22} / n_{2+}$. We then compute:

$$
\begin{aligned}
& i\left(\tau_{L}\right)=-\left(\frac{n_{11}}{n_{1+}}\right) \log \left(\frac{n_{11}}{n_{1+}}\right)-\left(\frac{n_{12}}{n_{1+}}\right) \log \left(\frac{n_{12}}{n_{1+}}\right) \\
& i\left(\tau_{R}\right)=-\left(\frac{n_{21}}{n_{2+}}\right) \log \left(\frac{n_{21}}{n_{2+}}\right)-\left(\frac{n_{22}}{n_{2+}}\right) \log \left(\frac{n_{22}}{n_{2+}}\right)
\end{aligned}
$$

and, then we can compute $I(s, \tau)$.

\section{Recursive Partitioning}

- In order to grow a tree, we start with the root node, which consists of the learning set L. Using the "goodness-of-split" criterion for a single variable, the tree algorithm finds the best split at the root node for each of the variables, $X_{1}$ to $X_{r}$.

- The best split $s$ at the root node is then defined as the one that has the largest value of over all $r$ single-variable best splits at that node.

- We next split each of the child nodes of the root node in the same way. We repeat the above computations for the left child node, except that we consider only those observations that fall into each child node.

- When those splits are completed, we continue to split each of the subsequent nodes. This sequential splitting process of building a tree layer-by-layer is called recursive partitioning.

\section{Stop spliting}

- For example, we can declare a node to be terminal if it fails to be larger than a certain critical size ( $n_{\min }$ is some previously declared minimum size of a node).

- Another early action was to stop a node from splitting if the largest goodness-of-split value at that node is smaller than a certain predetermined limit.

- These stopping rules, however, do not turn out to be such good ideas. A better approach (Breiman et al. 2008) is to let the tree grow to saturation and then "prune" it back.

\section{Plurality rule}

We can see Fig.6.

How do we associate a class with a terminal node?

Suppose at terminal node $\tau$ there are $n(\tau)$ observations, of which $n_{k}(\tau)$ are from class $C_{k}$, $k=1, \ldots, K$. Then, the class which corresponds to the largest of the $\left\{n_{1}(\tau), \ldots, n_{k}(\tau)\right\}$ is assigned to $\tau$. 

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-08.jpg?height=477&width=1220&top_left_y=241&top_left_x=447)

Figure 6 :

This is called the plurality rule. This rule can be derived from the Bayes's rule classifier, where we assign the node $\tau$ to class $C_{i}$ if $p(i \mid \tau)=\max _{k} p(k \mid \tau)$.

\section{Choosing the best split in Regression Trees}

We now discuss the process of building a regression tree. Roughly speaking, there are two steps.

1. We divide the predictor space-that is, the set of possible values for $X_{1}, X_{2}, \ldots, X_{p}$ into $J$ distinct and non-overlapping regions, $R_{1}, R_{2}, \ldots, R_{J}$.

2. For every observation that falls into the region $R_{j}$, we make the same prediction, which is simply the mean of the response values for the training observations in $R_{j}$.

For instance, suppose that in Step 1 we obtain two regions, $R_{1}$ and $R_{2}$, and that the response mean of the training observations in the first region is 10 , while the response mean of the training observations in the second region is 20 . Then for a given observation $X=x$, if $x \in R_{1}$ we will predict a value of 10 , and if $x \in R_{2}$ we will predict a value of 20 .

We now elaborate on Step 1 above. How do we construct the regions $R_{1}, \ldots, R_{J}$ ? In theory, the regions could have any shape. However, we choose to divide the predictor space into high-dimensional rectangles, or boxes, for simplicity and for ease of interpretation of the resulting predictive model. The goal is to find boxes $R_{1}, \ldots, R_{J} ?$ that minimize the $\mathrm{RSS}$ (Residual Sum of Squares), given by

$$
\sum_{j=1}^{J} \sum_{i \in R_{j}}\left(y_{i}-\hat{y}_{R_{j}}\right)^{2}
$$

where $\hat{y}_{R_{j}}$ is the mean response for the training observations within the $\mathrm{j}$-th box. Unfortunately, it is computationally infeasible to consider every possible partition of the feature space into $J$ boxes. For this reason, we take a top-down, greedy approach that is known as recursive binary splitting. The approach is top-down because it begins at the top of the tree (at which point all observations belong to a single region) and then successively splits the predictor space; each split is indicated via two new branches further down on the tree. It is greedy because at each step of the tree-building process, the best split is made at that particular step, rather than looking ahead and picking a split that will lead to a better tree in some future step.

In order to perform recursive binary splitting, we first select the predictor $X_{j}$ and the cutpoint $s$ such that splitting the predictor space into the regions $\left\{X \mid X_{j} \leq s\right\}$ and $\left\{X \mid X_{j}>s\right\}$ leads to the greatest possible reduction in RSS. (The notation $\left\{X \mid X_{j} \leq s\right\}$ means the region of predictor space in which $X_{j}$ takes on a value less than $s$ ). That is, we consider all predictors $X_{1}, \ldots, X_{p}$, and all possible values of the cutpoint $s$ for each of the predictors, and then choose the predictor and cutpoint such that the resulting tree has the lowest $\mathrm{RSS}$. In greater detail, for any $j$ and $s$, we define the pair of half-planes

$$
R_{1}(j, s)=\left\{X \mid X_{j} \leq s\right\}, \quad R_{2}(j, s)=\left\{X \mid X_{j}>s\right\}
$$

and we seek the value of $j$ and $s$ that minimize the equation

$$
\sum_{i: x_{i} \in R_{1}}\left(y_{i}-\hat{y}_{R_{1}}\right)^{2}+\sum_{i: x_{i} \in R_{2}}\left(y_{i}-\hat{y}_{R_{2}}\right)^{2}
$$

where $\hat{y}_{R_{1}}$ is the mean response for the training observations in $R_{1}(j, s)$, and $\hat{y}_{R_{2}}$ is the mean response for the training observations in $R_{2}(j, s)$. Finding the values of $j$ and $s$ that minimize (1) can be done quite quickly, especially when the number of features $p$ is not too large.

Next, we repeat the process, looking for the best predictor and best cutpoint in order to split the data further so as to minimize the RSS within each of the resulting regions. However, this time, instead of splitting the entire predictor space, we split one of the two previously identified regions. We now have three regions. Again, we look to split one of these three regions further, so as to minimize the $\mathrm{RSS}$. The process continues until a stopping criterion is reached; for instance, we may continue until no region contains more than five observations.

Once the regions $R_{1}, \ldots, R_{J}$ have been created, we predict the response for a given test observation using the mean of the training observations in the region to which that test observation belongs.

\section{Estimating the misclassification rate}

Let $T$ be the tree classifier and let $\tilde{T}=\left\{\tau_{1}, \tau_{2}, \ldots, \tau_{L}\right\}$ denote the set of all terminal nodes of T. We can now estimate the true misclassification rate,

$$
R(T)=\sum_{l=1}^{L} R\left(\tau_{l}\right) P\left(\tau_{l}\right),
$$

for $T$, where $P(\tau)$ is the probability that an observation falls into node $\tau$ and $R(\tau)$ is the within-node misclassification rate of an observation in node $\tau$. If we estimate $R(\tau)$ by

$$
r(\tau)=1-\max _{k} p(k \mid \tau)
$$

and we estimate $P\left(\tau_{l}\right)$ by the proportion $p\left(\tau_{l}\right)$ of all observations that fall into node $\tau_{l}$, then, the resubstitution estimate of $R(T)$ is

$$
\hat{R}(T)=\sum_{l=1}^{L} r\left(\tau_{l}\right) p\left(\tau_{l}\right)
$$

The resubstitution estimate $\hat{R}(T)$, however, leaves much to be desired as an estimate of $R(T)$. First, bigger trees (i.e., more splitting) have smaller values of $\hat{R}(T)$; that is, $\hat{R}\left(T^{\prime}\right) \leq \hat{R}(T)$ ), where $T^{\prime}$ is formed by splitting a terminal node of $T$. For example, if a tree is allowed to grow until every terminal node contains only a single observation, then that node is classified by the class of that observation and $\hat{R}(T)=0$.

Second, using only the resubstitution estimate tends to generate trees that are too big for the given data.

Third, the resubstitution estimate $\hat{R}(T)$ is a much-too-optimistic estimate of $R(T)$. More realistic estimates of $R(T)$ are given below.

\section{Tree Pruning}

Since decision trees have a very high tendency to over-fit the data, a smaller tree with fewer splits might lead to increase the generalization capability. Lower variance (estimation error) at the cost of a little bias (approximation error).

One possible alternative to the process described above is to build the tree only so long as the decrease in the node impurity measure, due to each split exceeds some (high) threshold. However, due to greedy nature of the splitting algorithm, it is too short-sighted since a seemingly worthless split early on in the tree might be followed by a very good split i.e., a split that leads to a large reduction in impurity later on.

Therefore, a better strategy is to grow a very large tree $T_{0}$, and then prune it back in order to obtain a subtree.

\section{Cost complexity pruning}

A sequence of trees indexed by a nonnegative tuning parameter $\alpha$ is considered. For each value of $\alpha$ there corresponds a subtree $T \subset T_{0}$ such that the penalized misclassification rate

$$
R_{\alpha}(T)=R(T)+\alpha|T|
$$

is as small as possible. Here $|T|$ indicates the number of terminal nodes of the subtree $T$, Think of $\alpha|T|$ as a penalty term for tree size, so that $R_{\alpha}(T)$ penalizes $R(T)$ (2) for generating too large a tree. For each $\alpha$, we then choose that subtree $T(\alpha)$ of $T_{0}$ that minimizes $R_{\alpha}(T)$. The tuning parameter $\alpha$ controls a trade-off between the subtree's complexity and its fit to the training data. When $\alpha=0$, then the subtree $T$ will simply equal $T_{0}$. As $\alpha$ increases, there is a price to pay for having a tree with many terminal nodes, and so the above equation will tend to be minimized for a smaller subtree.

Breiman et al. (1984) showed that for every $\alpha$, there exists a unique smallest minimizing subtree.

Depending on the cost of each additional leaf (i.e. the $\alpha$ value) different sub-trees of $T_{0}$ minimise the error-complexity measure. Breiman and his colleagues proved that although $\alpha$ can run through a continuum of values there is a sequence of pruned trees such that each element is optimal for a range of $\alpha$, and so there is only a finite number of interesting $\alpha$ values.

$$
0=\alpha_{0}<\alpha_{1}<\alpha_{2}<\alpha_{3}<\cdots<\alpha_{M}
$$

Furthermore, they developed an algorithm that generates a parametric family of pruned trees

$$
T_{0} \prec T_{1} \prec T_{2} \prec T_{3} \prec \cdots \prec T_{M}
$$

such that each $T_{i}$ in the sequence is characterised by adifferent value $\alpha_{i}$. They proved that each tree $T_{i}$ in this sequence is optimal from the error-complexity perspective within the interval $\left[\alpha_{i}, \alpha_{i+1}\right)$.

So far, we have constructed a finite sequence of decreasing-size subtrees $T_{1}, T_{2}, T_{3}, \ldots, T_{M}$ by pruning more and more nodes from $T_{0}$. When do we stop pruning? Which subtree of the sequence do we choose as the "best" pruned subtree? Choice of the best subtree depends upon having a good estimate of the misclassification rate $R\left(T_{k}\right)$ corresponding to the subtree $T_{k}$. Breiman et al. (1984) offered two estimation methods: use an independent test sample or use cross-validation. When the data set is very large, use of an independent test set is straightforward and computationally efficient, and is, generally, the preferred estimation method. For smaller data sets, crossvalidation is preferred.

\section{Advantages and disadvantages of trees}

1. Trees are very easy to explain to people. In fact, they are even easier to explain than linear regression!

2. Some people believe that decision trees more closely mirror human decision-making than do the regression and classification approaches.

3. Trees can be displayed graphically, and are easily interpreted even by a non-expert (especially if they are small).

4. Trees can easily handle qualitative predictors without the need to create dummy variables.

5. Unfortunately, trees generally do not have the same level of predictive accuracy as some of the other regression and classification approaches. 

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-12.jpg?height=209&width=702&top_left_y=237&top_left_x=451)

a) Low randomness, high tree correlation

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-12.jpg?height=208&width=525&top_left_y=243&top_left_x=1147)

b) High randomness, low tree correlation

Figure 7:

However, by aggregating many decision trees, using methods like bagging, random forests, and boosting, the predictive performance of trees can be substantially improved. We introduce these concepts next.

\section{The randomness model}

A key aspect of decision forests is the fact that its component trees are all randomly different from one another (Fig. 7). This leads to de-correlation between the individual tree predictions and, in turn, to improved generalization. Forest randomness also helps achieve high robustness with respect to noisy data.

Randomness is injected into the trees during the training phase. Two of the most popular ways of doing so are:

- random training data set sampling (bagging approach) and

- randomized node optimization.

These two techniques are not mutually exclusive and could be used together.

\section{The ensemble model}

In a forest with $T$ trees all trees are trained independently (and possibly in parallel) (Fig. 8). During testing, each test point $v$ is simultaneously pushed through all trees (starting at the root) until it reaches the corresponding leaves. Tree testing can also often be done in parallel, thus achieving high computational efficiency on modern parallel CPU or GPU hardware. Combining all tree predictions into a single forest prediction may be done by a simple averaging operation.

\section{Random Forest}

\section{Algorithm}

1. Given training data set $\mathcal{L}=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}_{i=1}^{n}$. Fix $m \leq p(m=\sqrt{p})$ and the number of trees $B$.

2. For $b=1, \ldots, B$, do the following. 

![](https://cdn.mathpix.com/cropped/2023_03_29_31aa6b00cf3273f98667g-13.jpg?height=393&width=1054&top_left_y=264&top_left_x=514)

Figure 8 :

(a) Create a bootstrap version of the training data $\mathcal{L}_{b}^{*}$, by randomly sampling the $n$ rows with replacement $n$ times. The sample can be represented by the bootstrap frequency vector $w_{b}^{*}$.

(b) Grow a maximal-depth tree $T_{b}$ using the data in $\mathcal{L}_{b}^{*}$, sampling $m$ of the $p$ features at random prior to making each split.

(c) Save the tree, as well as the bootstrap sampling frequencies for each of the training observations.

3. Output the ensembles of the trees $\left\{T_{b}\right\}_{1}^{B}$

To make a prediction at a new point $\mathbf{x}$ :

Regression: Compute the random-forest fit at any prediction point as the average

$$
\hat{f}_{\mathrm{rf}}(\mathbf{x})=\frac{1}{B} \sum_{b=1}^{B} T_{b}(\mathbf{x})
$$

Classification: Let $\hat{C}_{b}(\mathbf{x})$ be the class prediction of the $b$-th random forest tree. Then $\hat{C}_{r f}(\mathbf{x})=$ majority vote $\left\{\hat{C}_{b}(\mathbf{x})\right\}_{1}^{B}$

\section{Out-of-Bag error estimates}

Random forests deliver cross-validated error estimates at virtually no extra cost. The idea is similar to the bootstrap error estimates.

In making the prediction for observation pair $\left(x_{i}, y_{i}\right)$, we average all the random-forest trees $T_{b}\left(x_{i}\right)$ for which that pair is not in the corresponding bootstrap sample:

$$
\hat{f}_{\mathrm{rf}}^{(i)}(\mathbf{x})=\frac{1}{B_{i}} \sum_{b: w_{b i}^{*}=0}^{B_{i}} T_{b}(\mathbf{x})
$$

where $B_{i}$ is the number of times observation $i$ was not in the bootstrap sample. We then compute the $\mathrm{OOB}$ error estimate

$$
\operatorname{err}_{O O B}=\frac{1}{n} \sum_{i=1}^{n} L\left(y_{i}, \hat{f}_{r f}^{(i)}\left(\mathbf{x}_{i}\right)\right)
$$

where $L$ is the loss function of interest, such as misclassification or squarederror loss. If $B$ is sufficiently large, we can see that the OOB error estimate is equivalent to leave-one-out cross-validation error.

\section{Assessing variable importance}

Computations are carried out one tree at a time.

As before, let $T_{b}$ be the tree classifier constructed from the bootstrap sample $\mathcal{L}_{b}^{*}$. First, drop the $O O B$ observations corresponding to $\mathcal{L}_{b}^{*}$ down the tree $T_{b}$, record the resulting classifications, and compute the $O O B$ error rate, $P E_{b}(O O B)$. Next, randomly permute the $O O B$ values on the $j$-th variable $X_{j}$ while leaving the data on all other variables unchanged. If $X_{j}$ is important, permuting its observed values will reduce our ability to classify successfully each of the $O O B$ observations. Then, we drop the altered $O O B$ observations down the tree $T_{b}$, record the resulting classifications, and compute the $O O B$ error rate, $P E_{b}\left(O O B_{j}\right)$, which should be larger than the error rate of the unaltered data. $\mathrm{A}$ raw $T_{b}$-score for $X_{j}$ can be computed by the difference between those two $O O B$ error rates,

$$
\operatorname{raw}_{b}(j)=P E_{b}\left(O O B_{j}\right)-P E_{b}(O O B) \quad b=1, \ldots, B
$$

Finally, average the raw scores over all the $B$ trees in the forest,

$$
\operatorname{imp}\left(X_{j}\right)=\frac{1}{B} \sum_{b=1}^{B} \operatorname{raw}_{b}(j)
$$

to obtain an overall measure of the importance of $X_{j}$. Call this measure the raw permutation accuracy importance score for the $j$-th variable. We calculate the score for each variable, and at the end of the procedure we rank variables according to the score.

\section{Bibliography}

A. Criminisi, J. Shotton and E. Konukoglu. Decision Forests for Classification, Regression, Density Estimation, Manifold Learning and Semi-Supervised Learning. Microsoft Research technical report TR-2011-114.

The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Second Edition. Springer. 2009