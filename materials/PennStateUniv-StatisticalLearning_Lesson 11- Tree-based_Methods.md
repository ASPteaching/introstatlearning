\title{
Lesson 11: Tree-based Methods
}

\section{Lesson 11: Tree-based Methods}

\section{Overview}

Textbook reading: Chapter 8: Tree-Based Methods.

Decision trees can be used for both regression and classification problems. Here we focus on classification trees. Classification trees are a very different approach to classification than prototype methods such as k-nearest neighbors. The basic idea of these methods is to partition the space and identify some representative centroids.

They also differ from linear methods, e.g., linear discriminant analysis, quadratic discriminant analysis, and logistic regression. These methods use hyperplanes as classification boundaries.

Classification trees are a hierarchical way of partitioning the space. We start with the entire space and recursively divide it into smaller regions. In the end, every region is assigned to a class label.

\section{Tree-Structured Classifier}

The following textbook presents Classification and Regression Trees (CART) :

Reference: Classification and Regression Trees by L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone, Chapman \& Hall, 1984.

Let's start with a medical example to get a rough idea about classification trees.

\section{A Medical Example}

One big advantage of decision trees is that the classifier generated is highly interpretable. For physicians, this is an especially desirable feature.

In this example, patients are classified into one of two classes: high risk versus low risk. It is predicted that the high-risk patients would not survive at least 30 days based on the initial 24-hour data. There are 19 measurements taken from each patient during the first 24 hours. These include blood pressure, age, etc.

Here a tree-structured classification rule is generated and can be interpreted as follows: 

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-02.jpg?height=642&width=733&top_left_y=113&top_left_x=136)

First, we look at the minimum systolic blood pressure within the initial 24 hours and determine whether it is above 91. If the answer is no, the patient is classified as high-risk. We don't need to look at the other measurements for this patient. If the answer is yes, then we can't make a decision yet. The classifier will then look at whether the patient's age is greater than 62.5 years old. If the answer is no, the patient is classified as low risk. However, if the patient is over 62.5 years old, we still cannot make a decision and then look at the third measurement, specifically, whether sinus tachycardia is present. If the answer is no, the patient is classified as low risk. If the answer is yes, the patient is classified as high risk.

Only three measurements are looked at by this classifier. For some patients, only one measurement determines the final result. Classification trees operate similarly to a doctor's examination.

\section{Objectives}

Upon successful completion of this lesson, you should be able to:

- Understand the basic idea of decision trees.

- Understand the three elements in the construction of a classification tree.

- Understand the definition of the impurity function and several example functions.

- Know how to estimate the posterior probabilities of classes in each tree node.

- Understand the advantages of tree-structured classification methods.

- Understand the resubstitution error rate and the cost-complexity measure, their differences, and why the cost-complexity measure is introduced.

- Understand weakest-link pruning.

- Understand the fact that the best-pruned subtrees are nested and can be obtained recursively.

- Understand the method based on cross-validation for choosing the complexity parameter and the final subtree.

- Understand the purpose of model averaging.

- Understand the bagging procedure.

- Understand the random forest procedure.

- Understand the boosting approach.

\section{1 - Construct the Tree}

11.1 - Construct the Tree 

\section{Notation}

We will denote the feature space by $\mathbf{X}$. Normally $\mathbf{X}$ is a multidimensional Euclidean space. However, sometimes some variables (measurements) may be categorical such as gender, (male or female).

CART has the advantage of treating real variables and categorical variables in a unified manner. This is not so for many other classification methods, for instance, LDA.

The input vector is indicated by $X \in \mathbf{X}$ contains $p$ features $X_{1}, X_{2}, \cdots, X_{p}$.

Tree-structured classifiers are constructed by repeated splits of the space $X$ into smaller and smaller subsets, beginning with $\mathbf{X}$ itself.

We will also need to introduce a few additional definitions:

Rollover the definitions on the right in the interactive image below:

node, terminal node (leaf node), parent node, child node.

One thing that we need to keep in mind is that the tree represents the recursive splitting of the space. Therefore, every node of interest corresponds to one region in the original space. Two child nodes will occupy two different regions and if we put the two together, we get the same region as that of the parent node. In the end, every leaf node is assigned with a class and a test point is assigned with the class of the leaf node it lands in.

\section{Additional Notation:}

- A node is denoted by $t$. We will also denote the left child node by $t_{L}$ and the right one by $t_{R}$.

- Denote the collection of all the nodes in the tree by $T$ and the collection of all the leaf nodes by $\tilde{T}$

- A split will be denoted by $s$. The set of splits is denoted by $S$.

Let's take a look at how these splits can take place.

The whole space is represented by $X$.
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-03.jpg?height=704&width=940&top_left_y=1852&top_left_x=114)

\begin{tabular}{|l|l|l|}
\hline & & $\mathrm{X}_{7}$ \\
\cline { 1 - 1 } $\mathrm{X}_{3}$ & \multirow{2}{*}{$\mathrm{X}_{5}$} & $\mathrm{x}_{8}$ \\
\cline { 1 - 1 } & & \\
\hline
\end{tabular}
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-04.jpg?height=880&width=1400&top_left_y=91&top_left_x=108)

\section{The Three Elements}

The construction of a tree involves the following three general elements:

1. The selection of the splits, i.e., how do we decide which node (region) to split and how to split it?

2. If we know how to make splits or 'grow' the tree, how do we decide when to declare a node terminal and stop splitting?

3. We have to assign each terminal node to a class. How do we assign these class labels?

In particular, we need to decide upon the following:

1. The pool of candidate splits that we might select from involves a set $Q$ of binary questions of the form $\{\mathrm{ls} \mathbf{x} \in A$ ?\}, $A \subseteq \mathbf{X}$. Basically, we ask whether our input $\mathbf{x}$ belongs to a certain region, $A$. We need to pick one $A$ from the pool.

2. The candidate split is evaluated using a goodness of split criterion $\Phi(s, t)$ that can be evaluated for any split $s$ of any node $t$.

3. A stop-splitting rule, i.e., we have to know when it is appropriate to stop splitting. One can 'grow' the tree very big. In an extreme case, one could 'grow' the tree to the extent that in every leaf node there is only a single data point. Then it makes no sense to split any farther. In practice, we often don't go that far.

4. Finally, we need a rule for assigning every terminal node to a class.

Now, let's get into the details for each of these four decisions that we have to make...

\section{1) Standard Set of Questions for Suggesting Possible Splits}

Let's say that the input vector $X=\left(X_{1}, X_{2}, \cdots, X_{p}\right)$ contains features of both categorical and ordered types. CART makes things simple because every split depends on the value of only a single variable. If we have an ordered variable - for instance, $X_{j}$ - the question inducing the split is whether $X_{j}$ is smaller or equal to some threshold? Thus, for each ordered variable $X_{j}, Q$ includes all questions of the form:

$$
\text { Is } X_{j} \leq c ?
$$

for all real-valued $c$

There are other ways to partition the space. For instance, you might ask whether $X_{1}+X_{2}$ is smaller than some threshold. In this case, the split line is not parallel to the coordinates. However, here we restrict our interest to the questions of the above format. Every question involves one of $X_{1}, \cdots, X_{p}$ , and a threshold.

Since the training data set is finite, there are only finitely many thresholds $c$ that results in a distinct division of the data points.

If $X_{j}$ is categorical, say taking values from $\{1,2, \ldots, M\}$, the questions $Q$, are of the form:

$$
\text { Is } X_{j} \in A ?
$$

where $A$ is any subset of $\{1,2, \ldots, M\}$

The splits or questions for all $p$ variables form the pool of candidate splits.

This first step identifies all the candidate questions. Which one to use at any node when constructing the tree is the next question ...

\section{2) Determining Goodness of Split}

The way we choose the question, i.e., split, is to measure every split by a 'goodness of split' measure, which depends on the split question as well as the node to split. The 'goodness of split' in turn is measured by an impurity function.

Intuitively, when we split the points we want the region corresponding to each leaf node to be "pure", that is, most points in this region come from the same class, that is, one class dominates.

Look at the following example. We have two classes shown in the plot by $\mathbf{x}$ 's and o's. We could split first by checking whether the horizontal variable is above or below a threshold, shown below:

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-05.jpg?height=297&width=343&top_left_y=2190&top_left_x=114)

The split is indicated by the blue line. Remember by the nature of the candidate splits, the regions are always split by lines parallel to either coordinate. For the example split above, we might consider it a good split because the left-hand side is nearly pure in that most of the points belong to the $\mathbf{x}$ class. Only two points belong to the $o$ class. The same is true of the right-hand side. Generating pure nodes is the intuition for choosing a split at every node. If we go one level deeper down the tree, we have created two more splits, shown below:

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-06.jpg?height=297&width=345&top_left_y=248&top_left_x=110)

now you see that the upper left region or leaf node contains only the $\mathbf{x}$ class. Therefore, it is $100 \%$ pure, no class blending in this region. The same is true for the lower right region. It only has o's. Once we have reached this level, it is unnecessary to further split because all the leaf regions are $100 \%$ pure. Additional splits will not make the class separation any better in the training data, although it might make a difference with the unseen test data.

\section{2 - The Impurity Function}

\section{2 - The Impurity Function}

The impurity function measures the extent of purity for a region containing data points from possibly different classes. Suppose the number of classes is $K$. Then the impurity function is a function of $p_{1}, \cdots, p_{K}$, the probabilities for any data point in the region belonging to class $1,2, \ldots, K$. During training, we do not know the real probabilities. What we would use is the percentage of points in class 1 , class 2 , class 3 , and so on, according to the training data set.

The impurity function can be defined in different ways, but the bottom line is that it satisfies three properties.

Definition: An impurity function is a function $\Phi$ defined on the set of all $K$-tuples of numbers $\left(p_{1}, \cdots, p_{K}\right)$ satisfying $p_{j} \geq 0, \quad j=1, \cdots, K, \Sigma_{j} p_{j}=1$ with the properties:

1. $\Phi$ achieves maximum only for the uniform distribution, that is all the $p_{j}$ are equal.

2. $\Phi$ achieves minimum only at the points $(1,0, \ldots, 0),(0,1,0, \ldots, 0), \ldots,(0,0, \ldots, 0,1)$, i.e., when the probability of being in a certain class is 1 and 0 for all the other classes.

3. $\Phi$ is a symmetric function of $p_{1}, \cdots, p_{K}$, i.e., if we permute $p_{j}, \Phi$ remains constant.

Definition: Given an impurity function $\Phi$, define the impurity measure, denoted as $i(t)$, of a node $t$ as follows:

$$
i(t)=\phi(p(1 \mid t), p(2 \mid t), \ldots, p(K \mid t))
$$

where $p(j \mid t)$ is the estimated posterior probability of class $j$ given a point is in node $t$. This is called the impurity function or the impurity measure for node $t$.

Once we have $i(t)$, we define the goodness of split $s$ for node $t$, denoted by $\Phi(s, t)$ :

$$
\Phi(s, t)=\Delta i(s, t)=i(t)-p_{R} i\left(t_{R}\right)-p_{L} i\left(t_{L}\right)
$$

$\Delta i(s, t)$ is the difference between the impurity measure for node $t$ and the weighted sum of the impurity measures for the right child and the left child nodes. The weights, $p_{R}$ and $p_{L}$, are the proportions of the samples in node $t$ that go to the right node $t_{R}$ and the left node $t_{L}$ respectively.

Look at the graphic again. Suppose the region to the left (shaded purple) is the node being split, the upper portion is the left child node the lower portion is the right child node. You can see that the proportion of points that are sent to the left child node is $p_{L}=8 / 10$. The proportion sent to the right child is $p_{R}=2 / 10$.

The classification tree algorithm goes through all the candidate splits to select the

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-07.jpg?height=305&width=331&top_left_y=316&top_left_x=1636)
best one with maximum $\Delta i(s, t)$.

Next, we will define $I(t)=i(t) p(t)$, that is, the impurity function of node $t$ weighted by the estimated proportion of data that go to node $t . p(t)$ is the probability of data falling into the region corresponding to node $t$. A simple way to estimate it is to count the number of points that are in node $t$ and divide it by the total number of points in the whole data set.

The aggregated impurity measure of tree $T, I(T)$ is defined by

$$
I(T)=\sum_{t \in \tilde{T}} I(t)=\sum_{t \in \tilde{T}} i(t) p(t)
$$

This is a sum over all of the leaf nodes. (Remember, not all of the nodes in the tree, just the leaf nodes: $\tilde{T}$.)

Note for any node $t$ the following equations hold:

$p\left(t_{L}\right)+p\left(t_{R}\right)=p(t)$

$p_{L}=p\left(t_{L}\right) / p(t)$

$p_{R}=p\left(t_{R}\right) / p(t)$

$p_{L}+p_{R}=1$

The region covered by the left child node, $t_{L}$, and the right child node, $t_{R^{\prime}}$, are disjoint and if combined, form the bigger region of their parent node $t$. The sum of the probabilities over two disjoined sets is equal to the probability of the union. $p_{L}$ then becomes the relative proportion of the left child node with respect to the parent node.

Next, we define the difference between the weighted impurity measure of the parent node and the two child nodes.

$$
\begin{aligned}
\Delta I(s, t) & =I(t)-I\left(t_{L}\right)-I\left(t_{R}\right) \\
& =p(t) i(t)-p\left(t_{L}\right) i\left(t_{L}\right)-p\left(t_{R}\right) i\left(t_{R}\right) \\
& =p(t) i(t)-p_{L} i\left(t_{L}\right)-p_{R} i\left(t_{R}\right) \\
& =p(t) \Delta i(s, t)
\end{aligned}
$$

Finally getting to this mystery of the impurity function...

It should be understood that no matter what impurity function we use, the way you use it in a classification tree is the same. The only difference is what specific impurity function to plug in. Once you use this, what follows is the same.

Possible impurity functions: 1. Entropy function: $\sum_{j=1}^{K} p_{j} \log \frac{1}{p_{j}}$. If $p_{j}=0$, use the limit $\lim p_{j} \rightarrow \log p_{j}=0$.

2. Misclassification rate: $1-\max _{j} p_{j}$.

3. Gini index: $\sum_{j=1}^{K} p_{j}\left(1-p_{j}\right)=1-\sum_{j=1}^{K} p_{j}^{2}$.

Remember, impurity functions have to 1) achieve a maximum at the uniform distribution, 2) achieve a minimum when $p_{j}=1$, and 3 ) be symmetric with regard to their permutations.

\section{Another Approach: The Twoing Rule}

Another splitting method is the Twoing Rule. This approach does not have anything to do with the impurity function.

The intuition here is that the class distributions in the two child nodes should be as different as possible and the proportion of data falling into either of the child nodes should be balanced.

The twoing rule: At node $t$, choose the split $s$ that maximizes:

$$
\frac{p_{L} p_{R}}{4}\left[\sum_{j}\left|p\left(j \mid t_{L}\right)-p\left(j \mid t_{R}\right)\right|\right]^{2}
$$

When we break one node to two child nodes, we want the posterior probabilities of the classes to be as different as possible. If they differ a lot, each tends to be pure. If instead, the proportions of classes in the two child nodes are roughly the same as the parent node, this indicates the splitting does not make the two child nodes much purer than the parent node and hence not a successful split.

In summary, one can use either the goodness of split defined using the impurity function or the twoing rule. At each node, try all possible splits exhaustively and select the best from them.

\section{3 - Estimate the Posterior Probabilities of Classes in Each Node}

\section{3 - Estimate the Posterior Probabilities of Classes in Each Node}

The impurity function is a function of the posterior probabilities of $k$ classes. In this section, we answer the question, "How do we estimate these probabilities?"

Let's begin by introducing the notation $N$, the total number of samples. The number of samples in class $j, 1 \leq j \leq K$, is $N_{j}$. If we add up all the $N_{j}$ data points, we get the total number of data points $N$.

We also denote the number of samples going to node $t$ by $N(t)$, and, the number of samples of class $j$ going to node $t$ by $N_{j}(t)$.

Then for every node $t$, if we add up over different classes we should get the total number of points back:

$$
\sum_{j=1}^{K} N_{j}(t)=N(t)
$$

And, if we add the points going to the left and the points going the right child node, we should also get the number of points in the parent node. 

$$
N_{j}\left(t_{L}\right)+N_{j}\left(t_{R}\right)=N_{j}(t)
$$

For a full tree (balanced), the sum of $N(t)$ over all the node $t$ 's at the same level is $N$.

Next, we will denote the prior probability of class $j$ by $\pi_{j}$. The prior probabilities very often are estimated from the data by calculating the proportion of data in every class. For instance, if we want the prior probability for class 1 , we simply compute the ratio between the number of points in class one and the total number of points, $N_{j} / N$. These are the so-called empirical frequencies for the classes.

This is one way of getting priors. Sometimes the priors may be pre-given. For instance, in medical studies, researchers collect a large amount of data from patients who have a disease. The percentage of cases with the disease in the collected data may be much higher than that in the population. In this case, it is inappropriate to use the empirical frequencies based on the data. If the data is a random sample from the population, then it may be reasonable to use empirical frequency.

The estimated probability of a sample in class $j$ going to node $t$ is $p(t \mid j)=N_{j}(t) / N_{j}$. Obviously,

$$
p\left(t_{L} \mid j\right)+p\left(t_{R} \mid j\right)=p(t \mid j)
$$

Next, we can assume that we know how to compute $p(t \mid j)$ and then we will find the joint probability of a sample point in class $j$ and in node $t$.

The joint probability of a sample being in class $j$ and going to node $t$ is as follows:

$$
p(j, t)=\pi_{j} p(t \mid j)=\pi_{j} N_{j}(t) / N_{j}
$$

Because the prior probability is assumed known (or calculated) and $p(t \mid j)$ is computed, the joint probability can be computed. The probability of any sample going to node $t$ regardless of its class is:

$$
p(t)=\sum_{j=1}^{K} p(j, t)=\sum_{j=1}^{K} \pi_{j} N_{j}(t) / N_{j}
$$

Note: $p\left(t_{L}\right)+p\left(t_{R}\right)=p(t)$.

Now, what we really need is $p(j \mid t)$. That is if I know a point goes to node $t$, what is the probability this point is in class $j$.

(Be careful because we have flipped the condition and the event to compute the probability for!)

The probability of a sample being in class $j$ given that it goes to node $t$ is:

$$
p(j \mid t)=p(j, t) / p(t)
$$

Probabilities on the right-hand side are both solved from the previous formulas.

For any $t, \sum_{j=1}^{K} p(j \mid t)=1$.

There is a shortcut if the prior is not pre-given, but estimated by the empirical frequency of class $j$ in the dataset!

When $\pi_{j}=N_{j} / N$, the simplification is as follows: - calculate $p(j \mid t)=N_{j}(t) / N(t)$ - for all the points that land in node $t$.

- $p(t)=N(t) / N$.

- $p(j, t)=N_{j}(t) / N$

This is the shortcut equivalent to the previous approach.

\section{3) Determining Stopping Criteria}

When we grow a tree, there are two basic types of calculations needed. First, for every node, we compute the posterior probabilities for the classes, that is, $p(j \mid t)$ for all $j$ and $t$. Then we have to go through all the possible splits and exhaustively search for the one with the maximum goodness. Suppose we have identified 100 candidate splits (i.e., splitting questions), to split each node, 100 class posterior distributions for the left and right child nodes each are computed, and 100 goodness measures are calculated. In the end, one split is chosen and only for this chosen split, the class posterior probabilities in the right and left child nodes are stored.

For the moment let's assume we will leave off our discussion of pruning for later and that we will grow the tree until some sort of stopping criteria is met.

A simple criterion is as follows. We will stop splitting a node $t$ when:

$$
\max _{s \in S} \Delta I(s, t)<\beta
$$

where $\Delta I$ (defined before) is the decrease in the impurity measure weighted by the percentage of points going to node $t$, s is the optimal split and $\beta$ is a pre-determined threshold.

We must note, however, the above stopping criterion for deciding the size of the tree is not a satisfactory strategy. The reason is that the tree growing method is greedy. The split at every node is 'nearsighted'. We can only look one step forward. A bad split in one step may lead to very good splits in the future. The tree growing method does not consider such cases.
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-10.jpg?height=744&width=1376&top_left_y=1853&top_left_x=110)This process of growing the tree is greedy because it looks only one step ahead as it goes. Going one step forward and making a bad decision doesn't mean that it is always going to end up bad. If you go a few steps more you might actually gain something. You might even perfectly separate the classes. A response to this might be that what if we looked two steps ahead? How about three steps? You can do this but it begs the question, "How many steps forward should we look?"

No matter how many steps we look forward, this process will always be greedy. Looking ahead multiple steps will not fundamentally solve this problem. This is why we need pruning.

\section{4) Determining Class Assignment Rules}

Finally, how do we decide which class to assign to each leaf node?

The decision tree classifies new data points as follows. We let a data point pass down the tree and see which leaf node it lands in. The class of the leaf node is assigned to the new data point. Basically, all the points that land in the same leaf node will be given the same class. This is similar to k-means or any prototype method.

A class assignment rule assigns a class $j=1, \cdots, K$ to every terminal (leaf) node $t \in \tilde{T}$. The class is assigned to node $t . \tilde{T}$ is denoted by $\kappa(t)$, e.g., if $\kappa(t)=2$, all the points in node $t$ would be assigned to class 2.

If we use 0-1 loss, the class assignment rule is very similar to k-means (where we pick the majority class or the class with the maximum posterior probability):

$$
\kappa(t)=\arg \max _{j} p(j \mid t)
$$

Let's assume for a moment that I have a tree and have the classes assigned for the leaf nodes. Now, I want to estimate the classification error rate for this tree. In this case, we need to introduce the resubstitution estimate $r(t)$ for the probability of misclassification, given that a case falls into node $t$. This is:

$$
r(t)=1-\max _{j} p(j \mid t)=1-p(\kappa(t) \mid t)
$$

Denote $R(t)=r(t) p(t)$. The resubstitution estimation for the overall misclassification rate $R(T)$ of the tree classifier $T$ is:

$$
R(T)=\sum_{t \in \tilde{T}} R(t)
$$

One thing that we should spend some time proving is that if we split a node $t$ into child nodes, the misclassification rate is ensured to improve. In other words, if we estimate the error rate using the resubstitution estimate, the more splits, the better. This also indicates an issue with estimating the error rate using the re-substitution error rate because it is always biased towards a bigger tree.

Let's go through the proof.

Proposition: For any split of a node $t$ into $t_{L}$ and $t_{R^{\prime}}$

$$
R(t) \geq R\left(t_{L}\right)+R\left(t_{R}\right)
$$

Proof: Denote $j *=\kappa(t)$.

Let's take a look at being in class $j *$ given that you are in node $t$.

$$
\begin{aligned}
p\left(j^{*} \mid t\right) & =p\left(j^{*}, t_{L} \mid t\right)+p\left(j^{*}, t_{R} \mid t\right) \\
& =p\left(j^{*} \mid t_{L}\right) p\left(t_{L} \mid t\right)+p\left(j^{*} \mid t_{R}\right) p\left(t_{R} \mid t\right) \\
& =p_{L} p\left(j^{*} \mid t_{L}\right)+p_{R} p\left(j^{*} \mid t_{R}\right) \\
& \leq p_{L} \max _{j} p\left(j \mid t_{L}\right)+p_{R} \max _{j} p\left(j \mid t_{R}\right)
\end{aligned}
$$

Hence,

$$
\begin{aligned}
r(\mid t) & =1-p\left(j^{*} \mid t\right) \\
& \geq 1-\left(p_{L} \max _{j} p\left(j \mid t_{L}\right)+p_{R} \max _{j} p\left(j \mid t_{R}\right)\right) \\
& =p_{L}\left(1-\max _{j} p\left(j \mid t_{L}\right)\right)+p_{R}\left(1-\max _{j} p\left(j \mid t_{R}\right)\right) \\
& =p_{L} r\left(t_{L}\right)+p_{R} r\left(t_{R}\right)
\end{aligned}
$$

Finally,

$$
\begin{aligned}
R(t) & =p(t) r(t) \\
& \geq p(t) p_{L} r\left(t_{L}\right)+p(t) p_{R} r\left(t_{R}\right) \\
& =p\left(t_{L}\right) r\left(t_{L}\right)+p\left(t_{R}\right) r\left(t_{R}\right) \\
& =R\left(t_{L}\right)+R\left(t_{R}\right)
\end{aligned}
$$

\section{4 - Example: Digit Recognition}

\section{4 - Example: Digit Recognition}

Here we have 10 digits, 0 through 9 , and as you might see on a calculator, they are displayed by different on-off combinations of seven horizontal and vertical light bars.
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-12.jpg?height=462&width=1012&top_left_y=1834&top_left_x=106)

In this case, each digit is represented by a 7-dimensional vector of zeros and ones. The $i^{\text {th }}$ sample is $x_{i}=\left(x_{i 1}, x_{i 2}, \cdots, x_{i 7}\right)$. If $x_{i j}=1$, the $j^{\text {th }}$ light is on; if $x_{i j}=0$, the $j^{\text {th }}$ light is off.

If the calculator works properly, the following table shows which light bars should be turned on or off for each of the digits. 

\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline Digit & $x \cdot 1$ & $x_{.2}$ & $x_{.3}$ & $x .4$ & $x \cdot 5$ & $x \cdot 6$ & $x .7$ \\
\hline 1 & 0 & 0 & 1 & 0 & 0 & 1 & 0 \\
2 & 1 & 0 & 1 & 1 & 1 & 0 & 1 \\
3 & 1 & 0 & 1 & 1 & 0 & 1 & 1 \\
4 & 0 & 1 & 1 & 1 & 0 & 1 & 0 \\
5 & 1 & 1 & 0 & 1 & 0 & 1 & 1 \\
6 & 1 & 1 & 0 & 1 & 1 & 1 & 1 \\
7 & 1 & 0 & 1 & 0 & 0 & 1 & 0 \\
8 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \\
9 & 1 & 1 & 1 & 1 & 0 & 1 & 1 \\
0 & 1 & 1 & 1 & 0 & 1 & 1 & 1 \\
\hline
\end{tabular}

Let's suppose that the calculator is malfunctioning. Each of the seven lights has probability 0.1 of being in the wrong state independently. In the training data set 200 samples are generated according to the specified distribution.

A classification tree is applied to this dataset. For this data set, each of the seven variables takes only two possible values: 0 and 1. Therefore, for every variable, the only possible candidate question is whether the value is on (1) or off (0). Consequently, we only have seven questions in the candidate pool: Is $x_{. j}=0 ?, j=1,2, \cdots, 7$.

In this example, the twoing rule is used in splitting instead of the goodness of split based on an impurity function. Also, the result presented was obtained using pruning and cross-validation.

\section{Classification Performance}

\section{Results:}

- The error rate estimated by using an independent test dataset of size 5000 is 0.30 .

- The error rate estimated by cross-validation using the training dataset which only contains 200 data points is also 0.30 . In this case, the cross-validation did a very good job for estimating the error rate.

- The resubstitution estimate of the error rate is 0.29 . This is slightly more optimistic than the true error rate.

Here pruning and cross-validation effectively help avoid overfitting. If we don't prune and grow the tree too big, we might get a very small resubstitution error rate which is substantially smaller than the error rate based on the test data set.

- The Bayes error rate is 0.26 . Remember, we know the exact distribution for generating the simulated data. Therefore the Bayes rule using the true model is known.

- There is little room for improvement over the tree classifier.

The tree obtained is shown below: 

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-14.jpg?height=868&width=1057&top_left_y=86&top_left_x=111)

We can see that the first question checks the on/off status of the fifth light. We ask whether $x_{5}=0$. If the answer is yes (the light is off), go to the left branch. If the answer is no, take the right branch. For the left branch, we ask whether the fourth light, $x_{4}$, is on or off. If the answer is yes, then we check the first light, $x_{1}$. If $x_{1}$ is off then we say that it is digit 1 , if the answer is no then it is a 7 . The square nodes are leaf nodes, and the number in the square is the class label, or the digit, assigned to the leaf node.

In general, one class may occupy several leaf nodes and occasionally no leaf node.

Interestingly, in this example, every digit (or every class) occupies exactly one leaf node. There are exactly 10 leaf nodes. But this is just a special case.

Another interesting aspect about the tree in this example is that $x_{6}$ and $x_{7}$ are never used. This shows that classification trees sometimes achieve dimension reduction as a by-product.

\section{Example: Waveforms}

Let's take a look at another example concerning waveforms. Let's first define three functions $h_{1}(\tau), h_{2}(\tau), h_{3}(\tau)$ which are shifted versions of each other, as shown in the figure below:

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-14.jpg?height=428&width=985&top_left_y=2062&top_left_x=113)

We specify each $h_{j}$ by its values at integers $\tau=1 \tilde{2} 1$. Therefore, every waveform is characterized by a 21-dimensional vector. Next, we will create in the following manner three classes of waveforms as random convex combinations of two of these waveforms plus independent Gaussian noise. Each sample is a 21dimensional vector containing the values of the random waveforms measured at $\tau=1,2, \cdots, 21$.

To generate a sample in class 1 , first we generate a random number $u$ uniformly distributed in $[0,1]$ and then we generate 21 independent random numbers $\epsilon_{1}, \epsilon_{2}, \cdots, \epsilon_{21}$ normally distributed with mean $=0$ and variance 1 (they are Gaussian noise.) Now, we can create a random waveform by a random sum of $h_{1}(j)$ and $h_{2}(j)$ with the weights given by the random number picked from the interval $[0,1]$ :

$$
x_{\cdot j}=u h_{1}(j)+(1-\nu) h_{2}(j)+\epsilon_{j}, \quad j=1, \cdots, 21
$$

This is a convex combination where the weights are nonnegative and add up to one, with Gaussian noise, $\epsilon_{j}$, added on top.

Similarly, to generate a sample in class 2 , we repeat the above process to generate a random number $\nu$ and 21 random numbers $\epsilon_{1}, \cdots, \epsilon_{21}$ and set

$$
x_{\cdot j}=u h_{1}(j)+(1-\nu) h_{3}(j)+\epsilon_{j}, \quad j=1, \cdots, 21
$$

This is a convex combination of $h_{1}(j)$ and $h_{3}(j)$ plus noise.

Then, the class 3 vectors are generated by a convex combination of $h_{2}(j)$ and $h_{3}(j)$ plus noise:

$$
x_{\cdot j}=u h_{2}(j)+(1-\nu) h_{3}(j)+\epsilon_{j}, \quad j=1, \cdots, 21
$$

Below are sample random waveforms generated according to the above description.
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-15.jpg?height=1268&width=1438&top_left_y=1515&top_left_x=113)Let's see how a classification tree performs.

Here we have generated 300 random samples using prior probabilities $(1 / 3,1 / 3,1 / 3)$ for training.

The set of splitting questions are whether $x_{j}$ is smaller than or equal to a threshold $c$ :

$\left\{\mathrm{Is} x_{\cdot j} \leq c ?\right\}$ for $c$ ranging over all real numbers and $j=1, \ldots, 21$

Next, we use the Gini index as the impurity function and compute the goodness of split correspondingly.

Then the final tree is selected by pruning and cross-validation.

\section{Results:}

- The cross-validation estimate of misclassification rate is 0.29 .

- The misclassification rate on a separate test dataset of size 5000 is 0.28 .

- This is pretty close to the cross-validation estimate!

- The Bayes classification rule can be derived because we know the underlying distribution of the three classes. Applying this rule to the test set yields a misclassification rate of 0.14.

We can see that in this example, the classification tree performs much worse than the theoretical optimal classifier.

Here is the tree. Again, the corresponding question used for every split is placed below the node. Three numbers are put in every node, which indicates the number of points in every class for that node. For instance, in the root node at the top, there are 100 points in class 1,85 points in class 2 , and 115 in class 3. Although the prior probabilities used were all one third, because random sampling is used, there is no guarantee that in the real data set the numbers of points for the three classes are identical.

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-16.jpg?height=1108&width=1125&top_left_y=1696&top_left_x=114)

If we take a look at the internal node to the left of the root node, we see that there are 36 points in class 1, 17 points in class 2 and 109 points in class 3, the dominant class. If we look at the leaf nodes represented by the rectangles, for instance, the leaf node on the far left, it has seven points in class 1, 0 points in class 2 and 20 points in class 3. According to the class assignment rule, we would choose a class that dominates this leaf node, 3 in this case. Therefore, this leaf node is assigned to class 3 , shown by the number below the rectangle. In the leaf node to its right, class 1 with 20 data points is most dominant and hence assigned to this leaf node.

We also see numbers on the right of the rectangles representing leaf nodes. These numbers indicate how many test data points in each class land in the corresponding leaf node. For the ease of comparison with the numbers inside the rectangles, which are based on the training data, the numbers based on test data are scaled to have the same sum as that on training.

Also, observe that although we have 21 dimensions, many of these are not used by the classification tree. The tree is relatively small.

\section{5 - Advantages of the Tree-Structured Approach}

\section{5 - Advantages of the Tree-Structured Approach}

As we have mentioned many times, the tree-structured approach handles both categorical and ordered variables in a simple and natural way. Classification trees sometimes do an automatic stepwise variable selection and complexity reduction. They provide an estimate of the misclassification rate for a test point. For every data point, we know which leaf node it lands in and we have an estimation for the posterior probabilities of classes for every leaf node. The misclassification rate can be estimated using the estimated class posterior.

Classification trees are invariant under all monotone transformations of individual ordered variables. The reason is that classification trees split nodes by thresholding. Monotone transformations cannot change the possible ways of dividing data points by thresholding. Classification trees are also relatively robust to outliers and misclassified points in the training set. They do not calculate an average or anything else from the data points themselves. Classification trees are easy to interpret, which is appealing especially in medical applications.

\section{6 - Variable Combinations}

\section{6 - Variable Combinations}

So far, we have assumed that the classification tree only partitions the space by hyperplanes parallel to the coordinate planes. In the two-dimensional case, we only divide the space either by horizontal or vertical lines. How much do we suffer by such restrictive partitions?

Let's take a look at this example...

In the example below, we might want to make a split using the dotted diagonal line which separates the two classes well. Splits parallel to the coordinate axes seem inefficient for this data set. Many steps of splits are needed to approximate the result generated by one split using a sloped line. 

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-18.jpg?height=562&width=625&top_left_y=96&top_left_x=110)

There are classification tree extensions which, instead of thresholding individual variables, perform LDA for every node.

Or we could use more complicated questions. For instance, questions that use linear combinations of variables:

$$
\sum a_{j} x_{. j} \leq c ?
$$

This would increase the amount of computation significantly. Research seems to suggest that using more flexible questions often does not lead to obviously better classification result, if not worse. Overfitting is more likely to occur with more flexible splitting questions. It seems that using the right sized tree is more important than performing good splits at individual nodes.

\section{7 - Missing Values}

\section{7 - Missing Values}

We may have missing values for some variables in some training sample points. For instance, geneexpression microarray data often have missing gene measurements.

Suppose each variable has $5 \%$ chance of being missing independently. Then for a training data point with 50 variables, the probability of missing some variables is as high as $92.3 \%$ ! This means that at least $90 \%$ of the data will have at least one missing value! Therefore, we cannot simply throw away data points whenever missing values occur.

A test point to be classified may also have missing variables.

Classification trees have a nice way of handling missing values by surrogate splits.

Suppose the best split for node $t$ is $s$ which involves a question on $X_{m}$. Then think about what to do if this variable is not there. Classification trees tackle the issue by finding a replacement split. To find another split based on another variable, classification trees look at all the splits using all the other variables and search for the one yielding a division of training data points most similar to the optimal split. Along the same line of thought, the second best surrogate split could be found in case both the best variable and its top surrogate variable are missing, so on so forth.

One thing to notice is that to find the surrogate split, classification trees do not try to find the second-best split in terms of goodness measure. Instead, they try to approximate the result of the best split. Here, the goal is to divide data as similarly as possible to the best split so that it is meaningful to carry out the future decisions down the tree, which descend from the best split. There is no guarantee the second best split divides data similarly as the best split although their goodness measurements are close.

\section{8 - Right Sized Tree via Pruning}

\section{8 - Right Sized Tree via Pruning}

In the previous section, we talked about growing trees. In this section, we will discuss pruning trees.

Let the expected misclassification rate of a tree $T$ be $R^{*}(T)$.

Recall we used the resubstitution estimate for $R^{*}(T)$. This is:

$$
R(T)=\sum_{t \in \tilde{T}} r(t) p(t)=\sum_{t \in \tilde{T}} R(t)
$$

Remember also that $r(t)$ is the probability of making a wrong classification for points in node $t$. For a point in a given leaf node $t$, the estimated probability of misclassification is 1 minus the probability of the majority class in node $t$ based on the training data.

To get the probability of misclassification for the whole tree, a weighted sum of the within leaf node error rate is computed according to the total probability formula.

We also mentioned in the last section the resubstitution error rate $R(T)$ is biased downward. Specifically, we proved the weighted misclassification rate for the parent node is guaranteed to be greater or equal to the sum of the weighted misclassification rates of the left and right child nodes, that is:

$$
R(t) \geq R\left(t_{L}\right)+R\left(t_{R}\right)
$$

This means that if we simply minimize the resubstitution error rate, we would always prefer a bigger tree. There is no defense against overfitting.

Let's look at the digit recognition example that we discussed before.

The biggest tree grown using the training data is of size 71. In other words, it has 71 leaf nodes. The tree is grown until all of the points in every leaf node are from the same class.

\begin{tabular}{ccc}
\hline No. Terminal Nodes & $R(T)$ & $R^{t s}(T)$ \\
\hline 71 & .00 & .42 \\
63 & .00 & .40 \\
58 & .03 & .39 \\
40 & .10 & .32 \\
34 & .12 & .32 \\
19 & .29 & .31 \\
10 & .29 & .30 \\
9 & .32 & .34 \\
7 & .41 & .47 \\
6 & .46 & .54 \\
5 & .53 & .61 \\
2 & .75 & .82 \\
1 & .86 & .91 \\
\hline
\end{tabular}

Then a pruning procedure is applied, (the details of this process we will get to later). We can see from the above table that the tree is gradually pruned. The tree next to the full tree has 63 leaf nodes, which is followed by a tree with 58 leaf nodes, so on so forth until only one leaf node is left. This minimum tree contains only a single node, the root.

The resubstitution error rate $R(T)$ becomes monotonically larger when the tree shrinks.

The error rate $R^{t s}$ based on a separate test data shows a different pattern. It decreases first when the tree becomes larger, hits minimum at the tree with 10 terminal nodes, and begins to increase when the tree further grows.

Comparing with $R(T), R^{t s}(T)$ better reflects the real performance of the tree. The minimum $R^{t s}(T)$ is achieved not by the biggest tree, but by a tree that better balances the resubstitution error rate and the tree size.

\subsection{1 - Preliminaries for Pruning}

\subsection{1 - Preliminaries for Pruning}

First, we would grow the tree to a large size. Denote this maximum size by $T_{\text {max }}$. Stopping criterion is not important here because as long as the tree is fairly big, it doesn't really matter when to stop. The overgrown tree will be pruned back eventually. There are a few ways of deciding when to stop:

1. Keep going until all terminal nodes are pure (contain only one class).

2. Keep going until the number of data in each terminal node is no greater than a certain threshold, say 5 , or even 1.

3. As long as the tree is sufficiently large, the size of the initial tree is not critical.

The key here is to make the initial tree sufficiently big before pruning back.

\section{Notation}

Now we need to introduce a notation... Let's take a look at the following definitions:

1. Descendant: a node $t^{\prime}$ is a descendant of node $t$ if there is a connected path down the tree leading from $t$ to $t^{\prime}$

2. Ancestor: $t$ is an ancestor of $t^{\prime}$ if $t^{\prime}$ is its descendant.

3. A branch $T_{t}$ of $T$ with root node $t \in T$ consists of the node $t$ and all descendants of $t$ in $T$.

4. Pruning a branch $T_{t}$ from a tree $T$ consists of deleting from $T$ all descendants of $t$, that is, cutting off all of $T_{t}$ except its root node. The tree pruned this way will be denoted by $T-T_{t}$

5. If $T^{\prime}$ is gotten from $T$ by successively pruning off branches, then $T^{\prime}$ is called a pruned subtree of $T$ and denoted by $T^{\prime}<T$ 
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-21.jpg?height=382&width=914&top_left_y=111&top_left_x=113)

\begin{tabular}{|l:l:}
\hline Video Explanation & Parts of a Tree \\
& https://www.youtube.com/watch/LfEKPJCYIOU
\end{tabular}

\section{Optimal Subtrees}

Even for a moderately sized $T_{\max }$, there is an enormously large number of subtrees and an even larger number ways to prune the initial tree to get any. We, therefore, cannot exhaustively go through all the subtrees to find out which one is the best in some sense. Moreover, we typically do not have a separate test dataset to serve as a basis for selection.

A smarter method is necessary. A feasible method of pruning should ensure the following:

- The subtree is optimal in a certain sense, and

- The search of the optimal subtree should be computationally tractable.

\subsection{2 - Minimal Cost-Complexity Pruning}

\subsection{2 - Minimal Cost-Complexity Pruning}

As we just discussed, $R(T)$, is not a good measure for selecting a subtree because it always favors bigger trees. We need to add a complexity penalty to this resubstitution error rate. The penalty term favors smaller trees, and hence balances with $R(T)$.

The definition for the cost-complexity measure:

For any subtree $T<T_{\max }$, we will define its complexity as $|\tilde{T}|$, the number of terminal or leaf nodes in $T$. Let $\alpha \geq 0$ be a real number called the complexity parameter and define the cost-complexity measure $R_{\alpha}(T)$ as:

$$
R_{\alpha}(T)=R(T)+\alpha|\tilde{T}|
$$

The more leaf nodes that the tree contains the higher complexity of the tree because we have more flexibility in partitioning the space into smaller pieces, and therefore more possibilities for fitting the training data. There's also the issue of how much importance to put on the size of the tree. The complexity parameter $\alpha$ adjusts that. In the end, the cost complexity measure comes as a penalized version of the resubstitution error rate. This is the function to be minimized when pruning the tree.

Which subtree is selected eventually depends on $\alpha$. If $\alpha=0$ then the biggest tree will be chosen because the complexity penalty term is essentially dropped. As $\alpha$ approaches infinity, the tree of size 1, i.e., a single root node, will be selected.

In general, given a pre-selected $\alpha$, find the subtree $T(\alpha)$ that minimizes $R_{\alpha}(T)$, i.e.,

$$
R_{\alpha}(T(\alpha))=\min _{T \preceq T_{\max }} R_{\alpha}(T)
$$

The minimizing subtree for any $\alpha$ always exists since there are only finitely many subtrees.

Since there are at most a finite number of subtrees of $T_{\max }, R_{\alpha}(T(\alpha))$ yields different values for only finitely many $\alpha^{\prime}$ s. $\backslash(T(\alpha) \backslash$ continues to be the minimizing tree when $\alpha$ increases until a jump point is reached.

Two questions:

1. Is there a unique subtree $T<T_{\max }$ which minimizes $R_{\alpha}(T)$ ?

2. In the minimizing sequence of trees $T_{1}, T_{2}, \cdots$ is each subtree obtained by pruning upward from the previous subtree, i.e., does the nesting $T_{1}>T_{2}>\cdots>t_{1}$ hold?

If the optimal subtrees are nested, the computation will be a lot easier. We can first find $T_{1}$, and then to find $T_{2}$, we don't need to start again from the maximum tree, but from $T_{1}$, (because $T_{2}$ is guaranteed to be a subtree of $T_{1}$ ). In this way when $\alpha$ increases, we prune based on a smaller and smaller subtree.

Definition: The smallest minimizing subtree $T_{\alpha}$ for complexity parameter $\alpha$ is defined by the conditions:

1. $R_{\alpha}(T(\alpha))=\min _{T \leq T \max } R_{\alpha}(T)$

2. If $R_{\alpha}(T)=R_{\alpha}(T(\alpha))$, then $T(\alpha) \leq T$. - if another tree achieves the minimum at the same $\alpha$, then the other tree is guaranteed to be bigger than the smallest minimized tree $T(\alpha)$.

By definition, (according to the second requirement above), if the smallest minimizing subtree $T(\alpha)$ exists, it must be unique. Earlier we argued that a minimizing subtree always exists because there are only a finite number of subtrees. Here we go one step more. We can prove that the smallest minimizing subtree always exists. This is not trivial to show because one tree smaller than another means the former is embedded in the latter. Tree ordering is a partial ordering.

The starting point for the pruning is not $T_{\max }$, but rather $T_{1}=T(0)$, which is the smallest subtree of $T_{\max }$ satisfying:

$$
R\left(T_{1}\right)=R\left(T_{\max }\right)
$$

The way that you get $T_{1}$ is as follows:

First, look at the biggest tree, $T_{\max }$, and for any two terminal nodes descended from the same parent, for instance $t_{L}$ and $t_{R}$, if they yield the same re-substitution error rate as the parent node $t_{\text {, }}$ prune off these two terminal nodes, that is, if $R(t)=R\left(t_{L}\right)+R\left(t_{R}\right)$, prune off $t_{L}$ and $t_{R}$. This process is applied recursively. After we have pruned one pair of terminal nodes, the tree shrinks a little bit. Then based on the smaller tree, we do the same thing until we cannot find any pair of terminal nodes satisfying this equality. The resulting tree at this point is $T_{1}$.

Let's take a look at an example using a plot to see how $T_{\max }$ is obtained.
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-23.jpg?height=824&width=1448&top_left_y=400&top_left_x=110)

We will use $T_{t}$ to denote a branch rooted at $t$. Then, for $T_{t}$, we define $R\left(T_{t}\right)$, (the resubstitution error rate for this branch ) by:

$$
R\left(T_{t}\right)=\sum_{t^{\prime} \in \tilde{T}_{t}} R\left(t^{\prime}\right)
$$

where $\tilde{T}_{t}$ is the set of terminal nodes of $T_{t}$.

What we can prove is that, if $t$ is any non-terminal or internal node of $T_{1}$, then it is guaranteed to have a smaller re-substitution error rate than the re-substitution error rate of the branch, that is, $R(t)>R\left(T_{t}\right)$. If we prune off the branch at $t$, the resubstitution error rate will strictly increase.

\section{Weakest-Link Cutting}

The weakest link cutting method not only finds the next $\alpha$ which results in a different optimal subtree but find that optimal subtree.

Remember, we previously defined $R_{\alpha}$ for the entire tree. Here, we extend the definition to a node and then for a single branch coming out of a node.

For any node $t \in T_{1}$, we can set $R_{\alpha}(t)=R(t)+\alpha$.

Also, for any branch $T_{t}$, we can define $R_{\alpha}\left(T_{t}\right)=R\left(T_{t}\right)+\alpha\left|\tilde{T}_{t}\right|$.

We know that when $\alpha=0, R_{0}\left(T_{t}\right)<R_{0}(t)$. The inequality holds for sufficiently small $\alpha$. If we gradually increase $\alpha$, because $R_{\alpha}\left(T_{t}\right)$ increases faster with $\alpha$ (the coefficient in front of $\alpha$ is larger than that $\operatorname{in}\left(R_{\alpha}(t)\right)$, at a certain $\alpha$ ( $\alpha_{1}$ below), we will have $R_{\alpha}\left(T_{t}\right)=R_{\alpha}(t)$.

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-24.jpg?height=634&width=928&top_left_y=240&top_left_x=104)

If $\alpha$ further increases, the inequality sign will be reversed, and we have $R_{\alpha}\left(T_{t}\right)>R_{\alpha}(t)$. Some node $t$ may reach the equality earlier than some other. The node that achieves the equality at the smallest $\alpha$ is called the weakest link. It is possible that several nodes achieve equality at the same time, and hence there are several weakest link nodes.

Solve the inequality $R_{\alpha}\left(T_{t}\right)<R_{\alpha}(t)$ and get

$$
\alpha<\frac{R(t)-R\left(T_{t}\right)}{\left|\tilde{T}_{t}\right|-1}
$$

The right hand side is the ratio between the difference in resubstitution error rates and the difference in complexity, which is positive because both the numerator and the denominator are positive.

Define a function $g_{1}(t), t \in T_{1}$ by

$g_{1}(t)\left\{\begin{array}{cc}\frac{R(t)-R\left(T_{t}\right)}{\left|\tilde{T}_{t}\right|-1}, & t \notin \tilde{T}_{1} \\ +\infty, & t \in \tilde{T}_{1}\end{array}\right.$

The weakest link $\bar{t}_{1}$ in $T_{1}$ achieves the minimum of $g_{1}(t)$

$$
g_{1}\left(\bar{t}_{1}\right)=\min _{t \in T_{1}} g_{1}(t)
$$

and put $\alpha_{2}=g_{1}\left(\bar{t}_{1}\right)$. To get the optimal subtree corresponding to $\alpha_{2}$, simply remove the branch growing out of $\bar{t}_{1}$. When $\alpha$ increases, $\bar{t}_{1}$ is the first node that becomes more preferable than the branch $T_{\bar{t}_{1}}$ descended from it. If there are several nodes that simultaneously achieve the minimum $g_{1}(t)$, we remove the branch grown out of each of these nodes. $\alpha_{2}$ is the first value after $\alpha_{1}=0$ that yields an optimal subtree strictly smaller than $T_{1}$. For all $\alpha_{1} \leq \alpha<\alpha_{2}$, the smallest minimizing subtree is the same as $T_{1}$.

$$
\text { Let } T_{2}=T_{1}-T_{\bar{t}_{1}}
$$

Repeat the previous steps. Use $T_{2}$ instead of $T_{1}$ as the starting tree, find the weakest link in $T_{2}$ and prune off at all the weakest link nodes to get the next optimal subtree. 

$$
\begin{aligned}
g_{2}(t) & =\left\{\begin{array}{cc}
\frac{R(t)-R\left(T_{2 t}\right)}{\left|\tilde{T}_{2 t}\right|-1}, & t \in T_{2}, t \notin \tilde{T}_{2} \\
+\infty, & t \in \tilde{T}_{2}
\end{array}\right. \\
g_{2}\left(\bar{t}_{2}\right) & =\underset{t \in T_{2}}{ } \min _{2}(t) \\
\alpha_{3} & =g_{2}\left(\bar{t}_{2}\right) \\
T_{3} & =T_{2}-T_{\bar{t}_{2}}
\end{aligned}
$$

\section{Computation}

In terms of computation, we need to store a few values at each node.

- $R(t)$, the resubstitution error rate for node $t$. This only need to be computed once.

- $R\left(T_{t}\right)$, the resubstitution error rate for the branch coming out of node $t$. This may need to be updated after pruning because $T_{t}$ may change after pruning.

- $\left|T_{t}\right|$, the number of leaf nodes on the branch coming out of node $t$. This may need to be updated after pruning.

\section{$\mathbf{R ( t )}$}

In order to compute the resubstitution error rate $R(t)$ we need the proportion of data points in each class that land in node $t$. Let's suppose we compute the class priors by the proportion of points in each class. As we grow the tree, we will store the number of points land in node $t$, as well as the number of points in each class that land in node $t$. Given those numbers, we can easily estimate the probability of node $t$ and the class posterior given a data point is in node $t . R(t)$ can then be calculated.

As far as calculating the next two numbers, a) the resubstitution error rate for the branch coming out of node $t$, and $b$ ) the number of leaf nodes that are on the branch coming out of node $t$, these two numbers change after pruning. After pruning we to need to update these values because the number of leaf nodes will have been reduced. To be specific we would need to update the values for all of the ancestor nodes of the branch.

A recursive procedure can be used to compute $R\left(T_{t}\right)$ and $\left|T_{t}\right|$.

To find the number of leaf nodes in the branch coming out of node $t$, we can do a bottom-up sweep through the tree. The number of leaf nodes for any node is equal to the number of leaf nodes for the right child node plus the number of leaf nodes for the right child node. A bottom-up sweep ensures that the number of leaf nodes is computed for a child node before for a parent node. Similarly, $R\left(T_{t}\right)$ is equal to the sum of the values for the two child nodes of $t$. And hence, a bottomup sweep would do.

Once we have the three values at every node, we compute the ratio $g(t)$ and find the weakest link. The corresponding ratio at the weakest link is the new $\alpha$. It is guaranteed that the sequence of $\alpha$ obtained in the pruning process is strictly increasing. If at any stage, there are multiple weakest links, for instance, if $g_{k}\left(\bar{t}_{k}\right)=g_{k}\left(\bar{t}_{k}^{\prime}\right)$, then define:

$$
T_{k+1}=T_{k}-T_{\bar{t}_{k}}-T_{\bar{t}_{k}^{\prime}}
$$

In this case, the two branches are either nested or share no node. The pruning procedure gives sequence of nested subtrees:

$$
T_{1}>T_{2}>T_{3}>\cdots>t_{1}
$$

Each embedded in the other.

\section{Theorem for $\alpha_{k}$}

The theorem states that the $\left\{\alpha_{k}\right\}$ are an increasing sequence, that is, $\alpha_{k}<\alpha_{k+1}, k \geq 1$, where $\alpha_{1}=0$.

For any $k \geq 1, \alpha_{k} \leq \alpha<\alpha_{k+1}$, the smallest optimal subtree $T(\alpha)=T\left(\alpha_{k}\right)=T_{k}$, i.e., is the same as the smallest optimal subtree for $\alpha_{k}$.

Basically, this means that the smallest optimal subtree $T_{k}$ stays optimal for all the $\alpha$ 's starting from $k$ until it reaches $\alpha_{k}+1$. Although we have a sequence of finite subtrees, they are optimal for a continuum of $\alpha$.

At the initial steps of pruning, the algorithm tends to cut off large sub-branches with many leaf nodes very quickly. Then pruning becomes slower and slower as the tree becoming smaller. The algorithm tends to cut off fewer nodes. Let's look at an example.

\section{Digital Recognition Example}

$T_{1}$ is the smallest optimal subtree for $\alpha_{1}=0$. It has 71 leaf nodes. Next, by finding the weakest link, after one step of pruning, the tree is reduced to size 63 (8 leaf nodes are pruned off in one step). Next, five leaf nodes pruned off. From $T_{3}$ to $T_{4}$, the pruning is significant, 18 leave nodes removed. Towards the end, pruning becomes slower.

For classification purpose, we have to select a single $\alpha$, or a single subtree to use.

\subsection{3 - Best Pruned Subtree}

\subsection{3 - Best Pruned Subtree}

There are two approaches to choosing the best-pruned subtree:

- Use a test sample set

If we have a large test data set, we can compute the error rate using the test data set for all the subtrees and see which one achieves the minimum error rate. However, in practice, we rarely have a large test data set. Even if we have a large test data set, instead of using the data for testing, we might rather use this data for training in order to train a better tree. When data is scarce, we may not want to use too much for testing.

\section{- Cross-validation}

How to conduct cross-validation for trees when trees are unstable? If the training data vary a little bit, the resulting tree may be very different. Therefore, we would have difficulty to match the trees obtained in each fold with the tree obtained using the entire data set.

However, although we said that the trees themselves can be unstable, this does not mean that the classifier resulting from the tree is unstable. We may end up with two trees that look very different, but make similar decisions for classification. The key strategy in a classification tree is to focus on choosing the right complexity parameter $\alpha$. Instead of trying to say which tree is best, a classification tree tries to find the best complexity parameter $\alpha$.

So, let's look at this...

\section{Pruning by Cross-Validation}

Let's consider $V$-fold cross-validation.

We will denote the original learning sample $L$ which is divided randomly into $V$ subsets, $L_{v}, \quad v=1, \cdots, V$. We will also let the training sample set in each fold be $L^{(v)}=L-L_{v}$.

Next, the tree is grown on the original set and we call this $T_{m a x}$. Then, we repeat this procedure for every fold in the cross-validation. So, $V$ additional trees $T_{m a x}^{(v)}$ are grown on $L^{(v)}$.

For each value of the complexity parameter $\alpha$, we will let $T^{(v)}, L^{(v)}(\alpha), v=1, \cdots, V$, be the corresponding minimal cost-complexity subtrees of $T_{\max } T^{(v)} T_{\max }$.

For each maximum tree, we obtain a sequence of critical values for $\alpha$ that are strictly increasing:

$$
\alpha_{1}<\alpha_{2}<\alpha_{3} \cdots<\alpha_{k}<\cdots
$$

Then, to find the corresponding minimal cost-complexity subtree at $\alpha$, we will find $\alpha_{k}$ from the list such that $\alpha_{k} \leq \alpha_{k+1}$. The optimal subtree corresponding to $\alpha_{k}$ is the subtree for $\alpha$.

The cross-validation error rate of $T(\alpha)$ is computed by this formula:

$$
R^{C V}(T(\alpha))=\frac{1}{V} \sum_{v=1}^{V} \frac{N_{m i s s}^{(v)}}{N^{(v)}}
$$

where $N^{(v)}$ is the number of samples in the test set $L_{v}$ in fold $v_{\text {; }}$ and $N_{m i s s}^{(v)}$ is the number of misclassified samples in $L_{v}$ using the smallest minimizing subtree at $\alpha, T^{(v)}(\alpha)$.

Remember: $T^{(v)}(\alpha)$ is a pruned tree of $T_{m a x}^{(v)}$ trained from $L_{v}$

Although $\alpha$ is continuous, there are only finitely many minimum cost-complexity trees grown on $L$. Consider each subtree obtained in the pruning of the tree grown on $L$. Let $T_{k}=T\left(\alpha_{k}\right)$. To compute the cross-validation error rate of $T_{k}$, let $\alpha_{k}^{\prime}=\sqrt{\alpha_{k} \alpha_{k+1}^{\prime}}$. Then compute the cross-validation error rate using the formula below:

$$
R^{C V}\left(T_{k}\right)=R^{C V}\left(T\left(\alpha_{k}^{\prime}\right)\right.
$$

The $T_{k}$ yielding the minimum cross-validation error rate is chosen.

\section{Computational Cost}

How much computation is involved? Let's take a look at this when using V-fold cross validation.

1. Grow $V+1$ maximum trees.

2. For each of the $V+1$ trees, find the sequence of subtrees with minimum cost-complexity.

3. Suppose we have obtained the maximum tree grown on the original data set $T_{m a x}$ and has $K$ subtrees. Then, for each of the $(K-1) \alpha_{k^{\prime}}^{\prime}$ we would compute the misclassification rate of each of the $V$ test sample set, average the error rates and use the mean as the cross-validation error rate.

4. Choose the best subtree of $T_{m a x}$ with minimum $R^{C V}\left(T_{k}\right)$.

The main computation occurs with pruning. Once the trees and the subtrees are obtained, to find the best one out of these is computationally light. For programming, it is recommended that under every fold and for every subtree, compute the error rate of this subtree using the corresponding test data set under that fold and store the error rate for that subtree. This way, later we can easily compute the cross-validation error rate given any $\alpha$.

\subsection{4 - Related Methods for Decision Trees}

\subsection{4 - Related Methods for Decision Trees}

\section{Random Forests}

Leo Breiman developed an extension of decision trees called random forests. There is publicly available software for this method. Here is a good place to start:

http://en.wikipedia.org/wiki/Random forest ${ }^{[5]}$

Random forests train multiple trees. To obtain a single tree, when splitting a node, only a randomly chosen subset of features are considered for thresholding. Leo Breiman did extensive experiments using random forests and compared it with support vector machines. He found that overall random forests seem to be slightly better. Moreover, random forests come with many other advantages.

\section{Other extensions}

The candidate questions in decision trees are about whether a variable is greater or smaller than a given value. There are some extensions to more complicated questions, or splitting methods, for instance, performing LDA at every node, but the original decision tree method seems to stay as the most popular and there is no strong evidence that the extensions are considerably better in general. 

\section{9 - Bagging and Random Forests}

\section{9 - Bagging and Random Forests}

In the past, we have focused on statical learning procedures that produce a single set of results. For example:

- A regression equation, with one set of regression coefficients or smoothing parameters.

- A classification regression tree with one set of leaf nodes.

Model selection is often required: a measure of fit associated with each candidate model.

\section{The Aggregating Procedure:}

Here the discussion shifts to statistical learning building on many sets of outputs that are aggregated to produce results. The aggregating procedure makes a number of passes over the data.

On each pass, inputs $X$ are linked with outputs $Y$ just as before. However, of interest now is the collection of all the results from all passes over the data. Aggregated results have several important benefits:

- Averaging over a collection of fitted values can help to avoid overfitting. It tends to cancel out the uncommon features of the data captured by a specific model. Therefore, the aggregated results are more stable.

- A large number of fitting attempts can produce very flexible fitting functions.

- Putting the averaging and the flexible fitting functions together has the potential to break the bias-variance tradeoff.

\section{Revisit Overfitting:}

Any attempt to summarize patterns in a dataset risk overntting. All fitting procedures adapt to the data on hand so that even if the results are applied to a new sample from the same population, fit quality will likely decline. Hence, generalization can be somewhat risky.

"optimism increases linearly with the number of inputs or basis functions ..., but decreases as the training sample size increases." -- Hastie, Tibshirani and Friedman (unjustified).

\section{Decision Tree Example:}

Consider decision trees as a key illustration. The overfitting often increases with (1) the number of possible splits for a given predictor; (2) the number of candidate predictors; (3) the number of stages which is typically represented by the number of leaf nodes. When overfitting occurs in a classification tree, the classification error is underestimated; the model may have a structure that will not generalize well. For example, one or more predictors may be included in a tree that really does not belong.

Ideally, one would have two random samples from the same population: a training dataset and a test dataset. The fit measure from the test data would be a better indicator of how accurate the classification is. Often there is only a single dataset. The data are split up into several randomly chosen, non-overlapping, partitions of about the same size. With ten partitions, each would be a part of the training data in nine analyses and serve as the test data in one analysis. The following figure illustrates the 2-fold cross validation for estimating the cross-validation prediction error for model A and model $B$. The model selection is based on choosing the one with the smallest cross-validation prediction error.

- We always evaluate the same model while switching the training and test data.
![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-30.jpg?height=776&width=1450&top_left_y=964&top_left_x=106)

\section{9 - R Scripts}

\section{9 - R Scripts}

\section{$\mathbf{R}$}

\section{Acquire Data}

\section{Diabetes data}

The diabetes data set is taken from the UCI machine learning database on Kaggle: Pima Indians Diabetes Database ${ }^{[6]}$

- 768 samples in the dataset

- 8 quantitative variables

- 2 classes; with or without signs of diabetes Load data into $R$ as follows:

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-31.jpg?height=212&width=1396&top_left_y=191&top_left_x=112)

In RawData, the response variable is its last column; and the remaining columns are the predictor variables.

responseY $=$ as. $\operatorname{matrix}(\operatorname{RawData}[, \operatorname{dim}(\operatorname{RawData})[2]])$

predictorX = as.matrix $(\operatorname{RawData}[, 1:(\operatorname{dim}(\operatorname{RawData})[2]-1)])$

data.train = as. data. frame(cbind(responseY, predictor $X)$ )

names(data.train) $=c(" Y ", " X 1 "$, "X2", "X3", "X4", "X5", "X6", "X7", "X8")

\section{Classification and Regression Trees}

The generation of a tree comprises two steps: to grow a tree and to prune a tree.

\subsection{Grow a Tree}

In $R$, the tree library can be used to construct classification and regression trees (see $R$ Lab 8). As an alternative, they can also be generated through the rpart library package and the rpart (formula) function grows a tree of the data. For the argument method, rpart (formula, method="class") specifies the response is a categorical variable, otherwise rpart (formula, method="anova") is assumed for a continuous response.

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-31.jpg?height=217&width=1709&top_left_y=1534&top_left_x=111)

To plot the tree, the following code can be executed in R: plot (result, uniform=TRUE ) plots the tree nodes, which are vertically equally spaced. Then text (result, use.n=TRUE ) writes the decision equation at each node.

plot (model.tree, uniform $=T$ )

text (model.tree, use. $n=T$ ) 

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-32.jpg?height=1154&width=1062&top_left_y=100&top_left_x=220)

Figure 1: Classification tree obtained by the tree growing function

In Figure 1, the plot shows the predictor and its threshold used at each node of the tree, and it shows the number of observations in each class at each terminal node. Specifically, the numbers of points in Class 0 and Class 1 are displayed as $\%$

At each node, we move down the left branch if the decision statement is true and down the right branch if it is false. The functions in the rpart library draw the tree in such a way that left branches are constrained to have a higher proportion of 0 values for the response variable than right branches. So, some of the decision statements contain "less than" $(<)$ symbols and some contain "greater than or equal to" (>=) symbols (whatever is needed to satisfy this constraint). By contrast, the functions in the tree library draw the tree in such a way that all the decision statements contain "less than" (<) symbols. Thus, either branch may have a higher proportion of 0 values for the response value than the other.

\subsection{Prune a Tree}

To obtain the right sized tree to avoid overfitting, the cptable element of the result generated by rpart can be extracted.

model.tree $\backslash$ \$cptable

The results are shown as follows: 

$\begin{array}{rrrrrr} & & \text { CP } & \text { nsplit rel error } & \text { xerror } & \text { xstd } \\ 1 & 0.24253731 & 0 & 1.0000000 & 1.0000000 & 0.04928752 \\ 2 & 0.10447761 & 1 & 0.7574627 & 0.8171642 & 0.04668666 \\ 3 & 0.01741294 & 2 & 0.6529851 & 0.7537313 & 0.04552694 \\ 4 & 0.01492537 & 5 & 0.6007463 & 0.7201493 & 0.04485359 \\ 5 & 0.01305970 & 9 & 0.5410448 & 0.7500000 & 0.04545421 \\ 6 & 0.01119403 & 12 & 0.4925373 & 0.7313433 & 0.04508278 \\ 7 & 0.01000000 & 15 & 0.4589552 & 0.7425373 & 0.04530720\end{array}$

The cptable provides a brief summary of the overall fit of the model. The table is printed from the smallest tree (no splits) to the largest tree. The "CP" column lists the values of the complexity parameter, the number of splits is listed under"nsplit", and the column "xerror" contains crossvalidated classification error rates; the standard deviation of the cross-validation error rates are in the "xstd" column. Normally, we select a tree size that minimizes the cross-validated error, which is shown in the "xerror" column printed by ()\$cptable.

Selection of the optimal subtree can also be done automatically using the following code:

opt <- model.tree \\$cptable[which.min(model.tree\$cptable[ , "xerror"]), "CP"]

opt stores the optimal complexity parameter. Now, to prune a tree with the complexity parameter chosen, simply do the following. The pruning is performed by function prune, which takes the full tree as the first argument and the chosen complexity parameter as the second.

model.ptree <- prune(model.tree, $c p=$ opt)

The pruned tree is shown in Figure 2 using the same plotting functions for creating Figure 1.

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-33.jpg?height=879&width=1243&top_left_y=1628&top_left_x=338)

Figure 2: Pruned classification tree Further information on the pruned tree can be accessed using the summary ( ) function.

\subsection{0 - Bagging}

\subsection{0 - Bagging}

There is a very powerful idea in the use of subsamples of the data and in averaging over subsamples through bootstrapping.

Bagging exploits that idea to address the overfitting issue in a more fundamental manner. It was invented by Leo Breiman, who called it "bootstrap aggregating" or simply "bagging" (see the reference: "Bagging Predictors," Machine Learning, 24:123-140, 1996, cited by 7466).

In a classification tree, bagging takes a majority vote from classifiers trained on bootstrap samples of the training data.

Algorithm: Consider the following steps in a fitting algorithm with a dataset having $N$ observations and a binary response variable.

1. Take a random sample of size $N$ with replacement from the data (a bootstrap sample).

2. Construct a classification tree as usual but do not prune.

3. Assign a class to each terminal node, and store the class attached to each case coupled with the predictor values for each observation.

4. Repeat Steps 1-3 a large number of times.

5. For each observation in the dataset, count the number of trees that it is classified in one category over the number of trees.

6. Assign each observation to a final category by a majority vote over the set of trees. Thus, if 51\\% of the time over a large number of trees a given observation is classified as a "1", that becomes its classification.

Although there remain some important variations and details to consider, these are the key steps to producing "bagged" classification trees. The idea of classifying by averaging over the results from a large number of bootstrap samples generalizes easily to a wide variety of classifiers beyond classification trees.

\section{Margins:}

Bagging introduces a new concept, "margins." Operationally, the "margin" is the difference between the proportion of times a case is correctly classified and the proportion of times it is incorrectly classified. For example, if over all trees an observation is correctly classified $75 \backslash \%$ of the time, the margin is $0.75-0.25=0.50$.

Large margins are desirable because a more stable classification is implied. Ideally, there should be large margins for all of the observations. This bodes well for generalization to new data. 

\section{Out-Of-Bag Observations:}

For each tree, observations not included in the bootstrap sample are called "out-of-bag" observations. These "out-of-bag" observations can be treated as a test dataset and dropped down the tree.

To get a better evaluation of the model, the prediction error is estimated only based on the "out-ofbag" observations. In other words, the averaging for a given observation is done only using the trees for which that observation was not used in the fitting process.

\section{Example: Domestic Violence}

Why Bagging Works? The core of bagging's potential is found in the averaging over results from a substantial number of bootstrap samples. As a first approximation, the averaging helps to cancel out the impact of random variation. However, there is more to the story, some details of which are especially useful for understanding a number of topics we will discuss later.

Data were collected to help forecast incidents of domestic violence within households. For a sample of households to which sheriff's deputies were dispatched for domestic violence incidents, the deputies collected information on a series of possible predictors of future domestic violence, for example, whether police officers had been called to that household in the recent past.

The following three figures are three classification trees constructed from the same data, but each using a different bootstrap sample. 

![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-36.jpg?height=1534&width=1425&top_left_y=204&top_left_x=110)



![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-37.jpg?height=1416&width=1448&top_left_y=200&top_left_x=107)



![](https://cdn.mathpix.com/cropped/2023_03_23_a4ef54461120e80fe6f5g-38.jpg?height=1408&width=1440&top_left_y=207&top_left_x=111)

It is clear that the three figures are very different. Unstable results may be due to any number of common problems: small sample sizes, highly correlated predictors; or heterogeneous terminal nodes. Interpretations from the results of a single tree can be quite risky when a classification tree performs in this manner. However, when a classification tree is used solely as a classification tool, the classes assigned may be relatively stable even if the tree structure is not.

The same phenomenon can be found in conventional regression when predictors are highly correlated. The regression coefficients estimated for particular predictors may be very unstable, but it does not necessarily follow that the fitted values will be unstable as well.

It is not clear how much bias exists in the three trees. But it is clear that the variance across trees is large. Bagging can help with the variance. The conceptual advantage of bagging is to aggregate fitted values from a large number of bootstrap samples. Ideally, many sets of fitted values, each with low bias but high variance, may be averaged in a manner that can effectively reduce the bite of the bias-variance tradeoff. The ways in which bagging aggregates the fitted values are the basis for many other statistical learning developments. 

\section{Bagging a Quantitative Response:}

Recall that a regression tree maximizes the reduction in the error sum of squares at each split. All of the concerns about overfitting apply, especially given the potential impact that outliers can have on the fitting process when the response variable is quantitative. Bagging works by the same general principles when the response variable is numerical.

- For each tree, each observation is placed in a terminal node and assigned the mean of that terminal node.

- Then, the average of these assigned means over trees is computed for each observation.

- This average value for each observation is the bagged fitted value.

\section{$\mathbf{R}$ Package for Bagging}

In R, the bagging procedure (i.e., bagging( ) in the ipred library) can be applied to classification, regression, and survival trees.

nbagg gives an integer giving the number of bootstrap replications. control gives control details of the rpart algorithm.

We can also use the random forest procedure in the "randomForest" package since bagging is a special case of random forests.

\subsection{1 - From Bagging to Random Forests}

\subsection{1 - From Bagging to Random Forests}

Bagging constructs a large number of trees with bootstrap samples from a dataset. But now, as each tree is constructed, take a random sample of predictors before each node is split. For example, if there are twenty predictors, choose a random five as candidates for constructing the best split. Repeat this process for each node until the tree is large enough. And as in bagging, do not prune.

\section{Random Forests Algorithm}

The random forests algorithm is very much like the bagging algorithm. Let $N$ be the number of observations and assume for now that the response variable is binary.

1. Take a random sample of size $N$ with replacement from the data (bootstrap sample).

2. Take a random sample without replacement of the predictors.

3. Construct a split by using predictors selected in Step 2.

4. Repeat Steps 2 and 3 for each subsequent split until the tree is as large as desired. Do not prune. Each tree is produced from a random sample of cases and at each split a random sample of predictors. 5. Drop the out-of-bag data down the tree. Store the class assigned to each observation along with each observation's predictor values.

6. Repeat Steps 1-5 a large number of times (e.g., 500).

7. For each observation in the dataset, count the number of trees that it is classified in one category over the number of trees.

8. Assign each observation to a final category by a majority vote over the set of trees. Thus, if $51 \%$ of the time over a large number of trees a given observation is classified as a "1", that becomes its classification.

\section{Why Random Forests Work}

Variance reduction: the trees are more independent because of the combination of bootstrap samples and random draws of predictors.

- It is apparent that random forests are a form of bagging, and the averaging over trees can substantially reduce instability that might otherwise result. Moreover, by working with a random sample of predictors at each possible split, the fitted values across trees are more independent. Consequently, the gains from averaging over a large number of trees (variance reduction) can be more dramatic.

Bias reduction: a very large number of predictors can be considered, and local feature predictors can play a role in tree construction.

- Random forests are able to work with a very large number of predictors, even more, predictors than there are observations. An obvious gain with random forests is that more information may be brought to reduce bias of fitted values and estimated splits.

- There are often a few predictors that dominate the decision tree fitting process because on the average they consistently perform just a bit better than their competitors. Consequently, many other predictors, which could be useful for very local features of the data, are rarely selected as splitting variables. With random forests computed for a large enough number of trees, each predictor will have at least several opportunities to be the predictor defining a split. In those opportunities, it will have very few competitors. Much of the time a dominant predictor will not be included. Therefore, local feature predictors will have the opportunity to define a split.

Indeed, random forests are among the very best classifiers invented to date (Breiman, 2001a).

Random forests include 3 main tuning parameters.

- Node Size: unlike in decision trees, the number of observations in the terminal nodes of each tree of the forest can be very small. The goal is to grow trees with as little bias as possible.

- Number of Trees: in practice, 500 trees is often a good choice.

- Number of Predictors Sampled: the number of predictors sampled at each split would seem to be a key tuning parameter that should affect how well random forests perform. Sampling 25 each time is often adequate. 

\section{Taking Costs into Account}

In the example of domestic violence, the following predictors were collected from 500+ households: Household size and number of children; Male / female age (years); Marital duration; Male / female education (years); Employment status and income; The number of times the police had been called to that household before; Alcohol or drug abuse, etc.

Our goal is not to forecast new domestic violence, but only those cases in which there is evidence that serious domestic violence has actually occurred. There are 29 felony incidents which are very small as a fraction of all domestic violence calls for service (4\%). And they would be extremely difficult to forecast. When a logistic regression was applied to the data, not a single incident of serious domestic violence was identified.

There is a need to consider the relative costs of false negatives (fail to predict a felony incident) and false positives (predict a case to be a felony incident when it is not). Otherwise, the best prediction would be assuming no serious domestic violence with an error rate of $4 \%$. In random forests, there are two common approaches. They differ by whether costs are imposed on the data before each tree is built, or at the end when classes are assigned.

Weighted Classification Votes: After all of the trees are built, one can differentially weight the classification votes over trees. For example, one vote for classification in the rare category might count the same as two votes for classification in the common category.

Stratified Bootstrap Sampling: When each bootstrap sample is drawn before a tree is built, one can oversample one class of cases for which forecasting errors are relatively more costly. The procedure is much in the same spirit as disproportional stratified sampling used for data collection (Thompson, 2002).

Using a cost ratio of 10 to 1 for false negatives to false positives favored by the police department, random forests correctly identify half of the rare serious domestic violence incidents.

In summary, with forecasting accuracy as a criterion, bagging is in principle an improvement over decision trees. It constructs a large number of trees with bootstrap samples from a dataset. Random forests are in principle an improvement over bagging. It draws a random sample of predictors to define each split.

\section{$\mathbf{R}$ Package for Random Forests}

In R, the random forest procedure can be implemented by the "randomForest" package.

$r f=$ randomForest $(x=X$.train, $y=$ as. factor ( $Y$.train), importance $=T$, do. trace $=50$, ntree $=200$, classwt $=c(5,1))$

print $(r f)$

\subsection{2 - Boosting}

\subsection{2 - Boosting}

Boosting, like bagging, is another general approach for improving prediction results for various statistical learning methods. It is also particularly well suited to decision trees. Section 8.2.3 in the textbook provides details.

After completing the reading for this lesson, please finish the Quiz and R Lab on Canvas (check the course schedule for due dates).

\begin{tabular}{|l|l|}
\hline \multicolumn{2}{c|}{ Legend } \\
\hline$[1]$ & Link \\
\hline$\uparrow$ & Has Tooltip/Popover \\
\hline\lceil & Toggleable Visibility \\
\hline
\end{tabular}

Source: https://www.google.com/

Links:

1. https://www.youtube.com/watch/1ow2tF9Ezgs

2. https://www.youtube.com/watch/iOJRhjb3y $\underline{j o}$

3. https://www.youtube.com/watch/LfEKPJcYIOU

4. https://www.youtube.com/watch/wpkGWZwJUTU

5. http://en.wikipedia.org/wiki/Random forest

6. https://www.kaggle.com/uciml/pima-indians-diabetes-database