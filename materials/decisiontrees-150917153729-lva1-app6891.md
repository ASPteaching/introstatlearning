![](https://cdn.mathpix.com/cropped/2023_03_29_28353530c600743d6547g-01.jpg?height=653&width=998&top_left_y=449&top_left_x=1265)

\title{
Decision Trees
}

\section{bigml}

![](https://cdn.mathpix.com/cropped/2023_03_29_28353530c600743d6547g-01.jpg?height=146&width=904&top_left_y=2299&top_left_x=768)

![](https://cdn.mathpix.com/cropped/2023_03_29_28353530c600743d6547g-01.jpg?height=276&width=311&top_left_y=2225&top_left_x=1727)

UNIVERSITAT

POLITECNICA DE VALENNCIA - What is a decision tree?

- History

- Decision tree learning algorithm

- Growing the tree

- Pruning the tree

- Capabilities 

\section{What is a decision tree?}

- Hierarchical learning model that recursively partitions the space using decision rules

- Valid for classification and regression

- Ok, but what is a decision tree? 

\section{Example for classification}

- Labor negotiations: predict if a collective agreement is good or bad

Internal nodes

![](https://cdn.mathpix.com/cropped/2023_03_29_28353530c600743d6547g-04.jpg?height=1996&width=3306&top_left_y=576&top_left_x=-2)



\section{Example for regression}

- Boston housing: predict house values in Boston by neigbourhood

![](https://cdn.mathpix.com/cropped/2023_03_29_28353530c600743d6547g-05.jpg?height=1783&width=3274&top_left_y=755&top_left_x=277)



\section{History}

- Precursors: Expert Based Systems (EBS)

\section{$\mathrm{EBS}=$ Knowledge database + Inference Engine}

- MYCIN: Medical diagnosis system based, 600 rules

- XCON: System for configuring VAX computers, 2500 rules $(1982)$

- The rules were created by experts by hand!!

- Knowledge acquisition has to be automatized

- Substitute the Expert by its archive with solved cases 

\section{History}

- CHAID (CHi-squared Automatic Interaction Detector) Gordon V. Kass ,1980

- CART (Classification and Regression Trees), Breiman, Friedman, Olsen and Stone, 1984

- ID3 (Iterative Dichotomiser 3), Quinlan, 1986

- C4.5, Quinlan 1993: Based on ID3 

\section{Computational}

- Consider two binary variables. How many ways can we split the space using a decision tree?

![](https://cdn.mathpix.com/cropped/2023_03_29_28353530c600743d6547g-08.jpg?height=1289&width=1719&top_left_y=866&top_left_x=964)

- Two possible splits and two possible assignments to the leave nodes $\rightarrow$ At least 8 possible trees 

\section{Computational}

- Under what conditions someone waits in a restaurant?

\begin{tabular}{|c||c|c|c|c|c|c|c|c|c|c||c||c|}
\hline \multicolumn{1}{|c||}{ Example } & \multicolumn{10}{|c|}{ Attributes } & \multicolumn{1}{|c|}{ Goal } \\
\cline { 2 - 11 } & Alt & Bar & Fri & Hun & Pat & Price & Rain & Res & Type & Est & WillWait \\
\hline$X_{1}$ & Yes & No & No & Yes & Some & $\$ \$ \$$ & No & Yes & French & $0-10$ & Yes \\
$X_{2}$ & Yes & No & No & Yes & Full & $\$$ & No & No & Thai & $30-60$ & No \\
$X_{3}$ & No & Yes & No & No & Some & $\$$ & No & No & Burger & $0-10$ & Yes \\
$X_{4}$ & Yes & No & Yes & Yes & Full & $\$$ & No & No & Thai & $10-30$ & Yes \\
$X_{5}$ & Yes & No & Yes & No & Full & $\$ \$$ & No & Yes & French & $>60$ & No \\
$X_{6}$ & No & Yes & No & Yes & Some & $\$ \$$ & Yes & Yes & Italian & $0-10$ & Yes \\
$X_{7}$ & No & Yes & No & No & None & $\$$ & Yes & No & Burger & $0-10$ & No \\
$X_{8}$ & No & No & No & Yes & Some & $\$ \$$ & Yes & Yes & Thai & $0-10$ & Yes \\
$X_{9}$ & No & Yes & Yes & No & Full & $\$$ & Yes & No & Burger & $>60$ & No \\
$X_{10}$ & Yes & Yes & Yes & Yes & Full & $\$ \$ \$$ & No & Yes & Italian & $10-30$ & No \\
$X_{11}$ & No & No & No & No & None & $\$$ & No & No & Thai & $0-10$ & No \\
$X_{12}$ & Yes & Yes & Yes & Yes & Full & $\$$ & No & No & Burger & $30-60$ & Yes \\
\hline
\end{tabular}

There are $2 \times 2 \times 2 \times 2 \times 3 \times 3 \times 2 \times 2 \times 4 \times 4=9216$ cases

and two classes $\rightarrow 2^{9216}$ possible hypothesis and many more possible trees!!!! 

\section{Computational}

- It is just not feasible to find the optimal solution

- A bias should be selected to build the models.

- This is a general a problem in Machine Learning. 

\section{Computational}

For decision trees a greedy approach is generally selected:

- Built step by step, instead of building the tree as a whole

- At each step the best split with respect to the train data is selected (following a split criterion).

- The tree is grown until a stopping criterion is met

- The tree is generally pruned (following a pruning criterion) to avoid over-fitting. 

\section{Basic Decision Tree Algorithm}

trainTree (Dataset L)

1. $\mathrm{T}=$ growTree (L)

2. pruneTree $(T, L)$

3. re urn T

growTree (Dataset L)

Removes

subtrees

uncertain

1. T.S

=findBestsplit (L)

2. if $\mathrm{T} . \mathrm{S}==$ null return null

3. (L1, L2) = splitData $(\mathrm{L}, \mathrm{T} . \mathrm{s})$

about their

4. T.left = growTree $($ LI)

5. T.right = growTree (L2)

validity.

6. return T 

\section{Finding the best split}

findBestSplit (Dataset L)

1. Try all possible splits

2. return best

\section{But which one is best?}

![](https://cdn.mathpix.com/cropped/2023_03_29_28353530c600743d6547g-13.jpg?height=1397&width=3229&top_left_y=1129&top_left_x=109)



\section{Split Criterion}

- It should measure the impurity of node:
$i(t)=\sum i=1 \uparrow m$ 綍 $f \downarrow i(1-f \downarrow i)$
Gini impurity

$(\mathrm{CART})$

where $f_{i}$ is the fraction of instances of class $i$ in node $t$

- The improvement of a split is the variation of impurity before and after the split

$\Delta i(t, s)=i(t)-p \downarrow L i(t \downarrow L)-p \downarrow R i(t \downarrow R)$

where $p_{L}\left(p_{R}\right)$ is the proportion of instances going to the left (right) node 

\section{Split Criterion}
$\Delta i(t, s)=i(t)-p \downarrow L i(t \downarrow L)-p \downarrow R i(t \downarrow R)$
$i(t)=2 \times 8 / 12 \times 4 / 12=0,44$
Which one is best?
$p \downarrow L=5 / 12 ; p \downarrow R=$ $7 / 12$
$i(t \downarrow L)=2 \times 2 / 5 \times$
$3 / 5$
$i(t \downarrow R)=2 \times 6 / 7 \times$ $1 / 7$
$\Delta i(t, s)=0.102$
0.011
0.056 