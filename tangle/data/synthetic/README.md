# Synthetic Dataset for Federated Learning

This document intends to explain the synthetic dataset proposed by [1] in an "understandable" way.

Caldas et al. [1] base their synthetic dataset on Li et al.'s [2] who in turn based their synthetic dataset on Shamir et al's [3].

For comprehensibility these three datasets are explained in a chronological order: [3], [2], [1]

## A. Shamir et al. [3]

Generate $` n `$ i.i.d. (independent and identically distributed - sampled from the same distribution) samples $` (x,y) `$.

### Input $` x `$

The input $` x \in \mathbb{R}^{500} `$, is a 500-dimensional vector sampled from $` x \in \mathcal{N}(0,\Sigma) `$, where $` \Sigma_{j,j}=j^{-1.2} `$ is a diagonal covariance matrix. $` x `$ is hence sampled from a multivariate normal distribution with mean 0 and the covariance matrix $` \Sigma `$. $` \Sigma `$ being diagonal (0 everywhere but on the diagonal) makes the (500) dimnesions of the normal distribution linearly independent (each feature in $` x `$ is independent of each other and has hence not any relation to it's other features) with shrinking variances in the dimensions (the values on the diagonal). The smaller the variance in the normal distirbution the more likely a sample is closer to the mean (here 0).

### True Model $` x \to y `$

The "true model" $` x \to y `$ is defined by $` y=sum(x)+\epsilon, \epsilon \sim \mathcal{N}(0,1) `$. \
The relationship between each $` x `$ and it's corresponding label $` y `$ is the sum over all features in x with some gaussian noise (random $` \epsilon `$ sampled from a normal distribution with mean 0 and variance 1) added.

## B. Li et al. [2]

Li et al. impose additional heterogeneity to [3] to make it more suitable for federated learning. They do this by introducing "devices". A device differs from other devices in **(a)** it's "true model" **(b)** the cluster (multivariate normal distribution) from which $` x `$ is sampled has a shifted mean.

Generate $` k `$ devices, each with a set of $` n `$ samples $` (X_k,Y_k) `$, with $` x\in\mathbb{R}^{60} `$ and $` y\in\{0,1\}^{10} `$.

### Input $` x `$

Sample $` n `$ different $` x_k \sim \mathcal{N}(v_k, \Sigma) `$, $` \Sigma_{j,j}=j^{-1.2} `$ is diagonal (see 1.), $` v_k \sim \mathcal{N}(B_k,1) `$, $` B_k \sim \mathcal{N}(0,1) `$.

For each device $` k `$:

1. Sample a 60-dimensional vector $` B_k `$ from a multivariate normal distribution wit mean 0 and the covariance matrix being an identity matrix (all 0 with 1's on the diagonal). 
2. Sample a 60-dim. vector $` v_k `$ from a mult. normal distribution with mean $` B_k `$ and covariance matrix 1 (as above). Each device now has a vector v_k which is sampled from a similar cluster (mult. norm distr. wit cov. matrix 1) but clustered around a different center (mean).
3. Sample $` n `$ different 60-dim. $` x_k `$ from a mult. norm. distr. clustered around $` v_k `$. The clusters shape is as in A defined by $` \Sigma `$. This leads to a device specific set $` X_k `$ with differing means but from a similarly shaped distribution. When normalized the distribution of $` X `$ across all devices should be approx. equal.

### True Model $` x \to y `$

The "true models" describing the relationships between $` x_k `$ and $` y_k `$ for each device $` k `$ is given by \
$` y_k = argmax(softmax(W_kx_k + b_k)) `$, $` W_k \in \mathbb{R}^{10 \times 60} `$, $` b_k \in \mathbb{R}^{10} `$. \
Where $` W_k \sim \mathcal{N}(u_k,1) `$, $` b_k \sim \mathcal{N}(u_k,1) `$, $` u_k \sim \mathcal{N}(0,1) `$.

The true model architecture is an ANN without an activation function with 10 neurons. The output is a 10-dim vector with 0's and a 1 for the highest number coming from the 10 neurons.
The task is hence a multiclass classification (10 classes). The weight matrix $` W_k `$ and the bias vector $` b_k `$ are sampled for each device, such that the true models differ for each device.

For each device $` k `$:

1. Sample a 10-dim vector $` u_k `$ from a mult. norm. distr. with mean 0 and an identity cov-matrix (see **B.1.**).
2. Sample a 1-dim bias vector $` b_k `$ from a mult. norm. distr. with mean $` u_k `$ and cov-matrix 1 (as above).
3. Sample 60 different 10-dim. vectors $` w_k `$ from a mult. norm. distr. with mean $` u_k `$ and cov-matrix 1 (as above). Glue them together to form the $` 60 \times 10 `$ weight matrix $` W_k `$.

This leads to $` k `$ different tasks which weights come from similar distributions (equal in shape, but centered around different means $` u_k `$). This might make the tasks somewhat similar but to different to effectively apply meta-learning. But maybe the introduced heterogeneity is not enough to cripple meta-learning approaches. Surely it mimics the idea of devices differing slightly in their "true model" and their local datasets distributions.

## C. Caldas et al. (LEAF) [1]

Caldas et al. take Li et al.'s [2] rules for building a synthetic dataset and add even more rules to increase heterogeneity with the goal to make the tasks so distinct such that common meta-learning approaches would struggle to effectively learn on the synthetic data set. This is supposedly reached by having device specific "true models" (as in **B**) which are potentially sampled from more than one cluster center (the means from which the model weights are drawn are further apart than in **B**).

Unfortunately there is a discrepancy between the paper [1] and their published github repository introducing their system for public usage [4].

The math describing the synthetic dataset in the paper [1] doesn't add up*, hence the following description is based on the code published in [4].

\* The matrix dimensions don't work out. Given the following "true model" definition:
```math y = argmax(sigmoid(xw + \epsilon))``` $` w `$ cannot be $` (d+1) `$-dimensional for y to be $` s `$-dimensional while $` x `$ is $` (d+1) `$-dimensional.

---

### Input $` x `$

TL,DR; Sampling procedure is same as **B**. Each device has a different amount of samples. The amount is sampled from a lognormal distribution. You can set the dimensionality of $` x `$ to $` d `$.

Each device $` k `$ has a set of samples $` (X_k, Y_k) `$.
The number of samples differs between the devices and is defined by:
$` n_k = min(m_k+5,1000) `$, where $` m_k=Lognormal(3,2) `$. It lies between 5 and 1000.
(Defined in leaf/data/synthetic/main.py)

For each device $` k `$:
Sample $` n_k `$ different $` x_k \sim \mathcal{N}(v_k, \Sigma), x\in \mathbb{R}^d `$, where
$` \Sigma_{j,j}=j^{-1.2} `$ is diagonal (see 1.),
$` v_k \sim \mathcal{N}(B_k,1) `$,
$` B_k \sim \mathcal{N}(0,1) `$.

Add a leading 1 to $` x `$ to account for the "bias" in the "true model". Increasing it's dimensionality to $` x' \in \mathbb{R}^{d+1} `$

Almost the same as in **B** with the only difference, that the amount of samples differs for each device.

---

### True Model $` x \to y `$

Before creating the dataset a probability vector $` p `$ with length $` l `$ must be specified by the user. Such that all entries add up to 1 resulting in a probability distribution.

$` p = (p_1,..,p_j,..p_l), 0 \le p_j \le 1, sum(p) = 1 `$

Each device has it's own "true model":

$` y_k = argmax(softmax^*(x_kw_k + \epsilon)) `$, where:

$` \epsilon \sim \mathcal{N}(0,0.1) `$
$` w_k = Qu_k, w_k \in \mathbb{R}^{d+1\times s} `$
$` Q \sim \mathcal{N}(0,1), Q \in \mathbb{R}^{d+1\times s \times l} `$
$` u_k \sim \mathcal{N}(\mu_j,1),  u_k \in \mathbb{R}^{l} `$
$` \mu_j \sim \mathcal{N}(C_j,1), C_j \sim \mathcal{N}(0,1),  \mu_j \in \mathbb{R}^{l} `$

\* The paper states that it uses a sigmoid function instead of a softmax function. In the end it doenst make any difference since neither sigmoid nor softmax have any influence on the resulting dataset (because of the argmax afterwards).

---

**To create a dataset:**

1. Sample and assign a l-dimensional (length of the $` p `$-vector) vector $` \mu_j `$ to each probability $` p_j `$ in $` p `$. These are supposedly the cluster centers from which the true model weights are being sampled. This does not really matchup in the code. By setting $` l `$ to 1, the "true model" weights are sampled from only one cluster since the $` Q `$ matrix is only 2-dimensional.

For each device $` k `$:

2. Sample $` n_k `$ $` (d+1) `$-dimensional vectors $` x_k `$.
3. Sample one from $` l `$ $` l `$-dimensional vectors $` \mu_j `$ according to the probability distribution $` p `$.
4. Sample a $` l `$-dim. vector $` u_k `$ from a mult. norm. distr. with mean $` \mu_j `$ and an identity cov. matrix (all 0, 1 on diagonal).
5. Collaps $` Q `$ from a 3-dimensional matrix to a 2-dimensional matrix by $` w_k = Qu_k `$.
6. Determine each $` y_k `$ for each $` x_k `$ with $` y_k=argmax(softmax(x_kw_k + \epsilon)) `$. $` \epsilon `$ is just a little gaussian noise, to "reproduce" real world noise in the data.

## D. Our Model

Our model is based on [1] but the task selection is changed in a way that it is more comprehensible.

The only two differnces in the creation of our synthetic dataset are:

1. Sample $` l `$ matrices $` Q_j \sim \mathcal{N}(\mu_j,1), Q_j \in \mathbb{R}^{d+1 \times s} `$, where 
$` \mu_j \sim \mathcal{N}(A_j,1), \mu_j \in \mathbb{R}^1 `$
$` A_j \sim \mathcal{N}(0,1), A_j \in \mathbb{R}^1 `$
Stack these matrices to form $` Q, Q \in \mathbb{R}^{d+1 \times s \times l} `$
2. When generating a new task, sample an identifier ($` l `$-dimensional vector all 0 but the $` j^{th} `$ position is 1, e.g. (0,1,0)) $` j_k `$ according to the probabilities in $` p = (p_1,...,p_j,...,p_l) `$. This states the choice of one slice from $` Q `$.
3. Add some gaussian noise $` \delta `$ to the chosen slice from Q for a little incluster variance between tasks from the same cluster.

$` y_k=argmax(w_kx_k+ \epsilon), y_k \in \mathbb{R}^s,  `$

$` w_k = Qj_k+ \delta_k , w_k \in \mathbb{R}^{d+1 \times s} `$

$` \delta_k \sim \mathcal{N}(0,0.1), \delta_k \in \mathbb{R}^{d+1 \times s} `$

## Setup Instructions (from original Readme)

- `pip3 install numpy`
- `pip3 install pillow`
- Run ```python main.py -num-tasks 1000 -num-classes 5 -num-dim 60``` to generate the initial data.
- Run the ```./preprocess.sh``` (as with the other LEAF datasets) to produce the final data splits. We suggest using the following tags:
  - ```--sf``` := fraction of data to sample, written as a decimal; set it to 1.0 in order to keep the number of tasks/users specified earlier.
  - ```-k``` := minimum number of samples per user; set it to 5.
  - ```-t``` := 'user' to partition users into train-test groups, or 'sample' to partition each user's samples into train-test groups.
  - ```--tf``` := fraction of data in training set, written as a decimal; default is 0.9.
  - ```--smplseed``` := seed to be used before random sampling of data.
  - ```--spltseed``` :=  seed to be used before random split of data.

i.e.
- ```./preprocess.sh -s niid --sf 1.0 -k 5 -t sample --tf 0.6```

Make sure to delete the rem_user_data, sampled_data, test, and train subfolders in the data directory before re-running preprocess.sh

### Notes

- More details on ```preprocess.sh```:
  - The order in which ```preprocess.sh``` processes data is 1. generating all_data (done here by the ```main.py``` script), 2. sampling, 3. removing users, and 4. creating train-test split. The script will look at the data in the last generated directory and continue preprocessing from that point. For example, if the ```all_data``` directory has already been generated and the user decides to skip sampling and only remove users with the ```-k``` tag (i.e. running ```preprocess.sh -k 50```), the script will effectively apply a remove user filter to data in ```all_data``` and place the resulting data in the ```rem_user_data``` directory.
  - File names provide information about the preprocessing steps taken to generate them. For example, the ```all_data_niid_1_keep_64.json``` file was generated by first sampling 10 percent (.1) of the data ```all_data.json``` in a non-i.i.d. manner and then applying the ```-k 64``` argument to the resulting data.
- Each .json file is an object with 3 keys:
  1. 'users', a list of users
  2. 'num_samples', a list of the number of samples for each user, and 
  3. 'user_data', an object with user names as keys and their respective data as values.
- Run ```./stats.sh``` to get statistics of data (data/all_data/all_data.json must have been generated already)
- In order to run reference implementations in ```../models``` directory, the ```-t sample``` tag must be used when running ```./preprocess.sh```

## References

[1] S. Caldas et al., "LEAF: A Benchmark for Federated Settings", arXiv preprint arXiv:1812.01097, 2019, https://arxiv.org/1812.01097.pdf \
[2] T. Li et al., "Fair resource allocation in federated learning", arXiv preprint arXiv:1905.10497, 2019, https://arxiv.org/pdf/1905.10497.pdf \
[3] O. Shamir et al., "Communication Efficient Distributed Optimization using an Approximate Newton-type Method", In International Conference on Machine Learning, 2014, https://arxiv.org/pdf/1312.7853.pdf \
[4] https://github.com/TalwalkarLab/leaf/tree/master/data/synthetic
