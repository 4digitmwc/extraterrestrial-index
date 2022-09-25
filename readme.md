# The Conceptualization of ETI Model for Outlier Detection in 4 Digit MWC 2023

*"Appreciate the incompleteness as it is."*

## Introduction

This article will go over a summary of the estimation of **Extraterrestrial Index (ETI)** from **4 Digit MWC 4 Data**. It is going to be a baseline model for the performance measurement of **4 digit mwc 4 participants** in order to detect the **outliers** or **outperformers** from the list of participants.

## Data

The baseline data was obtained from 4 digit mwc 4 dataset. We collected scores from individual players who are participated from Qualifiers to Finals Week 2. The beatmaps have been categorized into 3 categories, RC, HB and LN. Which will benefit in feature extractions using ETI in this process.

## ETI Model

The ETI Model is idealized from the concept of "average performance" of a given player. We then calculate the average standardized scores which have been transformed (normalized) in some degree. Then we intend to capture the outliers by considering the right tail end of the ETI distribution.

### Assumptions

- The scores of every map from every player is **independently identically distributed (iid)**, which means the scores are distributed in one probability distribution and for all pair of **beatmap A** and **beatmap B**, the score set of these two beatmaps are independent from each other. We assume the score in each map is **Normally Distributed**.
- The scores from each player will be **randomly censored** with the same probability for every map. This assumption is still yet to be generalized later but overall this assumption already satisfied on Ranked / Loved beatmaps.

### Model Derivations

We first assume the follows for the simplified model of ETI:

- There is no censored scores by a single player in every beatmap.

This assumption is an ideal one but it will help us grasp the idea of the model better when we create a model with [General Assumptions](#assumptions).

At first, the ETI was intended to be Log-Normal Distribution with the Normal Distribution being standardized (expectation=0, variance=1). We then use the first assumption to assume that the scores are normally distributed with expected value = 0 and variance = 1 and assume that there are k maps, we then consider the average standardized scores of a player in every map:

$$
X = \frac{x_1 + x_2 + x_3 + ... + x_k}{k} = \frac{\sum x_i}{k}
$$

Where $x_i$ are standardized scores of a player, note that this is a random variable.

We then calculate the expectation and the variance of $X$

$$
\mathbb{E}[X] = \mathbb{E}\left[\frac{\sum x_i}{k}\right] = \frac{\sum \mathbb{E}[x_i]}{k} = 0
$$

$$
\text{Var}[X] = \text{Var}\left[\frac{\sum x_i}{k}\right] = \frac{1}{k^2} \text{Var}\left[\sum x_i\right]
$$

from one of the assumptions that $x_i$ is independent, we get:

$$
\text{Var}[X] = \frac{1}{k^2} \text{Var}\left[\sum x_i\right] = \frac{1}{k^2} \cdot k = \frac{1}{k}
$$

According to **Central Limit Theorem**, we know that $X$ is **Normally Distributed** random variable with Expectation 0 and Variance 1/k.

In order to get the Variance of 1, we need to standardize them by multiplying $\sqrt{k}$, which is the inverse of standard deviation of random variable $X$. We then exponentiate the random variable and get:

$$
\text{ETI} = e^{\sqrt{k} \cdot \text{Avg}[x_i]}
$$

Where:

- $k$ is the number of beatmaps
- $x_i$ is the scores of a given player on each beatmap

That was a beautiful derivation isn't it ? We went through some little theory of Mathematical Statistics and now we have some brief formula for ETI. Anyway this is just for an ideal assumption where scores of all players from all beatmaps are available. Next, we will consider the case where the scores are censored with constant probability $p$.

The next part which will come in handy is missing data validation. We then consider the **worse-case environment** where the hidden score of a player is less than or equal to the minimum of all scores, so the imputation function can be written as follows:

$$
f(x_{ip}) = 
\begin{cases}
x_{ip} & x_{ip} \neq \text{NaN}\\
\min{x_i} & \text{otherwise.}
\end{cases}
$$

Where $x_{ip}$ is the score of a player $p$ in a map $i$.

We can modify the formula of our ETI as follows:

$$
\text{ETI}_p = e^{\sqrt{k} \cdot \text{Avg}_i[f(x_{ip})]}
$$

Which is the final formula for our ETI model.

There is a little concern about how the formula will violate the assumptions that the input should be independently identically distributed. However this concern should be addressed in a discussion.

### Model Procedure

From the derivation, we then can divide the model into three major parts:

- **Imputation**: We first impute the censored scores with the minimum scores of all players in a map.

- **Standardization**: We then scale the scores into the distribution with mean 0 and variance 1.

- **Evaluation**: We then average the standardized scores of each player, then multiply by the square root of the numbers of beatmaps. After that, we exponentiate them and obtain the ETI of each players.

There are additional adjustments which is needed to add in the model, the adjustments are as follows:

- Seperate the ETI between beatmap categories in order to get the 3d vectors, then use these features for evaluating the outliers.
- There is a case where there are players who don't play some categories, with that case we replace the ETI with the minimum of ETI of that category.

### Experimentation on Ideal Assumptions

We then do the **Monte-Carlo Simulation** on the model in order to obtain the expected values of ETI from the model, as it is too complicated to compute it explicitly.

In order to conduct the experiments, we need to obtain the following values:

- Number of the participants
- The rate of censoring for each beatmap category

We then run the simulation to get the approximate expectation, which are:

- The expectation for ETI of RC maps is approximately 1.75684
- The expectation for ETI of HB maps is approximately 1.76916
- The expectation for ETI of LN maps is approximately 1.80041
- The expectation for geometric mean of ETI of all categories is approximately 1.18962

## Results

We then tried the model with 4 Digit MWC 4 Dataset, the results are as the following figures:

- **Figure 1** is the top 15 geometric average ETI of players in each category. We can see that the numbers are inflated
![](https://cdn.discordapp.com/attachments/1017070848177344553/1021033873188073492/unknown.png)

- **Figure 2** is the natural logarithm of the top 15 ETI of players, basically just a Transform procedure without explonentiating the mean.
![](https://cdn.discordapp.com/attachments/1017070848177344553/1021033919145054208/unknown.png)

- **Figure 3** is the summary of the result we obtained from Figure 1. We can see the inflation of the values and unbalanced variances
![](https://cdn.discordapp.com/attachments/1017070848177344553/1021034071322792006/unknown.png)

- **Figure 4** is the summary of logarithm of the result, we can see clearer that the log of ETI we obtained has the similar variance but little difference in average or approximate expectations
![](https://cdn.discordapp.com/attachments/1017070848177344553/1021034518267830293/unknown.png)

- **Figure 5** is the correlation matrix of log ETI in rice, ln and hybrid, we can see that there is a high chance that there is a multicollinearity in Hybrid and LN and moderate correlation between rice and hybrid. There is a low-moderate correlation between rice and LN too.
![](https://cdn.discordapp.com/attachments/1017070848177344553/1021034735981568070/unknown.png)

- **Figure 6 and 7** is the linear regression plot with scatter plot of Log ETI of RC and LN, HB and LN respectively

![](https://cdn.discordapp.com/attachments/1017070848177344553/1021314408087048272/unknown.png)

![](https://cdn.discordapp.com/attachments/1017070848177344553/1021314408586158120/unknown.png)

- **Figure 8** is the scatter plot of the relation between log ETI of RC and LN, we can see the increase of variances when the log of ETI of either beatmap category increases. This is called **Heteroscedasticity** in Linear Regression Model.

![](https://cdn.discordapp.com/attachments/1017070848177344553/1021314308820451369/unknown.png)

The correlation of three categories can be explained as follows:

- Since **HB** has the mixed elements between **RC** and **LN**, the expected skillset of **HB** tend to have the correlation between **RC** and **LN** as well.
- The heteroscedasticity of **RC** and **LN** can lead to the prerequisites of a given player in order to play a given difficulty of LNs, ie. some LNs require Rice skills but also require some experiences from a given player. To elaborate further, we know that there are Rice players who can perform well in Rice but have no experiences on LN and there are players who play both categories well. Hence this can be explained by the behavioral nature of players.

## Further Questions

- Does the Data Imputation procedure address the censored data case well ? Is there any edge case where the minimum doesn't work well ? If so, is there any other approximation for the missing data validation ?
- Does the ETI Model capture the feature of difficulty of the beatmaps as well ?
- Can ETI be used directly as a classifier for outliers, or we need to use more models to address the classification ?
- If we use the parametric models with ETI in order to capture the outperformers, how are we going to address the multicollinearity occured in each beatmap category ?
