# Abstract {-}

# Introduction

# Data {#sec:data}

## Exploratory Analysis

## Feature Engineering
Before training the machine learning models, several features were engineered from the raw game data. The aim of these features was to capture the performance of a team, both in their athletic abilities as well as their mental strength.

### ELO Rating {#sec:elo-rating}
[@Elo1978]
@TODO: @Dave - needs checking and amending. Plus reference at the right spot

The ELO rating system is a method for calculating the relative skill levels of players or teams in competitive games. In the context of NCAA basketball, each team is assigned an ELO rating that is updated after each game based on the outcome and the expected probability of winning. The win probability for a matchup between team $A$ and team $B$ is calculated using the logistic function:

$$
P(A \text{ beats } B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

where $R_A$ and $R_B$ are the ELO ratings of teams $A$ and $B$ respectively. After each game, the ELO ratings are updated according to:

$$
R_A^{new} = R_A^{old} + K \cdot (S_A - P(A \text{ beats } B))
$$

where $K$ is a constant that determines how much ratings change after each game (typically set between 16 and 32), and $S_A$ is the actual outcome (1 for a win, 0 for a loss). The same update is applied symmetrically to team $B$.

### Quality

### ELO Delta Sliding Window
The ELO Delta Sliding Window feature captures the change in a team's ELO rating over a specified window of games. The idea behind the it was to capture the mentality of a team, as a team's confidence, and with that their performance, might increase or decrease with the change in ELO. The ELO delta is calculated as such: $\Delta \text{R}_w = \text{R}_{\text{current}} - \text{R}_{w}$, where $\text{R}_{\text{current}}$ is the team's ELO rating at the current game and $\text{R}_{w}$ is their ELO rating $w$ games prior.

Different window sizes $w$ were considered and ultimately chosen with a grid search that tries to maximize the improvement of the brier score of the raw ELO predictions in @sec:elo-rating. The following formula was used to calculate the predictions:

$$
P(A \text{ beats } B) = \frac{1}{1 + 10^{(R_B + \omega \cdot \Delta R_{B,w} - R_A + \omega \cdot \Delta R_{A,w})/400}}
$$

Where $\omega$ is a weight for the delta adjustment. Given this formula the difference of the brier score between the adjusted ELO predictions and the raw ELO predictions was calculated for different window sizes $w$ and weights $\omega$. The pair that maximizes $\text{Brier}_{\text{raw}} - \text{Brier}_{\text{adjusted}}$ was then chosen to use for the final feature; window size $w = 3$ with a weight of $\omega = 0.1$.

### Win Streaks


## Dataset Preparation {#sec:dataset-preparation}

### Season Averages {#sec:season-averages}

### Weighted Season Averages {#sec:weighted-season-averages}

### Sliding Window Averages {#sec:sliding-window-averages}


# Methods
This section describes the various approaches used during the project, starting with statistical approaches to several machine learning methods.

## Statistical Approaches
To establish a baseline for our machine learning approaches, we implemented several statistical approaches.All of these models were based on the entire compact regular or tourney season results described in @sec:data.

### Random
In the context of this project the random baseline considered was a prediction of a 50% win probability for each team of every matchup in the 2025 tournament. This does not take into account any data at all, and serves as the baseline of our statistical approaches.

### Seed Ratio
With the seed ratio approach, the mean seed of a team is calculated over all the past tournaments in the dataset. If a team has not participated in any tournaments, a seed of 32 is assigned to it, which is below the worst normal seed of 16. For each matchup of the 2025 tournament that seed is then used to calculate the win probability of a team as follows:
$$
P(A \text{ beats } B) = \frac{\mu_B}{\mu_A + \mu_B}
$$

Where $\mu$ is the mean seed of the respective team, and $\mu = 32$ if the team has not participated in any tournaments.

### Win Ratio {#sec:win-ratio}
The win ratio approach is similar to the seed ratio approach, but instead of using the mean seed of a team, the win ratio over all past regular and tournament games is used. The win rate $wr$ is calculated as follows:
$$
wr = \frac{w + \alpha }{w+l + \alpha \cdot d}
$$

Where $w$ is the number of wins and $l$ is the number of losses of a team over all games in the @sec:data. To account for small sample sizes, we applied Laplace smoothing [@Laplace1814] with $\alpha = 5$ and $d = 2$. This would result in a $wr = 0.5$ for a team without any games played. The win probability for a matchup between team $A$ and $B$ is then calculated as follows:
$$
P(A \text{ beats } B) = \frac{wr_A}{wr_A + wr_B}
$$

### Head-to-Head Ratio
The head-to-head ratio approach is an extension of the win ratio approach, but instead of using the overall win ratio of a team, the head-to-head win ratio between two teams is used if they have played against each other a minimum number of games. The head-to-head win rate $hwr$ for teams $A$ and $B$ is calculated analogous to @sec:win-ratio:
$$
hwr_{A,B} = \frac{w_{A,B} + \alpha }{w_{A,B}+l_{A,B} + \alpha \cdot d}
$$

Where $w_{A,B}$ is the number of wins of team $A$ against team $B$ and $l_{A,B}$ is the number of losses of team $A$ against team $B$. The same Laplace smoothing [@Laplace1814] with $\alpha = 5$ and $d = 2$ is applied. If teams $A$ and $B$ have played less games against each other than a given threshold, the overall win ratio as described in @sec:win-ratio is used instead. The win probability for a matchup between team $A$ and $B$ is then calculated as follows:
$$
P(A \text{ beats } B) = 
\begin{cases}
hwr_{A,B} & \text{if } w_{A,B} + l_{A,B} >= c \\
\frac{wr_A}{wr_A + wr_B} & \text{otherwise}
\end{cases}
$$

Where $c$ is the minimum number of games played between two teams to use the head-to-head win ratio, which was set to $c=3$ for this project.

### Point Ratio
The point ratio approach predicts matchup outcomes based on the average points scored by teams across all past regular season and tournament games. For each team, a discounted mean score is calculated where recent games are weighted more heavily than older games. Additionally, tournament games can be weighted differently than regular season games to account for their potentially higher significance. The discounted score $ds$ for a team is calculated as:
$$
ds = \frac{1}{n} \sum_{i=1}^{n} s_i \cdot w_i \cdot \gamma^{y - y_i}
$$

Where $s_i$ is the score in game $i$, $w_i$ is the weight for the game type (regular season or tournament), $\gamma$ is the discount factor, $y$ is the current season, $y_i$ is the season of game $i$, and $n$ is the total number of games. For this project, $w_{regular} = 0.75$, $w_{tourney} = 1.0$, and $\gamma = 0.9$ was used. The actual win probability for a matchup between team $A$ and $B$ is then similarly to previous methods:
$$
P(A \text{ beats } B) = \frac{ds_A}{ds_A + ds_B}
$$

## Classical Machine Learning Models
To improve upon the statistical baselines, several classical machine learning models were experimented with.

### Models
Machine learning models of the following types were trained during the course of this project:
- Logistic Regression [@Cox1958]
- Support Vector Machine (SVM) [@Cortes1995]
- Random Forest [@Breiman2001]
- XGBoost [@Chen2016]
- CatBoost [@Prokhorenkova2018]

XGBoost and CatBoost were implemented using their respective Python libraries [@xgboost-website; @catboost-website], while scikit-learn [@scikit-learn-website] was used for the other models. Each of these models has its own set of hyperparameters that were considered during their respective experiments.

### Ensemble Training Strategy
For these classical machine learning models, an ensemble training approach was implemented where multiple models are trained on different seasons independently, and their predictions are averaged during inference. This temporal ensemble strategy tries to counteract overfitting to a single season.

@TODO: @Dave - you'll no better how to elaborate on this

## Neural Networks
As a final modelling approach, deep learning techniques were explored.
@TODO: @Luca - might be good to mention the reasoning for this choice and what we hoped to achieve (NN as feature extractors, capture more complex patterns, etc.)

To experiment with deep learning approaches, a flexible neural network architecture was implemented using PyTorch Lightning [@Falcon2019]. The neural network framework supports various architectural configurations and training strategies to predict win probabilities.

### Architecture
@TODO: @Luca - add a nice graphic of the architecture here
The neural network architecture is fully configurable through hyperparameters:

@TODO: @Luca - add more details here about the architecture, number of layers, activations, etc.

### Loss Functions
Next to the two standard loss functions, mean squared error and binary cross-entropy (BCE), a custom loss function with the goal to force the model to make over confident predictions. Inspiration for this were the winning solutions of the Kaggle competition, which manually push confident predictions to be even more confident. @TODO: @Luca - add references

The idea was to use BCE with a penalty term, that increases the loss for predictions that far from the target. A fitting penalty term seemed to be the entropy [@shannon1948a; @shannon1948b], which represents the uncertainty of a random variable, in this case the prediction of the win probability.

![Comparison of BCE, entropy and a combination](./images/bce-entropy-combination.png){#fig:bce-entropy-combination width=50%}

@fig:bce-entropy-combination shows the curve of the BCE loss, the entropy and an addition of the two given the predictions for a true label of $1$. As can be seen by the combination of the BCE and the entropy, the loss is increased considerably for uncertain predictions (near $0.5$), while confident predictions (near $0$ or $1$) are only slightly affected. Based on this, the following loss funtion was defined:

$$
L = \text{BCE}(y_{pred}, y_{true}) + \lambda H(y_{pred})
$$

where $H(y_{pred}) = -y_{pred} \log(y_{pred}) - (1-y_{pred}) \log(1-y_{pred})$ is the entropy of the prediction, and $\lambda$ is a configurable weight. To find a $\gamma$ for which the loss of confident predictions is the most distinct, while maintaining a monotonically decreasing loss towards the true label, a binary search was conducted, which lead to $\gamma \approx 3.592$, which we reduced to $\gamma = 3.5$ for simplicity. The graph of the final loss function can be seen in @fig:bce-with-entropy-penalty. 

![BCE with entropy penalty loss function](./images/bce-with-entropy-penalty-loss-function.png){#fig:bce-with-entropy-penalty width=50%}

### Training Configuration
@TODO: @Luca - add more details here about training configuration, optimizers, schedulers, etc.

# Experiments

# Results

# Discussion

# Conclusion

@TODO: @LucaDave - future work/ideas
- Additional data sources
- CNN (Consider 1 regular season of a team 1 "image" or rather one depiction of a teams performance?)
- Time series aspect


\newpage

# References {-}
