# Abstract {-}

\newpage

# Introduction

Organized sports represent a cornerstone of global culture and entertainment. Beyond their intrinsic appeal, the sports industry and its major sporting 
events have had significant impact on economic growth in China [@wu2024], including the emergence and growth of related industries, such as sports betting [@aga2025]. 
In the United States (U.S.), sports betting generated $11.34 billion in revenue from January through September 2024, reflecting a year-over-year growth of 13.4% [@aga2025]. 
This expansion subsequently increases the economic stakes associated with accurately predicting sports outcomes, as both bettors and 
stakeholders seek to leverage data-driven insights for competitive advantage.

As an exemplary domain for sports forecasting, this project tackles the problem of predicting NCAA basketball tournament outcomes through deployment and comparison of naive 
statistical and machine learning (ML) approaches. Our work is framed within the context of the Kaggle competition "March Machine Learning Mania 2025," which tasks 
participants with forecasting match results for both the men's and women's NCAA Division I March Madness basketball tournaments [@mmlm2025]. 
While our primary objective is to maximize predictive performance on the competition leaderboard, we also aim to evaluate the relative strengths 
and limitations of different modeling strategies.

The NCAA March Madness tournaments serve as particularly compelling case studies for sports prediction, shown through the research of @kim2023 
and their application of various ML approaches. These single-elimination competitions determine the national champions in Division I college basketball, 
with each tournament featuring 68 teams competing over approximately three weeks in March and April. Teams qualify through two mechanisms: Automatic bids 
awarded to conference tournament winners, and at-large selections based on regular season performance metrics. The selection committee then assigns seeds 
within four regional brackets, establishing the tournament structure and initial matchups [@ncaa_marchmadness2025].

We approach this challenge by implementing and comparing several modeling frameworks, ranging from traditional statistical methods to state-of-the-art (SOTA)
machine learning methods. Our analysis addresses three central questions: First, what level of predictive accuracy can be achieved using different 
methodological approaches? Second, how do factors such as data availability, feature engineering, and model architecture influence prediction quality? 
Third, are there meaningful differences in predictability between the men's and women's tournaments that reflect underlying competitive dynamics?

## Kaggle Competition

The "March Machine Learning Mania 2025" competition on Kaggle [@mmlm2025] is an annual event that challenges data scientists
and sports enthusiasts to develop models predicting the outcomes of the "March Madness" NCAA Division I basketball championship tournaments. 
Participants are provided with a carefully curated dataset including historical game results, team statistics, and tournament brackets from previous years. 
A submission consists of predicted win probabilities for every possible matchup for both women's and men's tournaments in 2025. 
Every participating team may mark two submissions for evaluation on the public leaderboard of which the better one is considered. 
The evaluation metric used in the competition is the Brier score [@brier1950], which is mathematically identical to the mean squared error between 
predicted probabilities and binary outcomes. The competition started on February 10, 2025, and ended on March 20, 2025, giving teams exactly
one month to develop and refine their models. Also, the competition includes a prize money pool of $50,000 attracting 1,727 teams that 
are listed on the competition's leaderboard.

# Existing Research

The prediction of NCAA Division I basketball tournament outcomes represents a compelling intersection of sports
analytics, statistical modeling, and machine learning. Each year, the NCAA men's and women's basketball tournaments
captivate millions of viewers and generate billions of dollars in legal and illegal wagers [@kvam2006].
Despite this enormous interest and the application of increasingly sophisticated predictive methods, no verifiable
perfect bracket has ever been documented [@mciver2025], with estimated odds of correctly predicting all
tournament games at approximately 1 in 120.2 billion even with basketball knowledge [@sprint2024]. The
single-elimination format and inherent unpredictability of tournament play create a challenging prediction environment
that has motivated researchers to develop approaches ranging from traditional statistical methods to state-of-the-art
machine learning architectures.

## Traditional Statistical Approaches

Early research in NCAA tournament prediction largely relied on seed-based methodologies, exploiting the tournament
selection committee's rankings of teams from 1 (strongest) to 16 (weakest) within each regional bracket.
@stekler2012 demonstrated that seed-based predictions using probit models achieved approximately 72.3%
accuracy across the first four rounds of tournaments from 2003 to 2010, while consensus rankings from multiple polling
sources achieved marginally better performance at 73.6%. However, these seed-based approaches face fundamental
structural limitations. As @stekler2012 notes, "the seedings in the regional rounds cannot be used in making
predictions for these final two rounds" because seeds only provide relative rankings within regions rather than absolute
team strength across the entire tournament field. This constraint necessitates alternative approaches for predicting
Final Four and championship outcomes.

Beyond simple seed comparisons, researchers developed more sophisticated rating systems to estimate team strength.
@kvam2006 compared their proposed model against established systems including the Associated Press poll,
ESPN/USA Today coaches poll, the Ratings Percentage Index (RPI), and the Sagarin and Massey ratings. Each of these
systems attempts to rank teams based on combinations of winning percentage, strength of schedule, and opponent quality.
@shen2016 provides a comprehensive literature review of traditional methods, highlighting the progression
from purely win-loss based approaches to models incorporating margin of victory and opponent strength. The consistent
finding across these traditional statistical approaches is that while seed-based predictions provide a reasonable
baseline, they leave substantial room for improvement through more nuanced team strength estimation.

## Rating Systems and Team Strength Estimation

Rating systems provide dynamic, continuously updated measures of team strength that overcome many limitations of static
tournament seeds or seasonal win-loss records. @kvam2006 introduced a Markov chain model for NCAA basketball
where teams represent states and game outcomes determine transition probabilities. Their approach uses logistic
regression to populate these transition probabilities based on win-loss records, home advantage, and margin of victory,
requiring only basic scoreboard data. The steady-state probabilities of the Markov chain then provide team rankings,
with the intuition that "the current state of the voter corresponds to the team that the voter now believes to be the
best" [@kvam2006].

The Elo rating system, originally developed for chess by Arpad Elo [@Elo1978], has been extensively adapted for
basketball and other team sports. @gomez2024 provides formal mathematical foundations for Elo in sports
contexts, explaining how the system updates team ratings after each game based on the expected versus actual outcome.
The probability that team A defeats team B is modeled as a logistic function of their rating difference, with the victor
gaining rating points (and the loser losing an equal amount) proportional to the upset magnitude. @gomez2024
extends the classical Elo framework through stochastic process formulations that enable score prediction throughout a
game rather than just final outcomes. For basketball specifically, the authors demonstrate that Elo-based systems
achieve competitive predictive performance while maintaining computational simplicity and interpretability.

Dynamic rating systems that separately track home and away performance or incorporate temporal decay of older results
offer further refinements. While @constantinou2013 develops their pi-rating system for football,
their methodology of maintaining separate home and away ratings with learning rates that determine how newly acquired
information updates ratings provides insights applicable to basketball. The key principle, as they state, is that "a
rating system should provide relative measures of superiority between adversaries and overcomes all of the above
complications" of static league tables or tournament seeds [@constantinou2013]. Empirical evidence from
basketball analytics supports these dynamic approaches: @migliorati2021 demonstrates in NBA contexts that
models using Elo ratings or relative win frequencies substantially outperform models based on complex box score
statistics, suggesting that carefully designed single features capturing team strength can be more effective than
high-dimensional statistical aggregations.

## Feature Engineering: Beyond Box Scores

While traditional box score statistics (field goal percentages, rebounds, assists, turnovers) provide obvious predictive
signals, research increasingly demonstrates the value of alternative feature sources. @lopez2015 argues
that Las Vegas point spreads (betting odds) represent highly efficient aggregations of available information, as sportsbooks have
strong incentives to set accurate lines to balance betting action. Their winning entry in the 2014 Kaggle March Machine
Learning Mania competition combined point spread data with possession-based efficiency metrics from Ken Pomeroy's
analytics platform. As they conclude, "we provide evidence that the combination of modest statistical methods with
informative data can meet or exceed the accuracy of more complex models" [@lopez2015]. Pomeroy's efficiency
metrics, detailed in @lopez2015, measure offensive and defensive points per 100 possessions adjusted for
opponent strength, offering normalized team performance measures that account for pace-of-play variations across teams.

A critical insight from recent research concerns data contamination. @yuan2015 explicitly addresses this
issue in their mixture-of-modelers approach, defining contaminated data as "archival data for a given NCAA season which
incorporated the results of the final tournament from that year." Many publicly available datasets and rating systems
update continuously throughout the season, inadvertently including post-tournament information when researchers train
models on historical data. For example, metrics like games played (GP) strongly predict tournament success in historical
data simply because teams that advance further play more tournament games, but this information is not available
pre-tournament for new predictions. @yuan2015 demonstrates that careful use of pre-tournament versions of
rating systems substantially improves genuine predictive performance.

Beyond statistical metrics, @kim2023 emphasizes non-box score factors that influence tournament outcomes.
Their analysis of 1370 tournament games from 2006-2017 incorporates conference affiliation, geographic proximity to 
tournament venues (enabling larger fan support), travel distance and time zone effects on player performance, 
and teams' historical tournament experience. As they note, "little research has focused on situational factors in 
predicting sports tournament outcomes" [@kim2023], yet these contextual elements can substantially impact game results. 
Recent deep learning applications further expand the feature space: @habib2025 combines Elo ratings with GLM-based team
quality metrics derived from historical match results and strength of opposition, demonstrating that sophisticated
feature engineering enhances model performance across multiple architectures.

## Ensemble and Mixture Approaches

Ensemble methods that combine predictions from multiple models have proven particularly effective for tournament
prediction. @yuan2015 documents their mixture-of-modelers approach where a Harvard University team
collectively developed over 30 different models for the 2014 NCAA tournament. These models employed diverse algorithms
including multiple variants of logistic regression (with L1 regularization, L2 regularization, and backward elimination
for feature selection), decision trees, and neural networks. The team optimized predictions using log loss as the
evaluation metric, leading to probability estimates that are well-calibrated instead of merely accurate classifications. 
Eventually, their ensemble strategy improved robustness against overfitting to particular patterns in the training data.

Comparative studies of individual algorithm performance provide context for ensemble benefits. @shen2016
evaluated Support Vector Machines (SVM), Random Forests (RF), and Bayesian models with probability self-consistency
constraints on March Madness data, finding RF achieved 68.2% accuracy, SVM 66.1%, and their Bayesian approach
approximately 50%. Similarly, @kim2023 compared five machine learning approaches on 685 tournament games,
reporting Artificial Neural Networks achieved the highest accuracy at 67%, followed by SVM (65%), k-Nearest Neighbors (
63%), logistic regression (63%), and Random Forests (61%). These relatively modest performance differences across
algorithms, combined with the documented success of ensembles, suggest that model diversity and appropriate feature
engineering may be more important than algorithm selection alone.

A critical consideration for ensemble approaches in tournament prediction is the risk of overfitting to historical
tournament data. As @lopez2015 demonstrates through simulation, even with perfectly accurate game-level
probability estimates, the stochastic nature of tournament brackets means a model might have only approximately 12%
probability of finishing first among hundreds of competitors and less than 50% probability of finishing in the top ten.
This inherent variance motivates ensemble strategies that aggregate across different training approaches to reduce
model-specific overfitting while maintaining predictive accuracy.

## Machine Learning and Deep Learning Approaches

The progression from classical machine learning to deep learning architectures reflects both methodological advancement
and the challenge of limited training data in NCAA contexts. Classical ML comparisons consistently show competitive
performance across algorithms when provided with well-engineered features. @kim2023 reports that Artificial
Neural Networks marginally outperformed other classical algorithms (67% vs 61-65% accuracy), though all approaches
clustered within a narrow performance range. @shen2016 similarly finds Random Forests and SVMs performing
comparably (68% vs 66%). These modest differences suggest that for NCAA prediction, where training data is limited to
historical tournament games (63 games per season and team), the choice among classical ML algorithms matters less than feature
quality and avoiding overfitting.

Recent work has explored deep learning architectures specifically designed for sequential and temporal data.
@habib2025 compares Long Short-Term Memory (LSTM) networks and Transformer models for predicting the 2025
NCAA tournaments using data from 2003-2024. Their comprehensive feature set includes Elo ratings, GLM-based team quality
metrics, tournament seeds, and aggregated box score statistics. Critically, they evaluate models using both Binary
Cross-Entropy (BCE) loss and Brier loss functions, revealing important tradeoffs: Transformer models optimized with BCE
achieved superior discriminative power (AUC of 0.8473), while LSTM models trained with Brier loss demonstrated better
probabilistic calibration (Brier score of 0.1589). As @habib2025 discusses, the choice between maximizing
discrimination versus calibration depends on the specific prediction task and evaluation criteria.

@migliorati2021 examines feature selection through deep learning for NBA prediction, demonstrating that
relatively simple neural network architectures (few layers, modest numbers of units) can achieve strong performance when
the input features effectively capture team strength. Their finding that single features like Elo ratings outperform
complex box score aggregations suggests that deep learning's primary value may lie in learning optimal feature
representations rather than necessarily requiring deep architectures. The development of home/away feature variants
further illustrates how domain knowledge about basketball (home court advantage effects) can be incorporated into neural
network inputs.

Novel approaches continue to emerge. @sprint2024 explores using Large Language Models (LLMs) with social
network data, collecting over 1.1 million tweets from official Division I team Twitter accounts across 2021-23 seasons.
Their approach employs few-shot and zero-shot learning techniques, feeding recent tweets as context to LLMs to predict
game outcomes, and combining LLM-generated embeddings with XGBoost for classification. While innovative, these
cutting-edge methods have not yet demonstrated clear superiority over well-executed traditional approaches, highlighting
the ongoing challenge of balancing methodological novelty with practical prediction performance.

## Model Evaluation and Quantifying Success

Appropriate evaluation metrics are crucial for assessing predictive performance, particularly in probabilistic
forecasting contexts. @stekler2012 employs the Brier Score (also called Quadratic Probability Score), which
measures the mean squared error between predicted probabilities and binary outcomes. For each game, if a model predicts
team A has probability p of winning and A actually wins, the contribution to the Brier Score is $(1-p)^{2}$, while if A loses
it is $p^{2}$ [@brier1950]. This metric explicitly penalizes confident incorrect predictions more heavily than uncertain predictions,
encouraging well-calibrated probability estimates.

@yuan2015 details the log loss evaluation function used in passed Kaggle competitions, defined as the negative
log-likelihood of the observed outcomes given predicted probabilities. Mathematically, for N games with predicted
probability $\hat{y}_i$ and observed outcome $y_i$, log loss equals $-\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$. 
This metric, equivalent to the loss function minimized in logistic regression, heavily penalizes confident incorrect predictions (approaching
infinity as $\hat{y}$ approaches 0 for an observed win). @habib2025 discusses the distinction between
discriminative metrics like AUC-ROC (measuring a model's ability to rank teams correctly) and calibration metrics like
Brier score (measuring whether predicted probabilities match empirical results), noting that optimization for one
may not guarantee optimization for the other.

## Synthesis and Research Gaps

The existing research reveals several consistent themes across methodologies and time periods. First, feature
engineering and data quality consistently emerge as more important than algorithm selection. @lopez2015's success with 
"modest statistical methods" combined with Las Vegas spreads and @migliorati2021's finding that simple Elo ratings
outperform complex box score models both underscore this principle. Second, ensemble approaches that aggregate
predictions across diverse models generally outperform individual models, as @yuan2015's mixture-of-modelers work
demonstrates. Third, proper evaluation requires careful attention to both the metrics employed and the role of
stochastic variation in tournament outcomes.

Several limitations persist across the literature. The scarcity of training data remains fundamental—with only 63
tournament games per season and team, and structural changes in team composition from year to year, models risk overfitting to
small sample statistical noise. The data contamination issue highlighted by @yuan2015 suggests that reported historical
performance may overstate true predictive capability when researchers inadvertently include post-tournament information.
The persistent difficulty in predicting upsets, particularly in early rounds where lower-seeded teams occasionally
defeat favorites, indicates that current approaches may not fully capture the factors driving individual game variance.

# Data {#sec:data}

## Exploratory Analysis

## Feature Engineering
Before training the machine learning models, several features were engineered from the raw game data. The aim of these features was to capture the 
performance of a team, both in their athletic abilities and mental strength.

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

### Team Quality

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

## Classical Machine Learning Models {#sec:classical-machine-learning-models}
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
As a final modelling approach, deep learning techniques were explored. The main idea was that a deep enough neural network could extract more features from the already existing ones and thus make better predictions than the classical machine learning models (@sec:classical-machine-learning-models), especially on the larger datasets (@sec:dataset-preparation).

To experiment with deep learning approaches, a flexible neural network architecture was implemented using PyTorch Lightning [@Falcon2019]. The neural network framework supports various architectural configurations and training strategies to predict win probabilities.

### Architecture {#sec:neural-network-architecture}
The architecture used is a deep neural network with theoretically any amount of fully connected layers. To experiment with different variations of that architecture, the depth of the network and the width and activation of each layer were made configurable as hyperparameters.

![Neural network architecture](./images/neural-network/architecture_light.png){#fig:neural-network-architecture width=20%}

@fig:neural-network-architecture shows a schematic of the neural network architecture used. The input can be any of the features described in @sec:dataset-preparation, which are then passed through multiple blocks of linear layers, ending in a sigmoid activation, which then represents the win probability prediction.

### Loss Functions {#sec:loss-functions}
Next to the two standard loss functions, mean squared error (MSE) and binary cross-entropy (BCE), a custom loss function with the goal to force the model to make over confident predictions. Inspiration for this were the winning solutions of the Kaggle competition, which manually push confident predictions to be even more confident. One of these examples is the 1st place solution by @odeh2025marchMLMania.

The idea was to use BCE with a penalty term, that increases the loss for predictions that far from the target. A fitting penalty term seemed to be the entropy [@shannon1948a; @shannon1948b], which represents the uncertainty of a random variable, in this case the win probability.

![Comparison of BCE, entropy and a combination](./images/neural-network/bce-entropy-combination.png){#fig:bce-entropy-combination width=50%}

@fig:bce-entropy-combination shows the curve of the BCE loss, the entropy and an addition of the two given the predictions for a true label of $1$. As can be seen by the combination of the BCE and the entropy, the loss is increased considerably for uncertain predictions (near $0.5$), while confident predictions (near $0$ or $1$) are only slightly affected. Based on this, the following loss funtion was defined:

$$
L = \text{BCE}(y_{pred}, y_{true}) + \lambda H(y_{pred})
$$

where $H(y_{pred}) = -y_{pred} \log(y_{pred}) - (1-y_{pred}) \log(1-y_{pred})$ is the entropy of the prediction, and $\lambda$ is a configurable weight. To find a $\gamma$ for which the loss of confident predictions is the most distinct, while maintaining a monotonically decreasing loss towards the true label, a binary search was conducted, which lead to $\gamma \approx 3.592$, which we reduced to $\gamma = 3.5$ for simplicity. The graph of the final loss function can be seen in @fig:bce-with-entropy-penalty. 

![BCE with entropy penalty loss function](./images/neural-network/bce-with-entropy-penalty-loss-function.png){#fig:bce-with-entropy-penalty width=50%}

### Training Configuration
With the architecture as hyperparameter approach the training could be defined by the following groups of hyperparameters.

| Hyperparameter | Description |
|---------------|----------------------------------------------------|
| Architecture | The architecture of the neural network was described in @sec:neural-network-architecture. Specified as an array of widths of linear layers, where the length of the array determines the depth of the network. |
| Learning Rate | The learning rate together with learning rate schedulers and their specific parameters. Used schedulers include StepLR, ReduceLROnPlateau, ExponentialLR and CosineAnnealingLR. |
| Loss Function | The objective function used to train the model. Possible values included MSE, BCE, and the BCE with entropy penalty described in @sec:loss-functions. |
| Optimization | The framework supports various optimizers, however, the optimizer used for all experiments was AdamW. |
| Regularization | Dropout layers and weight decay were used to counteract overfitting. |
: Main neural network training hyperparameters {#tbl:neural-network-hyperparameters}

@tbl:neural-network-hyperparameters summarizes the main hyperparameters used to configure the training, it is not an exhaustive list of all possible hyperparameters. The implementation supported even more hyperparameters, such as different weight initialization strategies, and more values for certain hyperparameters, like the mean absolute error loss function, but for simplicity these were not considered during this project.

The training procedure includes early stopping based on the validation loss, as well as model checkpointing. The progress of the training was tracked using Weights & Biases [@wandb], which ultimately served as the tool to select the best model out of multiple training runs.
 
# Experiment

# Results

# Discussion

# Conclusion

Taken from Existing Research, but think rephrased would fit better here:

> Our research addresses several gaps in this existing literature. While previous work often focuses on single
> methodological paradigms (pure statistical, classical ML, or deep learning), we provide systematic comparison across
> these approaches using identical feature sets and evaluation protocols. Our temporal ensemble strategy, where models are
> trained separately on different historical seasons and predictions averaged, directly addresses overfitting concerns
> while maintaining the ensemble diversity benefits documented by @yuan2015. We incorporate comprehensive feature
> engineering informed by the non-box score factors emphasized by @kim2023, the dynamic rating principles from @kvam2006
> and @constantinou2013, and the awareness of data contamination from @yuan2015. Finally, our parallel analysis of both
> men's and women's tournaments enables investigation of whether predictive patterns and optimal methodologies generalize
> across these related but distinct competitive environments.

@TODO: @LucaDave - future work/ideas
- Additional data sources
- CNN (Consider 1 regular season of a team 1 "image" or rather one depiction of a teams performance? Feed lags of team games as input to NN)
- Time series aspect


\newpage

# References {-}
