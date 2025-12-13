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
reporting Neural Networks achieved the highest accuracy at 67%, followed by SVM (65%), k-Nearest Neighbors 
(63%), logistic regression (63%), and Random Forests (61%). These relatively modest performance differences across
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
performance across algorithms when provided with well-engineered features. @kim2023 reports that 
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
relatively simple Neural Network architectures (few layers, modest numbers of units) can achieve strong performance when
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

The dataset used for this project was obtained from the Kaggle competition "March Machine Learning Mania 2025" [@mmlm2025]. 
It consists of a total of 36 CSV files containing various information about NCAA Division I basketball games, teams, venues, coaches and tournaments.

## Data Description

For this project, only a subset of the data available was used. All data used contains historical information up to and including the 2025 regular season. These are described in the following sections.

### Regular Season Detailed Results {#sec:regular-season-detailed-results}

The regular season detailed results contain basic game information as well as box-score statistics for every NCAA Division I basketball game played in the regular season. For men's basketball the available data start at the 2003 season and for women's basketball at the 2010. @tbl:regular-season-detailed-results describes the features available in this dataset. Features containing \[WL\] in their name are available for both the winning and losing team of a game, e.g. "WTeamID" and "LTeamID" for the winning and losing team IDs respectively.

| Feature      | Description                              |
|--------------|------------------------------------------|
| Season       | The season in which the game was played  |
| DayNum       | The day number within the season (1-133) |
| \[WL\]TeamID | The ID of the team                       |
| \[WL\]Score  | The score of the team                    |
| WLoc         | The location of the winning team (H/A/N) |
| NumOT        | Number of overtime periods played        |
| \[WL\]FGM    | Field Goals Made                         |
| \[WL\]FGA    | Field Goals Attempted                    |
| \[WL\]FGM3   | 3-Point Field Goals Made                 |
| \[WL\]FGA3   | 3-Point Field Goals Attempted            |
| \[WL\]FTM    | Free Throws Made                         |
| \[WL\]FTA    | Free Throws Attempted                    |
| \[WL\]OR     | Offensive Rebounds                       |
| \[WL\]DR     | Defensive Rebounds                       | 
| \[WL\]Ast    | Assists                                  |
| \[WL\]TO     | Turnovers                                |
| \[WL\]Stl    | Steals                                   |
| \[WL\]Blk    | Blocks                                   |
| \[WL\]PF     | Personal Fouls                           |

: Features of the Regular Season Detailed Results {#tbl:regular-season-detailed-results}

The men's dataset contains a total of 118,882 unique games and the women's dataset 81,708 respectively.

### Tournament Detailed Results {#sec:tournament-detailed-results}

The tournament detailed results contains the same features as described in @sec:regular-season-detailed-results, however for games played in the NCAA Division I basketball tournaments. For men's basketball the available data starts at the 2003 season and for women's basketball at the 2010 respectively. The features are identical to those described in @tbl:regular-season-detailed-results except for `DayNum` which ranges from 134 to 154, representing the tournament game days.

The men's dataset contains a total of 1,382 unique games and the women's dataset 894 respectively.

### Tournament Seeds

The tournament seeds dataset contains the seed information for every team that participated in the NCAA Division I basketball tournaments. As can be seen in @tbl:tournament-seeds, the dataset contains the season, seed and team ID for every team that participated in the tournament for that season. For the men the available data starts at the 1985 season and for the women at the 1998 season respectively, however only data overlapping the range of the detailed results in @sec:regular-season-detailed-results and @sec:tournament-detailed-results was used.

| Feature     | Description                              |
|-------------|------------------------------------------|
| Season      | The season in which the game was played  |
| Seed        | The seed of the team in the tournament   |
| TeamID      | The ID of the team                       |

: Features of the Tournament Seeds {#tbl:tournament-seeds}

The seed feature additionally contains the conference region of the team as a prefix (e.g. `W01` for West region, seed 1). This was stripped and only the numerical seed value was used for this project.

## Exploratory Analysis

![Regular Season Games Overview](./images/pda/MWGamesOverview.png){#fig:regular-season-games-overview width=80%}

::: {#fig:score-analysis}
![Men](./images/pda/MScoreAnalysis.png){width=45%}
![Women](./images/pda/WScoreAnalysis.png){width=45%}

Score Analysis for both (a) men and (b) women.
:::

![Game Location Analysis](./images/pda/MWLocationAnalysis.png){#fig:location-analyis width=80%}

::: {#fig:seed-upsets}
![Men](./images/pda/MSeedUpsets.png){width=45%}
![Women](./images/pda/WSeedUpsets.png){width=45%}

Upset rate by seed for both (a) men and (b) women.
:::

::: {#fig:win-margin-correlation}
![Men](./images/pda/MCorrWinMarg.png){width=45%}
![Women](./images/pda/MCorrWinMarg.png){width=45%}

Pearson correlation of features and win margin for both (a) men and (b) women.
:::


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
The ELO Delta Sliding Window feature captures the change in a team's ELO rating over a specified window of games. The idea behind it was to capture the mentality of a team, as a team's confidence, and with that their performance, might increase or decrease with the change in ELO. The ELO delta is calculated as such: $\Delta \text{R}_w = \text{R}_{\text{current}} - \text{R}_{w}$, where $\text{R}_{\text{current}}$ is the team's ELO rating at the current game and $\text{R}_{w}$ is their ELO rating $w$ games prior.

Different window sizes $w$ were considered and ultimately chosen with a grid search that tries to maximize the improvement of the brier score of the raw ELO predictions in @sec:elo-rating. The following formula was used to calculate the predictions:

$$
P(A \text{ beats } B) = \frac{1}{1 + 10^{(R_B + \omega \cdot \Delta R_{B,w} - R_A + \omega \cdot \Delta R_{A,w})/400}}
$$

Where $\omega$ is a weight for the delta adjustment. Given this formula the difference of the brier score between the adjusted ELO predictions and the raw ELO predictions was calculated for different window sizes $w$ and weights $\omega$. The pair that maximizes $\text{Brier}_{\text{raw}} - \text{Brier}_{\text{adjusted}}$ was then chosen to use for the final feature; window size $w = 3$ with a weight of $\omega = 0.1$.

### Win Streaks

## Feature Importance {#sec:feature-importance}

### Default Features {#sec:default-features}


## Dataset Preparation {#sec:dataset-preparation}

### Season Averages {#sec:season-averages}
87 features (ranked)
67 data points per gender per season
2345 total data points (matchups)

### Weighted Season Averages {#sec:weighted-season-averages}
87 features (ranked)
405’732 total samples (matchups)

### Sliding Window Averages {#sec:sliding-window-averages}
81 features (ranked)
No seed & streak features
395’918 total samples (matchups)


# Methods
This section describes the various approaches used during the project, starting with statistical approaches to several machine learning methods.

## Statistical Approaches {#sec:statistical-approaches}
To establish a baseline for our machine learning approaches, we implemented several statistical approaches. All of these models were based on the entire compact regular or tourney season results described in @sec:data.

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

### Ensemble Training Strategy {#sec:ensemble-training-strategy}
For these classical machine learning models, an ensemble training approach was implemented where multiple models are trained on different seasons independently, and their predictions are averaged during inference. This temporal ensemble strategy tries to counteract overfitting to a single season.

@TODO: @Dave - you'll no better how to elaborate on this

## Neural Networks
As a final modelling approach, deep learning techniques were explored. The main idea was that a deep enough neural network (NN) could extract more features from the already existing ones and thus make better predictions than the classical machine learning models (@sec:classical-machine-learning-models), especially on the larger datasets (@sec:dataset-preparation).

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

+------------------+---------------------------------------------------------------------------------------------+
| Hyperparameter   | Description                                                                                 |
+==================+=============================================================================================+
| Architecture     | The architecture of the neural network was described in @sec:neural-network-architecture.   |
|                  | Specified as an array of widths of linear layers, where the length of the array determines  |
|                  | the depth of the network.                                                                   |
+------------------+---------------------------------------------------------------------------------------------+
| Learning Rate    | The learning rate together with learning rate schedulers and their specific parameters.     |
|                  | Used schedulers include StepLR, ReduceLROnPlateau, ExponentialLR and CosineAnnealingLR.     |
+------------------+---------------------------------------------------------------------------------------------+
| Loss Function    | The objective function used to train the model. Possible values included MSE, BCE, and the  |
|                  | BCE with entropy penalty described in @sec:loss-functions.                                  |
+------------------+---------------------------------------------------------------------------------------------+
| Optimization     | The framework supports various optimizers, however, the optimizer used for all experiments  |
|                  | was AdamW.                                                                                  |
+------------------+---------------------------------------------------------------------------------------------+
| Regularization   | Dropout layers and weight decay were used to counteract overfitting.                        |
+------------------+---------------------------------------------------------------------------------------------+

: Main neural network training hyperparameters {#tbl:neural-network-hyperparameters}

@tbl:neural-network-hyperparameters summarizes the main hyperparameters used to configure the training, it is not an exhaustive list of all possible hyperparameters. The implementation supported even more hyperparameters, such as different weight initialization strategies, and more values for certain hyperparameters, like the mean absolute error loss function, but for simplicity these were not considered during this project.

The training procedure includes early stopping based on the validation loss, as well as model checkpointing. The progress of the training was tracked using Weights & Biases [@wandb], which ultimately served as the tool to select the best model out of multiple training runs.
 
# Experiments {#sec:experiments}
The experiments conducted can be divided into the experiments with the statistical approaches and the experiments with the machine learning approaches. The statistical approaches were run on their respective subset of data as described in @sec:statistical-approaches. For the machine learning approaches, experiments were conducted for each model type (@sec:classical-machine-learning-models, @sec:neural-network-architecture) and each dataset (@sec:dataset-preparation), with a few exceptions. Additionally, ensemble experiments were conducted for each classical model type (@sec:ensemble-training-strategy).

+---------------------------------------------------------+----------------------+
| Experiment Type                                         | Models               |
+=========================================================+======================+
| Season Averages                                         | All                  |
| (@sec:season-averages)                                  |                      |
+---------------------------------------------------------+----------------------+
| Season Averages Ensembles                               | All, except NN & SVM |
| (@sec:season-averages, @sec:ensemble-training-strategy) |                      |
+---------------------------------------------------------+----------------------+
| Weighted Season Averages                                | All, except SVM      |
| (@sec:weighted-season-averages)                         |                      |
+---------------------------------------------------------+----------------------+
| Sliding Window Averages                                 | All, except SVM      |
| (@sec:sliding-window-averages)                          |                      |
+---------------------------------------------------------+----------------------+

: Overview of the conducted experiments {#tbl:experiment-overview}

@tbl:experiment-overview summarizes the experiments that were conducted for each dataset and model type. The SVM model was not trained on the Weighted Season Averages and Sliding Window Averages datasets because the SVM was not able to handle the size of these datasets.

All experiments were conducted using Weights & Biases [@wandb] sweeps with Bayesian optimization of the respective models hyperparameters (can be taken from the source code, @TODO: Source code reference), including dataset specific hyperparameters (see following sections). Like this an experiment for each model and experiment type as shown in @tbl:experiment-overview was run with a maximum of 100 runs. There was one exception for the latter for the neural networks, where one such experiment was run per loss function (@sec:loss-functions).

At the end of each experiment, the best model according to the lowest Brier score on the validation set of the respective dataset was selected for final evaluation on the test set on Kaggle.

## Dataset Hyperparameters
The dataloader of each dataset allows to specify a number $n$ of ranked features to use during training. The top $n$ features based on the features importance (@sec:feature-importance) would then be selected for training. Additionally the experiments with the Weighted Season Averages dataset (@sec:weighted-season-averages) allowed to specify weights for regular season and tournament games, as well as a discount factor for older games. These hyperparameters were also optimized during the sweeps.

## Data Split
The split into training and validation set depends on the experiment type. @tbl:data-split-overview summarizes how the data is splits for each experiment type.

+---------------------------+--------------------------+------------------+--------------------------+
| Experiment Type           | Training Data            | Validation Data  | Description              |
+===========================+==========================+==================+==========================+
| Season Averages           | Seasons 2003-2023        | Season 2024      |                          |
+---------------------------+--------------------------+------------------+--------------------------+
| Season Averages Ensembles | Each season, except one, | Each season once | @sec:ensemble-data-split |
|                           | for all seasons once     |                  |                          |
+---------------------------+--------------------------+------------------+--------------------------+
| Weighted Season Averages  | 75%                      | 25%              | Sampled from             |
|                           |                          |                  | entire dataset           |
+---------------------------+--------------------------+------------------+--------------------------+
| Sliding Window Averages   | 75%                      | 25%              | Sampled from             |
|                           |                          |                  | entire dataset           |
+---------------------------+--------------------------+------------------+--------------------------+

: Overview of the data splits for each experiment type {#tbl:data-split-overview}

### Ensemble Data Split {#sec:ensemble-data-split}

@TODO: @Dave - implement this

# Results {#sec:results}

All results described in this chapter follow the train/validation split described in @tbl:data-split-overview. The metric displayed 
is the Brier score, which is the evaluation metric used on Kaggle. Lower Brier scores are better. Additionally, the rank achieved on 
the Kaggle leaderboard for the respective test set is displayed. The Kaggle leaderboard contains a total of 1,727 submissions [@mmlm2025].

## Statistical Baselines {#sec:results-statistical-baselines}

| Model        | Train | Validation |    Test    |  Rank   |
|--------------|:-----:|:----------:|:----------:|:-------:|
| Point Ratio  |   -   |   0.2484   |   0.2548   |  1,423  |
| Random       |   -   |   0.2500   |   0.2500   |  1,272  |
| Head-to-Head |   -   |   0.2429   |   0.2350   |  1,210  |
| Win Ratio    |   -   |   0.2321   |   0.2284   |  1,190  |
| Seed Ratio   |   -   | **0.1961** | **0.1766** | **948** |

: Results Baseline Models {#tbl:results-baseline-models}

Among the statistical baseline models, Seed Ratio achieved the best performance with a validation Brier score of 0.1961 and test score of 0.1766, ranking 948th. The random baseline per definition has a validation and test score of 0.2500, ranking 1,272nd. Head-to-Head Ratio and Seed Ratio models performed slightly better than random with validation scores of 0.2429 and 0.2321 respectively, and test scores of 0.2350 and 0.2284, ranking 1,210th and 1,190th respectively. The Point Ratio model performed worst among all baselines with validation and test scores of 0.2484 and 0.2548 respectively, ranking 1,423rd.

## Season Averages {#sec:results-season-averages}

| Model                           |   Train    | Validation |    Test    |  Rank   |
|---------------------------------|:----------:|:----------:|:----------:|:-------:|
| Support Vector Machine          |   0.1804   |   0.1533   |   0.1369   |   768   |
| Logistic Regression             |   0.1671   |   0.1587   |   0.1191   |   288   |
| Random Forest                   | **0.1480** |   0.1547   | **0.1178** | **260** |
| XGBoost                         |   0.1675   |   0.1587   |   0.1234   |   408   |
| CatBoost                        |   0.1639   |   0.1561   |   0.1427   |   815   |
| Neural Network                  |   0.1916   | **0.1526** |   0.1251   |   521   |
| Neural Network - BCE            |   0.1902   |   0.1561   |   0.1189   |   283   |
| Neural Network - BCE + Entropy  |   0.2144   |   0.1551   |   0.1221   |   361   |

: Results Season Averages {#tbl:results-season-averages}

Random Forest achieved the best test performance with a score of 0.1178 and rank 260, while also having the lowest training score of 0.1480. Neural Network achieved the best validation score of 0.1526 but ranked lower at 521st with a test score of 0.1251. Neural Network - BCE and Logistic Regression obtained test scores of 0.1189 (rank 283) and 0.1191 (rank 288) respectively, both closely matching Random Forest's performance. The CatBoost and SVM models showed the highest test scores of 0.1427 (rank 815) and 0.1369 (rank 768) respectively. All models achieved very similar validation scores, ranging from 0.1526 to 0.1587, but more varying test scores ranging from 0.1178 to 0.1427. Compared to the ensemble variants in @tbl:results-season-averages-ensembles, individual models showed better test scores, with Random Forest at 0.1178 outperforming the best ensemble Logistic Regression model at 0.1189. Relative to the statistical baselines in @tbl:results-baseline-models, all machine learning models substantially improved performance, with the worst ML model (CatBoost at 0.1427) still outperforming the best baseline (Seed Ratio at 0.1766) by 0.0339.

| Model                  |   Train    | Validation |    Test    |  Rank   |
|------------------------|:----------:|:----------:|:----------:|:-------:|
| Support Vector Machine |   0.1627   |   0.1595   |   0.1242   |   453   |
| Logistic Regression    |   0.1673   |   0.1603   |   0.1218   |   351   |
| Random Forest          |   0.1532   |   0.1568   |   0.1184   |   270   |
| XGBoost                | **0.1119** |   0.1557   |   0.1241   |   448   |
| CatBoost               |   0.1588   |   0.1568   |   0.1192   |   290   |
| Neural Network         |   0.1933   | **0.1538** | **0.1181** | **266** |

: Results Season Averages - Default features {#tbl:results-season-averages-default-features}

With default features, Neural Network achieved the best test score of 0.1181 (rank 266) and best validation score of 0.1538, though it showed the highest training score of 0.1933. Random Forest obtained 0.1184 on test (rank 270), slightly higher than Neural Network. CatBoost achieved 0.1192 (rank 290), significantly improving from its ranked feature variant in @tbl:results-season-averages (0.1427, rank 815). Compared to the models in @tbl:results-season-averages where ranked features were used as hyperparameters, default features generally produced similar results, however with a significantly lower spread in test scores from 0.1181 to 0.1242. While the best default feature model (Neural Network at 0.1181) marginally exceeded the best optimized model (Random Forest at 0.1178) by 0.0003, the worst default feature model (SVM at 0.1242) substantially exceeded the worst optimized model (CatBoost at 0.1427).

## Season Averages Ensembles {#sec:results-season-averages-ensembles}

| Model                  |   Train    | Validation |    Test    |  Rank   |
|------------------------|:----------:|:----------:|:----------:|:-------:|
| Logistic Regression    |   0.1668   | **0.1676** | **0.1189** | **283** |
| Random Forest          | **0.1473** |   0.1684   |   0.1232   |   401   |
| XGBoost                |   0.1612   |   0.1681   |   0.1213   |   335   |
| CatBoost               |   0.1581   |   0.1682   |   0.1208   |   326   |

: Results Season Averages Ensembles {#tbl:results-season-averages-ensembles}

In the ensemble experiments, Logistic Regression achieved the best test score of 0.1189 (rank 283) and best validation score of 0.1676. CatBoost obtained 0.1208 on test (rank 326), followed by XGBoost at 0.1213 (rank 335). Random Forest showed the lowest training score of 0.1473 but achieved 0.1232 on test (rank 401). All ensemble models showed validation and test scores clustered between 0.1676 and 0.1684. and 0.1189 and 0.1232 respectively, therefore exhibiting less variance compared to individual models in @tbl:results-season-averages. Compared to individual models in @tbl:results-season-averages, ensemble XGBoost (0.1213) and CatBoost (0.1208) outperformed their individual counterparts (0.1234 and 0.1427 respectively), while the other models performed slightly worse.

| Model                  |   Train    | Validation |    Test    |  Rank   |
|------------------------|:----------:|:----------:|:----------:|:-------:|
| Logistic Regression    |   0.1658   | **0.1674** |   0.1186   |   278   |
| Random Forest          |   0.1531   |   0.1685   | **0.1171** | **245** |
| XGBoost                |   0.1568   |   0.1678   |   0.1181   |   266   |
| CatBoost               | **0.1405** |   0.1678   |   0.1182   |   267   |

: Results Season Averages Ensembles - Default features {#tbl:results-season-averages-ensembles-default-features}

With default features, Random Forest achieved the best test score of 0.1171 (rank 245), despite showing a higher validation score of 0.1685. XGBoost obtained 0.1181 on test (rank 266), followed closely by CatBoost at 0.1182 (rank 267). Logistic Regression achieved 0.1186 (rank 278) and the best validation score of 0.1674. Compared to the variants where ranked features were used as hyperparameters in @tbl:results-season-averages-ensembles, default features produced better results for every model type, with Random Forest improving from 0.1232 to 0.1171. This ensemble Random Forest (0.1171) matched the best score in @tbl:results-weighted-season-averages (CatBoost at 0.1171) and @tbl:results-sliding-window-averages (Neural Network - BCE at 0.1171), representing tied second-best performance across all approaches.

## Weighted Season Averages {#sec:results-weighted-season-averages}

| Model                             |   Train    | Validation |    Test    |  Rank   |
|-----------------------------------|:----------:|:----------:|:----------:|:-------:|
| Logistic Regression               |   0.1520   |   0.1520   |   0.1232   |   401   |
| Random Forest                     |   0.1508   |   0.1550   |   0.1204   |   316   |
| XGBoost                           |   0.1508   |   0.1549   |   0.1220   |   357   |
| CatBoost                          | **0.1500** |   0.1531   | **0.1171** | **245** |
| Neural Network                    |   0.1514   |   0.1526   |   0.1215   |   341   |
| Neural Network - Default Features |   0.1538   |   0.1530   |   0.1231   |   397   |
| Neural Network - BCE              |   0.1510   | **0.1513** |   0.1247   |   507   |
| Neural Network - BCE + Entropy    |   0.1886   |   0.1855   |   0.1265   |   570   |

: Results Weighted Season Averages {#tbl:results-weighted-season-averages}

CatBoost achieved the best test score of 0.1171 (rank 245) and lowest training score of 0.1500. Neural Network - BCE obtained the best validation score of 0.1513 but achieved 0.1247 on test (rank 507). Neural Network - BCE + Entropy showed the highest scores with 0.1886 on training, 0.1855 on validation, and 0.1265 on test (rank 570). CatBoost's test score of 0.1171 matched the best results from @tbl:results-season-averages-ensembles-default-features (Random Forest at 0.1171) and @tbl:results-sliding-window-averages (Neural Network - BCE at 0.1171), achieving tied best performance across these approaches. This represented a substantial improvement over individual Season Averages models in @tbl:results-season-averages where CatBoost achieved 0.1427. Except for the CatBoost model, all other models showed worse test performance than at least one of their counterparts in @sec:results-season-averages and @sec:results-season-averages-ensembles.

## Sliding Window Averages {#sec:results-sliding-window-averages}

| Model                             |   Train    | Validation |    Test    |  Rank   |
|-----------------------------------|:----------:|:----------:|:----------:|:-------:|
| Logistic Regression               |   0.1763   |   0.1773   |   0.1185   |   274   |
| Random Forest                     | **0.1650** |   0.1785   |   0.1228   |   386   |
| XGBoost                           |   0.1773   |   0.1779   | **0.1168** | **238** |
| CatBoost                          |   0.1655   |   0.1778   |   0.1195   |   297   | 
| Neural Network                    |   0.1779   |   0.1772   |   0.1187   |   278   |
| Neural Network - Default Features |   0.1781   |   0.1784   |   0.1213   |   335   |
| Neural Network - BCE              |   0.1777   | **0.1771** |   0.1171   |   245   |
| Neural Network - BCE + Entropy    |   0.2215   |   0.2221   |   0.1230   |   393   |

: Results Sliding Window Averages {#tbl:results-sliding-window-averages}

XGBoost achieved the best test score of 0.1168 (rank 238), representing the best performance across all experiments conducted. Neural Network - BCE obtained 0.1171 on test (rank 245), matching the top scores from @tbl:results-season-averages-ensembles-default-features and @tbl:results-weighted-season-averages. Random Forest showed the lowest training score of 0.1650 but achieved 0.1228 on test (rank 386). Neural Network - BCE + Entropy demonstrated the highest training and validation scores of 0.2215 and 0.2221 respectively, with a test score of 0.1230 (rank 393). All models except Neural Network - BCE + Entropy showed closely clustered validation scores between 0.1771 and 0.1785. Compared to Season Averages models in @tbl:results-season-averages, XGBoost improved from 0.1234 to 0.1168, demonstrating the value of sliding window features over season-averaged statistics. The XGBoost model's rank of 238 represents the highest achieved ranking among all experiments, beating the best models of both ensemble methods and weighted averaging strategies.

# Discussion {#sec:discussion}
@TODO

# Conclusion {#sec:conclusion}
@TODO

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

## Possible Future Work
One potential drawback of the approaches explored in this project is how the data was prepared and fed to the models. Each approach included an aggregation of the available data, which comes with a loss of information. For future work, we propose using the data without direct aggregation, instead considering the time series aspect by predicting matchup outcomes based on $n$ or all previous matchups. One approach would be to concatenate the statistics of the past $n$ games of each team in the matchup and use this as the input vector (for example, to a deep neural network) to predict the win probability of the current game. Another approach might be to feed the entire statistics of $n$ past games from each team to a recurrent neural network as samples at separate time steps and train a classification head on the latent representation of each team's series.

Another aspect that could be explored is different sources of data. One specific example would be to base predictions on the performance of individual players within each team, rather than on the team overall.

## Lessons Learned
@TODO

\newpage

# References {-}
