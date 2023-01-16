# Recommender System 2022 Challenge - Polimi

<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

This repo contains the code and the data used in the [Polimi's Recsys Challenge 2022](https://www.kaggle.com/competitions/recommender-system-2022-challenge-polimi)
<br>
The goal of the competition was to create a recommender system for TV shows, providing 10 recommendations for each user.

## Results

* Ranked 1st
* MAP@10 - private: 0.06213
* MAP@10 - public: 0.06259

## Goal
The application domain is TV shows recommendation. The datasets we provide contains both interactions of users with TV shows, as well as features related to the TV shows. The main goal of the competition is to discover which items (TV shows) a user will interact with.
Each TV show (for instance, "The Big Bang Theory") can be composed by several episodes (for instance, episode 5, season 3) but the data does not contain the specific episode, only the TV show. If a user has seen 5 episodes of a TV show, there will be 5 interactions with that TV show. The goal of the recommender system is not to recommend a specific episode, but to recommend a TV show the user has not yet interacted with in any way.

## Data description
The datasets includes around 1.8M interactions, 41k users, 27k items (TV shows) and two features: the TV shows length (number of episodes or movies) and 4 categories. For some user interactions the data also includes the impressions, representing which items were available on the screen when the user clicked on that TV shows.
The training-test split is done via random holdout, 85% training, 15% test.
The goal is to recommend a list of 10 potentially relevant items for each user. MAP@10 is used for evaluation. You can use any kind of recommender algorithm you wish e.g., collaborative-filtering, content-based, hybrid, etc. written in Python. Note that the impressions can be used to improve the recommendation quality (for example as additional features, a context, to estimate/reduce the recommender bias or as a negative interaction for the user) but are not used in any of the baselines.

## Evaluation
The evaluation metric for this competition is MAP@10.

## Recommender
Our best recommender is a hybrid composed of
* SLIM ElasticNet
* RP3Beta
* IALS

## XGBoost
We used XGBoost to rerank the recommendations of our hybrid model

Features:
* Item popularity
* User profile length
* Item genre
* Item length
* Predictions of other algorithms

## Team
* [Arturo Benedetti](https://github.com/Benedart)
* [Samuele Peri](https://github.com/john-galt-10)

## Credits
This repository is based on [Maurizio Dacrema's repository](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi)
