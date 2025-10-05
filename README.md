# SHiNES (Search for High Inclination Non-transiting Exoplanets)

This repository gives a baseline pipeline for detecting rare highly mutually inclined exoplanet systems with Transit Timing Variations (TTVs) and Transit Duration Variations (TDVs).

In order to tackle this largely undersampled category of exoplanet systems, we utilize a Deep Learning alghorithm to identify the probability of a highly mutually inclined non-transiting companion in a system with a known transiting exoplanet. Mutual inclination refers to the angle between the orbital planes of two or more planets in a system.  These discoveres are crucial to gaining deeper insight into planetary disk formation models, and finding them raises important questions about the stability of such systems over time and the processes that lead to the observed configurations. 

Our model consists of a Bi-LSTM encoder, Transformer, and Multi-Layer Perceptron to analyze Transit Timing Variations (TTVs) and Transit Duration Variations (TDVs) from Kepler telescope data. The model code itself is located in "model_generation." Our data generation code is locate din "data_generation." Our data generation code is used from the DeepTTV paper which outlined a data generation precedure akin to what we were planning on doing. Data generation is in a C++ architecture.

[Learn More](https://shines-hunter.vercel.app)
