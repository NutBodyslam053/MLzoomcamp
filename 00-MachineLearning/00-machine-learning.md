# Machine Learning (ML)
Machine learning (ML) is the process of discovering patterns in data. It involves two types of data: 
- **Features:** The properties of the objects being analyzed.
- **Target:** The property that the ML model will attempt to predict for new, unseen objects. 

The ML model uses the patterns learned from the features to make predictions when presented with new feature values.

## Types of ML
### Supervised Learning
The algorithm is trained on a `labeled dataset` and learns to `make predictions` based on that data.
- **Regression:**  The output is a `number`.
    - ***Linear Regression:*** *Predicts a continuous value (e.g., car/house/stock price prediction).*
- **Classification:** The output is a `category`.
    - ***Decision Trees:*** *Classifies data by splitting it based on a set of if-then rules (e.g., image classification, spam email detection, medical diagnosis).*
    - ***Neural Networks:*** *Models complex relationships between inputs and outputs (e.g., fraud/object detection).*
### Unsupervised Learning
The algorithm is trained on an `unlabeled dataset` and learns to `identify patterns` or structure in the data.
- **Clustering:**
    - ***K-Means:*** *clusters data into a specified number of groups (e.g., customer segmentation).*
- **Anomaly Detection:**
    - ***Autoencoders:*** *learns to represent data in a lower dimensional space (e.g., credit card fraud, network intrusion).*
- **Dimensionality Reduction:**
    - ***Principal Component Analysis (PCA):*** *reduces the dimensionality of a dataset (e.g., image compression).*
### Reinforcement Learning
The algorithm learns through `trial and error` by receiving feedback in the form of `rewards or punishments` based on its actions.
- **Game Playing:**
    - ***Deep Reinforcement Learning:*** *applies neural networks to reinforcement learning problems (e.g., train an AI to play complex video games such as Go, chess, or Dota 2).*
- **Robotics:**
    - ***Actor-Critic:*** *combines policy-based and value-based methods to improve performance (e.g., train a robot to navigate a physical environment and perform tasks).*
- **Autonomous Driving:**
    - ***Q-Learning:*** *learns to make decisions in a Markov Decision Process (MDP) environment (e.g., train a self-driving car)*
---
# CRISP-DM
CRISP-DM, or the Cross-Industry Standard Process for Data Mining, is an open standard model that outlines the standard approaches used by data mining experts. It is considered the most widely used analytics model in the industry.

1. **Business understanding:** It is important to ask if machine learning is necessary and ensure that the project goal is clearly defined and measurable.
2. **Data understanding:** Analyze available data sources to determine if additional data is needed.
3. **Data preparation:** Convert the data into a tabular format to be used in ML. Prepare the data by cleaning, transforming, and splitting it into training and testing sets.
4. **Modeling:** Select several model families that are appropriate for the problem based on domain knowledge and experience. Train and validate the models on the training set using appropriate evaluation metrics. Optimize the hyperparameters of the best performing models using cross-validation or grid search.
5. **Evaluation:** Evaluate the models on the testing set and compare their performance using appropriate evaluation metrics. Select the best performing model based on the evaluation metrics and use it to make predictions on new data.
6. **Deployment:** Deploy the model in production, made it available to all users and monitor its performance over time to ensure that it continues to meet the desired performance standards.
---