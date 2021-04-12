Data Source: https://www.kaggle.com/snap/amazon-fine-food-reviews

Performed Exploratory Data Analysis, Data Cleaning, Data Visualization and Text Featurization(BOW, Tfidf,Word2Vec). Build Several ML Models like KNN, Naive Bayes, Logistic Regression, SVM, Random Forest, GBDT, LSTM(RNNs) etc.

Objective:

Given a Text Review, Determine Whether the Review is Positive (Rating of 4 or 5) or Negative (Rating of 1 or 2).

About Dataset

The Amazon Fine Food Reviews dataset consists of Reviews of Fine Foods from Amazon.

Number of Reviews: 568,454 Number of users: 256,059 Number of Products: 74,258 Timespan: Oct 1999 - Oct 2012 Number of Attributes/Columns in data: 10

Attribute Information:

Id
ProductId - unique identifier for the product
UserId - unqiue identifier for the user
ProfileName
HelpfulnessNumerator - number of users who found the review helpful
HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
Score - rating between 1 and 5
Time - timestamp for the review
Summary - brief summary of the review
Text - text of the review
1. Exploratory Data Analysis, Natural Language Processing, Text Preprocessing and Visualization using TSNE
Defined Problem Statement.
Performed Exploratory Data Analysis(EDA) on Amazon Fine Food Reviews Dataset Plotted Word Clouds, Distribution plots, Histograms, etc.
Performed Data Cleaning & Data Preprocessing(Removed html tags, Punctuations, Stopwords and Stemmed the words using Porter Stemmer).
Plotted TSNE with Different Perplexity values for Different Featurization like BOW(uni-gram), Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec.
2. KNN
Applied K-Nearest Neighbour on Different Featurization like BOW, tfidf, Avg-Word2Vec and Tf-idf-Word2Vec. Applying 10-fold CV by using Brute Force Algorithm to Find Optimal 'K'.
Calculated MissClassification Error for each K value.
Evaluated the Test data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also Plotted Confusion matrix.
Failure Cases of K-NN:-
If my Query Point is Far Away from Neighbour Points I cannot Decide it's Particular Class.
If my Positive Points and Negative Points are jumbled so tightly there is no Useful Information in these cases my Machine learning Algorithms Fails.
KNN Limitations:-
Knn takes large Space Complexity of order(nd) and time complexity of order(nd).

Conclusions:
KNN is a very Slow Algorithm it takes very long Time to Train.
In K-nn We Should not take K-value even Because Classification is done by Majority vote.
Best Accuracy is Achieved by Avg Word2Vec Featurization Which is of 89.48%.
3. Naive Bayes
Applied Naive Bayes using Bernoulli NB and Multinomial NB on Different Featurization BOW, Tfidf.
Find Right Alpha(α) using Cross Validation.
Get Feature Importance for Positive class and Negative Class.
Evaluated the Test data on Various Performance metrics like Accuracy, F1-score, Precision, Recall,etc. also Plotted Confusion matrix.
Conclusions:
Naive Bayes is much Faster Algorithm than KNN.
Best F1 score is Acheived by Tf-idf Featurization which is 0.89.
4. Logistic Regression
Applied Logistic Regression on Different Featurization BOW, Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec.
Find Lambda(λ) By Grid Search & Randomized Search.
Evaluated the Test data on various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also Plotted Confusion matrix.
Showed How Sparsity Increases as we Increase lambda or decrease C when L1 Regularizer is used for each Featurization.
Did Pertubation Test to check whether the Features are multi-collinear or not.
Assumptions of Logistic Regression
logistic Regression does not require a Linear relationship between the Dependent and Independent variables.
The Error Terms (residuals) do not need to be Normally Distributed.
Homoscedasticity is not Required.
Finally, the Dependent Variable in Logistic Regression is not Measured on an interval or ratio scale.
Conclusions:
Sparsity Increases as we decrease C (increase lambda) When we use L1 Regularizer for Regularization.
Logistic Regression with Tfidf Featurization Performs best with F1_score of 0.89 and Accuracy of 93.385.
Logistic Regression is Faster Algorithm.
5. SVM
Applied SVM with RBF(Radial Basis Function) kernel on Different Featurization BOW, Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec.
Find Right C and Gamma (ɣ) Using Grid Search & Randomized Search Cross Validation.
Applied SGDClassifier on Featurization.
Evaluated Test Data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also plotted Confusion matrix.
Conclusions:
SGD with Bow By Random Search gives Better Results.
SGDClasiifier Takes Very Less Time to Train.
6. Decision Trees
Applied Decision Trees on Different Featurization BOW, Tfidf, Avg-Word2Vec and Tf-idf-Word2Vec To find the optimal depth using Cross Validation.
By doing Grid Search We finded max_depth.
Evaluated the Test data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also plotted Confusion matrix. 4.Plotted wordclouds of Feature Importance of both Positive class and Negative Class.
Conclusions:
Tf-idfw2vec Featurization(max_depth=11) gave the Best Results with Accuracy of 77.24.
7. Ensembles Models (Random Forest &Grident Boost Decision Tree)
1.Apply GBDT and Random Forest Algorithm To Find Right Baselearners using Cross Validation and Get Feature Importance for Positive class and Negative class. 2. Performing Grid Search for getting the Best Max_depth, Learning rate. 3. Evaluated the Test data on Various Performance Metrics like Accuracy, F1-score, Precision, Recall,etc. also plotted Confusion matrix. 4. Plotted Word Cloud of Feature Importance Received for RF and GBDT Classifier.

Conclusions:
Avgw2vec Featurization in Random Forest (BASE-LEARNERS=120) with Grid Search gave the Best Results with F1-score of 89.0311.
Tfidfw2v Featurization in GBDT (Learning Rate=0.05, DEPTH=3) gave the Best Results with F1-score of 88.9755.
8. LSTM(RNNs)
Getting Vocabulary of all the words and Getting Frequency of each word, Indexing Each word Converting data into Imdb dataset format Running the lstm model and Report the Accuracy.
Applied Different Architectures of LSTM on Amazon Fine Food Reviews Dataset.
Recurrent Neural Networks(RNN) with One LSTM layer.
Recurrent Neural Networks(RNN) with Two LSTM layer.
Recurrent Neural Networks(RNN) with Three LSTM layer.
Conclusions:
RNN with 1LSTM layer has got high Accuracy 92.76
