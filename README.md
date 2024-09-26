# Research for how to safely handle imbalanced datasets

# Project Description

### Main objective to explore in this project

Do different types of data need different techniques to treat imbalanced data,
or we can just choose the best available / popular technique and use it for any data?

### Context
Some people ignore importance of understanding their data and just try to use the most powerful technique available to crunch the data,
but it can produce low quality statistical metrics and high computational cost.

Why am I researching it?
I need to know the answer - do I need to evaluate and choose just once the best available techniques
to treat imbalanced data and then use them for any data with any distribution,
or I need to evaluate which technique works the best for my data every time when I need to work with data?


# Key Highlights
1) Datasets are used from Kaggle

2) Classifiers used in this project:
   - LogisticRegression
   - RandomForestClassifier
   - XGBClassifier

3) Samplings used in this project:
   - Nosampling
   - RandomUnderSampler
   - RandomOverSampler
   - SMOTE
   - tomek
   - enn
   - SMOTEENN
   - SMOTETomek

4) Evaluation:
   RepeatedStratifiedKFold

5) Metrics
   - Precision
   - Average Precision
   - Recall
   - F1 score
   - ROC AUC

   
# How to Run the Project

1)  Clone this repository: `git clone https://github.com/alexey-tyurin/imbalanced_datasets.git`
2)  Download dataset Credit Card Fraud Detection
    (instructions in section "Data sources" in `imbalanced_datasets_notebook.ipynb`)
3)  Open and run the Jupyter Notebook: `imbalanced_datasets_notebook.ipynb`


# Repository Structure

1) datasets
 
   Folder that includes diabetes_prediction_dataset.csv dataset

2) imbalanced_datasets_notebook.ipynb
   
   Jupyter Notebook


# Results and Discussion

Imbalanced datasets are hard to treat not because of accuracy metric as many people are saying,
but because of more fundamental problem - how optimization algorithms work in general and specifically how they work with some sampler of data.
Several science papers concluded that the methods SMOTE+ENN in combination with a logistic regression classifier give the best performance.
In this project several samplings including SMOTE+ENN with combination with several models were explored.

### Best results for mean recall achieved in this project

1) For diabetes_prediction_dataset.csv:

   Logistic Regression + SMOTEENN:				0.900706

   Random Forest + SMOTEENN:					0.882588

   XGBoost Classifier + RandomUnderSampler: 	0.913922

2) For creditcard.csv:

   Logistic Regression + SMOTEENN:				0.873492

   Random Forest + RandomUnderSampler:			0.854550

   XGBoost Classifier + RandomUnderSampler:	0.890529

These results show that for both diabetes_prediction_dataset.csv and creditcard.csv the best result for mean recall
was achieved not with SMOTE+ENN in combination with a logistic regression classifier,
but with XGBoost Classifier + RandomUnderSampler.


### Best results for mean ROC AUC achieved in this project

1) For diabetes_prediction_dataset.csv:
 
   Logistic Regression + RandomOverSampler:	0.962168

   Random Forest + RandomUnderSampler:			0.964199

   XGBoost Classifier + enn: 					0.978108

2) For creditcard.csv:

   Logistic Regression + RandomOverSampler:	0.966445

   Random Forest + SMOTETomek:					0.968731

   XGBoost Classifier + enn:					0.970635

These results show that for both diabetes_prediction_dataset.csv and creditcard.csv the best result for ROC AUC
was achieved not with SMOTE+ENN in combination with a logistic regression classifier,
but with XGBoost Classifier + enn.


### Conclusion
We need to understand our data and treat them according to their nature.
Best result might depend on distribution of the data.
Instead of trying to use just one best available / popular technique,
we need to try several combinations of them and see which combination can yield best results.
Range of these combinations might depend on budget, computing resources and time available for it.
Also, important consideration: evaluation metric to use as it can depend on what is important to measure for specific project at hands
and different combinations can be winners for different metrics.
We should not rely on just one method, but need to look at all toolbox available for handling imbalanced data like
Data-Level Techniques (Resampling) and Algorithm-Level Techniques (Cost-Sensitive Learning, Ensemble Methods, Anomaly Detection),
choose several methods that can help for our dataset at hands, experiment with them and find combination that gives best results for our chosen metrics.


### Following up questions
1) Can we make rules for which kind of techniques to treat imbalanced data are the best for which type of data distribution?
2) How can we evaluate computational performance for techniques to treat imbalanced data?


# Acknowledgments

### Datasets
1) Credit Card Fraud Detection
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download&select=creditcard.csv

2) Diabetes prediction dataset
   https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset


### References
[1] Ajinkya More. Survey of resampling techniques for improving classification performance in unbalanced datasets, 22 Aug 2016.
https://arxiv.org/pdf/1608.06048

[2] Yang, F., Wang, K., Sun, L. et al.
A hybrid sampling algorithm combining synthetic minority over-sampling technique and edited nearest neighbor for missed abortion diagnosis.
BMC Med Inform Decis Mak 22, 344 (2022).
https://doi.org/10.1186/s12911-022-02075-2

[3] Kumar Abhishek, Dr. Mounir Abdelaziz
Machine Learning for Imbalanced Data: Tackle imbalanced datasets using machine learning and deep learning techniques, November 30, 2023.
https://www.amazon.com/Machine-Learning-Imbalanced-Data-imbalanced-ebook/dp/B0C4B5H7GB/


# Contact Information

For any questions or feedback, please contact Alexey Tyurin at altyurin3@gmail.com.



