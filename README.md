# bias-in-credit-models

Machine learning is being deployed to do large-scale decision making, which can strongly impact the life of individuals. By not considering and analysing such scenarios, we may end up building models that fail to treat societies equally and even infringe anti-discrimination laws.

There are several algorithmic interventions to identify unfair treatment based on what is considered to be fair. This project focuses on showing how these interventions can be applied in a case study using a classification-based credit model. 

# Case Study Outline
I made use of a public loan book from Bondora, a P2P lending platform based in Estonia. I looked into two different protected groups: gender and age.

Bondora provides lending to less credit-worthy customers, with the presence of much higher default rates than seen in traditional banks. This means that the interests collected are significantly higher. On average, the loan amount for this dataset was around €2,100 with a payment duration of 38 months and interest rate of 26.30%. 

For traditional banks, the cost of a false positive (misclassifying a defaulting loan) is many times greater than reward of a true positive (correctly classifying a non-defaulting loan). Given the higher interest rates collected by Bondora compared to banks, I will assume for illustration purposes that the reward to cost ratio is much smaller at 1 to 2. This will be used to find the best thresholds to maximise profits while meeting all requirements for each algorithmic intervention.

I then developed a classification model that predicts whether a loan is likely to be paid back or not using the technique Gradient Boosted Decision Trees. With the results of the model predictions, I then analysed the following scenarios:

- _Maximise profit_ uses different classification thresholds for each group and only aims at maximising profit.
Fairness through unawareness uses the same classification threshold for all groups while maximising profit.
- _Demographic parity_ applies different classification thresholds for each group, while keeping the same fraction of positives in each group.
- _Equal opportunity_ uses different classification thresholds for each group, while keeping the same true positive rate in each group.
- _Equalised odds_ applies different classification thresholds for each group, while keeping the same true positive rate and false positive rate in each group.

# Project Structure

### I) Data Cleaning
_pre_process.py_ restructures the data by setting it in the right format and renaming as needed for visualisation.
The file _fill_missing_values.py_ make restructure the data and fill missing values that will be later used in the modeling phase.

### II) Data Exploration
Both notebook take the processed and restructured data and plots the distributions, correlations and missing data. 

### III) Credit Model
Does a grid search to find the best model using the technique Gradient Boosted Decision Trees. After finding the best model, it saves the predictions and the original data as CSV.

### IV) Model Analysis and Unfairness Detection
- _model_performance.ipynb_: Reviews the performance of the model using ROC curves and AUC for 'Gender' and 'Age Group.
- _unfairness_measures.py_: Finds the best thresholds for each protected class by maximising profits whie meeting each algorithmic intervention requirements. This then saves all results as CSV.
- _model_fairness_interventions.ipynb_: Reviews the results for from _unfairness_measures_

# More Information
For more information on each algorithmic intervention and the intepretation of the case study results, go to:
https://medium.com/@ValeriaCortezVD/preventing-discriminatory-outcomes-in-credit-models
