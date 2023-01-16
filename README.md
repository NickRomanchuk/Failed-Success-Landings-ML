# Failed-Success-Landings-ML
 Machine Learning program to predict whether a jump will be landed successfully based on kinematic variables

Code was originally written for presentation at a conference and has since been adapted to a full manuscript publication. Data is not available due to restrictions from the research ethics department of the University of Ottawa. However, the code can be reviewed to determine the model fitting procdure and subsequent model evaluation. A brief outline of the fitting procedure is outlined below:
1. Partition data into training and testing sets
2. Variable reduction based on correlation and effect size
3. Normalization and one-hot encoding of variables
4. Leave-one-out cross-validation (where one participant is left out) to determine optimal number of variables to include in log regression model
5. Synethic minority oversampling techique within the cross-validation to fix class imbalances
6. Evaluate model oon left-out testing dataset using confusion matrix and kappa statistic