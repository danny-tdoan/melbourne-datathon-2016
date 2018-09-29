# Overview
The classification model used for the Kaggle competition in Melbourne Datathon 2016. We ranked 6th out of 59 participants.

Link to the competition: <https://www.kaggle.com/c/melbourne-datathon-2016>

Taken from the competition:

> The objective is to predict if a job is in the 'Hotel and Tourism' category.
>
> In the 'jobs' table there is a column 'HAT' which stands for 'Hotel and Tourism'. The values in this column are 1 or 0 representing 'Yes' and 'No' meaning it is or is not in the Hotel and Tourism category. This binary flag is a look up from the column 'Subclasses'.
>
> Some of the rows have a value of -1 for HAT. These are the rows you need to predict.
>
> The prediction can be a  1/0 or a continuous number representing a probability of a job being in the HAT category.

We use two approaches:

1. An ensemble model that averages the results of XGBoost and RandomForest. The final model is blended from two
independent models: The first use the title, salary, location, job type as features. The second model extracts keywords
from the job description to use as features.

2. A model using Stacking. Submodels use RandomForestClassifier, ExtraTreesClassifier, XGBoost and LogisticRegression.