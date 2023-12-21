# Wine Quality Prediction

This Python code utilizes the scikit-learn library to implement a Random Forest Classifier for a wine quality prediction task. Here's a breakdown of the code:

1. **Import Libraries:**
   - `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`: Common data manipulation and visualization libraries.
   - `train_test_split` from `sklearn.model_selection`: Used to split the dataset into training and testing sets.
   - `RandomForestClassifier` from `sklearn.ensemble`: The machine learning model used for classification.
   - `accuracy_score` from `sklearn.metrics`: Used to evaluate the accuracy of the model predictions.

2. **Load Dataset:**
   - Reads a CSV file containing wine data into a Pandas DataFrame (`wine_dataset`).

3. **Exploratory Data Analysis (EDA):**
   - Displays the first few rows of the dataset (`wine_dataset.head()`).
   - Checks for null values in the dataset (`wine_dataset.isnull().sum()`).
   - Provides summary statistics of the dataset (`wine_dataset.describe()`).
   - Creates a categorical plot to visualize the distribution of wine qualities (`sns.catplot`).

4. **Correlation Analysis:**
   - Calculates the correlation matrix of the dataset (`correlation = wine_dataset.corr()`).
   - Generates a heatmap of the correlation matrix using Seaborn.

5. **Data Preprocessing:**
   - Splits the dataset into features (X) and target variable (Y).
   - Converts the 'quality' column into binary labels (1 if quality is greater than or equal to 7, 0 otherwise).
   - Splits the data into training and testing sets using `train_test_split`.

6. **Model Training:**
   - Initializes a Random Forest Classifier.
   - Fits the classifier to the training data.

7. **Training Evaluation:**
   - Makes predictions on the training set (`X_train`) and calculates training accuracy.

8. **Testing Evaluation:**
   - Makes predictions on the test set (`X_test`) and calculates testing accuracy.

9. **Individual Prediction:**
   - Takes a single data point (`new_input`) from the test set.
   - Converts the input to a NumPy array.
   - Retrieves the corresponding actual label (`y_input`).
   - Predicts the label for the input and calculates the accuracy for this individual prediction.

Overall, the code demonstrates the process of loading a dataset, exploring its characteristics, training a Random Forest Classifier, and evaluating its performance on both training and testing datasets. Additionally, it performs an individual prediction and evaluates its accuracy.
