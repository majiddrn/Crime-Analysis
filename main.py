import pandas as pd
from sklearn.impute import SimpleImputer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth, association_rules
import numpy as np
import nltk

def dataSetCleaner():
    # Load the dataset
    data = pd.read_csv('crime.csv', encoding='iso-8859-1')

    # Data reduction
    imputer_c = SimpleImputer(strategy='constant')
    data.drop(index=data['OFFENSE_CODE'].isnull().index, axis=1)
    data.drop(index=(data['OFFENSE_DESCRIPTION'] == '').index, axis=1)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data['Lat'] = imputer.fit_transform(data[['Lat']])
    data['Long'] = imputer.fit_transform(data[['Long']])

    data['REPORTING_AREA'].fillna(data['REPORTING_AREA'].mode().iloc[0])
    data['SHOOTING'].fillna('N', inplace=True)
    data['OCCURRED_ON_DATE'].fillna(data['OCCURRED_ON_DATE'].mode().iloc[0])
    data['YEAR'].fillna(data['YEAR'].mode().iloc[0])
    data['MONTH'].fillna(data['MONTH'].mode().iloc[0])
    data['DAY_OF_WEEK'].fillna(data['DAY_OF_WEEK'].mode().iloc[0])
    data['HOUR'].fillna(data['HOUR'].mode().iloc[0])
    data['UCR_PART'].fillna(data['UCR_PART'].mode().iloc[0])
    data['STREET'].fillna(data['STREET'].mode().iloc[0])

    # Remove outliers using z-score
    numeric_columns = ['Lat', 'Long']  # Add other numeric columns as needed
    z_scores = np.abs((data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std())
    data = data[(z_scores < 3).all(axis=1)]  # Set the z-score threshold as needed

    # Convert numerical to categorical

    # Add code here to convert relevant numerical columns to categorical variables if needed

    # Stemming and stop word deletion
    nltk.download('wordnet')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    data['OFFENSE_DESCRIPTION'] = data['OFFENSE_DESCRIPTION'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))

    # Perform additional preprocessing steps as required

    # Save the preprocessed dataset to a new CSV file
    data.to_csv('preprocessed_crime.csv', index=False)
    return data


# data = dataSetCleaner()
data = pd.read_csv('preprocessed_crime.csv', encoding='iso-8859-1')
# # Task 1: Statistic Comparison between OFFENSE_CODE_GROUP and DAY_OF_WEEK
# # stat_comparison = data.groupby(['OFFENSE_CODE_GROUP', 'DAY_OF_WEEK']).size().reset_index(name='Occurrences')
# # print(stat_comparison)
# # Calculate the percentage of occurrence of each group in each day of the week
# result = data.groupby(['OFFENSE_CODE_GROUP', 'DAY_OF_WEEK']).size().unstack().apply(lambda x: x / x.sum() * 100, axis=1)

# # Print the result
# print(result)


# Task 2: finding patterns 
# Perform one-hot encoding on the relevant features if needed
# Split the dataset into two halves
half_size = len(data) // 4
data_half = data[:half_size]
# print(data_half['HOUR'].unique())

# Perform one-hot encoding on the relevant features if needed
# encoded_data = data_half[['DAY_OF_WEEK','Location' ,'OFFENSE_CODE_GROUP']]
# Clean the 'HOUR' column

# Drop rows with missing or invalid 'HOUR' values

encoded_data = pd.get_dummies(data_half[['DAY_OF_WEEK','Location','HOUR' ,'OFFENSE_CODE_GROUP']])
# 'HOUR',  'DAY_OF_WEEK','Location','OFFENSE_CODE_GROUP'
# Apply Apriori algorithm to extract frequent itemsets
frequent_itemsets = apriori(encoded_data, min_support=0.1, use_colnames=True)

# Generate association rules based on the frequent itemsets
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.1)

# Print the association rules
print(encoded_data)
# encoded_data = pd.get_dummies(data[['Location', 'HOUR', 'DAY_OF_WEEK', 'OFFENSE_CODE_GROUP']])
# # Perform FP-Growth algorithm to extract frequent itemsets
# frequent_itemsets = fpgrowth(encoded_data, min_support=0.1, use_colnames=True)

# # Generate association rules based on the frequent itemsets
# rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

# # Print the association rules
# print(rules)

# # Task 3: predic in test data 
# # Create the base classifiers
# clf1 = RandomForestClassifier()
# clf2 = LogisticRegression()

# # Create the stacking classifier
# stacking_clf = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=LogisticRegression())

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the stacking classifier on the training data
# stacking_clf.fit(X_train, y_train)

# # Predict the shooting incidents for the test data
# predictions = stacking_clf.predict(X_test)

# # Calculate the accuracy of the classifier
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy:", accuracy)