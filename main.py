import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np

# Load the dataset
data = pd.read_csv('crime.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data['Lat'] = imputer.fit_transform(data[['Lat']])
data['Long'] = imputer.fit_transform(data[['Long']])

# Remove outliers using z-score
numeric_columns = ['Lat', 'Long']  # Add other numeric columns as needed
z_scores = np.abs((data[numeric_columns] - data[numeric_columns].mean()) / data[numeric_columns].std())
data = data[(z_scores < 3).all(axis=1)]  # Set the z-score threshold as needed

# Convert numerical to categorical
# Add code here to convert relevant numerical columns to categorical variables if needed

# Stemming and stop word deletion
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

data['OFFENSE_DESCRIPTION'] = data['OFFENSE_DESCRIPTION'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))

# Perform additional preprocessing steps as required

# Save the preprocessed dataset to a new CSV file
data.to_csv('preprocessed_crime.csv', index=False)