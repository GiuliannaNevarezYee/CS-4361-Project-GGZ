#!/usr/bin/env python
# coding: utf-8

# ## This is the project made into a jupyter notebook for ease of running certain parts without having to run the whole program again

# In[1]:


get_ipython().system('pip install numpy==1.24.3')
get_ipython().system('pip install gensim')


# In[2]:


import numpy as np
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


# In[3]:


# Read the dataset
df = pd.read_csv("anime-dataset.csv")

# Drop the unnecessary fields, like the ID of the anime
df = df.drop(['anime_id', 'Producers', 'Licensors', 'Studios'], axis=1)


# ### A general overview of our initial dataset

# In[4]:


# Give a general overview of the dataset
print(df.head())
print(df.describe())
print(df.info())


#  ### Plot the distribution of scores

# In[5]:


# Plot the score distribution
df['Score'].hist(bins=50)
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Distribution of Anime Scores')
plt.show()


# ### Process the title and synopsis using word embeddings and Doc2Vec.

# In[6]:


def score_to_tag(score):
    if 9 <= score <= 10:
        return "amazing"
    elif 7 <= score < 9:
        return "good"
    elif 5 <= score < 7:
        return "okay"
    elif 3 <= score < 5:
        return "bad"
    else:
        return "terrible"

# Check if an english name is available if not then replace it with the japanese name
df['English name'] = df['English name'].where(
    ~df['English name'].str.strip().isin(['', 'UNKNOWN', np.nan]),
    df['Name']
)
# Replace the empty description with an empty string
df['Synopsis'] = df['Synopsis'].replace(["No description available for this anime.", np.nan], "")

# Use word embeddings on the title and synopsis
tagged_titles = [
    TaggedDocument(words=word_tokenize(title.lower()), tags=[f"TITLE_{i}", score_to_tag(score)])
    for i, (title, score) in enumerate(zip(df['English name'], df['Score']))
]

tagged_synopses = [
    TaggedDocument(words=word_tokenize(synopsis.lower()), tags=[f"SYNOPSIS_{i}", score_to_tag(score)])
    for i, (synopsis, score) in enumerate(zip(df['Synopsis'], df['Score']))
]

# Train the models for the title and synopsis using Doc2Vec
title_model = Doc2Vec(vector_size=50, window=2, min_count=1, epochs=40, dm=1, workers=4)
title_model.build_vocab(tagged_titles)
title_model.train(tagged_titles, total_examples=title_model.corpus_count, epochs=title_model.epochs)

synopsis_model = Doc2Vec(vector_size=100, window=5, min_count=2, epochs=40, dm=1, workers=4)
synopsis_model.build_vocab(tagged_synopses)
synopsis_model.train(tagged_synopses, total_examples=synopsis_model.corpus_count, epochs=synopsis_model.epochs)

# Retrieve the vectors for the title and synopsis
title_vectors = np.vstack([
    title_model.dv[f"TITLE_{i}"] for i in range(len(df))
])

synopsis_vectors = np.vstack([
    synopsis_model.dv[f"SYNOPSIS_{i}"] for i in range(len(df))
])

# Turn the vectors into dataframes then add them to the dataframe and then drop the old columns
title_df = pd.DataFrame(title_vectors, columns=[f"title_vec_{i}" for i in range(title_vectors.shape[1])])
synopsis_df = pd.DataFrame(synopsis_vectors, columns=[f"synopsis_vec_{i}" for i in range(synopsis_vectors.shape[1])])

df_vectors = pd.concat([df.reset_index(drop=True), title_df, synopsis_df], axis=1)
df = df_vectors.drop(['Name', 'English name', 'Synopsis'], axis=1)


# ### Process the genres by using one-hot encoding

# In[7]:


# Split genres by commas
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['Genres'].str.split(', '))
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# Add to dataframe
df = pd.concat([df, genre_df], axis=1)
df = df.drop(['Genres'], axis=1)


# ### Process aired by turning it into how many days it aired for

# In[8]:


# Trying to turn the dates into a datetime
def parse_date(date_str):
    date_str = date_str.strip()

    # Checks if it's just a year
    if re.fullmatch(r'\d{4}', date_str):
        return datetime(int(date_str), 1, 1)

    # The other possible formats for the dates which are something like dd-mon-yy or mon dd, yyyy
    date_formats = ['%d-%b-%y', '%b %d, %Y']

    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


# Calculates how many days an anime aired for
def calculate_days_aired(aired_str):
    # Checks to if there's any info on it first
    if pd.isnull(aired_str) or aired_str.strip().lower() == 'unknown':
        return np.nan

    # Takes into account the 'to' in the formating
    if 'to' in aired_str:
        # If it has a 'to' it'll split it into a start date and an end date
        start_str, end_str = aired_str.split('to')
        start_date = parse_date(start_str)
        end_str = end_str.strip()

        # If the end date is ? or nothing that means it's still airing so set the date to today otherwise turn it into a date
        if end_str == '?' or end_str == '':
            end_date = datetime.now()
        else:
            end_date = parse_date(end_str)
    # If there is no 'to' then that means it only aired once so like a movie so just make the end date the start date
    else:
        start_date = parse_date(aired_str)
        end_date = start_date

    # Now we can calculate how many days it aired for
    if start_date and end_date:
        days = (end_date - start_date).days + 1
        return days if days > 0 else 1
    # Something happened so just put the value as NaN
    else:
        return np.nan

# Changes the dates aired into how many days it aired for to make it into a value that's easier to work with
df['Aired_Days'] = df['Aired'].apply(calculate_days_aired)
df = df.drop(['Aired'], axis=1)


# ### Process premiered by splitting the season and year and one-hot encoding the season

# In[9]:


# Function that splits the premiered feature into a season and a year that it premiered
def split_premiered(premiered_str):
    # Checks if the value is unknown to replace it with nan
    if premiered_str.lower() == 'unknown':
        return np.nan, np.nan

    # Splits the premiered values
    parts = premiered_str.split(' ')
    season = parts[0]

    # Tries to get the year if not then something went wrong and it can turn it into nan
    try:
        year = int(parts[1])
    except (IndexError, ValueError):
        year = np.nan

    return season, year

# Apply the function to the premiered feature column and drop it
df[['Season', 'Year']] = df['Premiered'].apply(lambda x: pd.Series(split_premiered(x)))
df = df.drop(['Premiered'], axis=1)


# ### Process the duration by turning it into minutes

# In[10]:


# Function that converts the duration feature into how many minutes
def parse_duration(duration_str):
    # Checks if it's null and turns it into nan
    if pd.isnull(duration_str):
        return np.nan

    # Search for the numbers that go before the hr, min, and sec
    hours = re.search(r'(\d+)\s*hr', duration_str)
    minutes = re.search(r'(\d+)\s*min', duration_str)
    seconds = re.search(r'(\d+)\s*sec', duration_str)

    total_seconds = 0

    # Converts the times into seconds
    if hours:
        total_seconds += int(hours.group(1)) * 3600  # 1 hr = 3600 sec
    if minutes:
        total_seconds += int(minutes.group(1)) * 60  # 1 min = 60 sec
    if seconds:
        total_seconds += int(seconds.group(1))       # 1 sec = 1 sec (shocking)

    # Return the total minutes if possible else just return nan
    return total_seconds / 60 if total_seconds > 0 else np.nan


# Apply the function to the duration feature and then drop it
df['Duration_Minutes'] = df['Duration'].apply(parse_duration)
df = df.drop(['Duration'], axis=1)


# ### Process episodes by turning it into a float

# In[11]:


# Convert the episodes feature into a float to keep things consistent
df['Episodes'] = df['Episodes'].replace('Unknown', np.nan)
df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')


# ### Process the, type, season, source, rating, and status by one-hot encoding them

# In[12]:


# One-hot encode the remaining features that can't be turned into just flat numbers
categorical_cols = ['Type', 'Season', 'Source', 'Rating', 'Status']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)


# ### Prepare everything to start training the models

# In[13]:


# Fill any nans with the median
df = df.fillna(df.median(numeric_only=True))

# Make our X by dropping the score and the y by keeping the score
X = df.drop('Score', axis=1)
y = df['Score']

# Help standardize the features by using standard scaler and normalize the score by using min max scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
score_scaler = MinMaxScaler()
y_scaled = score_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# Create our training and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_scaled, test_size=0.2, random_state=42)


# ### Logistic Regression implementation and feature importance graph

# In[32]:


# Train the linear regression model then calculate the RMSE
final_model = SGDRegressor(max_iter=1000, tol=1e-3)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

rmse_lgr = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Linear Regression RMSE: {rmse_lgr:.4f}")


# In[76]:


# Get the importance of the coefficients by getting their absolute values
coefs = np.abs(final_model.coef_)

# Make a dataframe to map the importance of each feature
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': coefs
}).sort_values(by='importance', ascending=False)

top_n = 20
top_features_df = importance_df.head(top_n)

# Plot the top 20 features
ax = top_features_df.plot(kind='barh', x='feature', y='importance', title='Linear Regression Top 20 Features by Importance', figsize=(12, 6), legend=False)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
for i, (index, row) in enumerate(top_features_df.iterrows()):
    ax.text(row['importance'] + 0.0005, i, f"{row['importance']:.4f}", va='center', fontsize=9)
plt.tight_layout()
plt.show()


# ### Random Forest Regressor along with feature importance graph

# In[17]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"Random Forest RMSE: {rmse_rf:.4f}")


# In[66]:


# Extract feature importance from the trained Random Forest model 
rf_feature_importance = rf_model.feature_importances_

# Set a threshold for feature importance (e.g., keep features with importance > 0.001)
importance_threshold = 0.001

# Get the important features (boolean mask for importance > threshold)
important_rf_features = rf_feature_importance > importance_threshold

# Get the names of the important features
X_important_rf = X.columns[important_rf_features]

# Get the importance values of the important features
importance_values = rf_feature_importance[important_rf_features]

# Sort the features by importance
sorted_indices = importance_values.argsort()
X_important_rf_sorted = X_important_rf[sorted_indices]
importance_values_sorted = importance_values[sorted_indices]

#  Plot the feature importance for the important features 
plt.figure(figsize=(12, 6))
bars = plt.barh(X_important_rf_sorted, importance_values_sorted)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Random Forest Regression Features by Importance (Filtered by Threshold)')
for bar, value in zip(bars, importance_values_sorted):
    width = bar.get_width()
    plt.text(width + 0.0005, bar.get_y() + bar.get_height()/2,
             f"{value:.4f}", va='center', fontsize=9)
plt.show()


# ### XGBRegressor along with feature importance graph

# In[30]:


from xgboost import XGBRegressor

# Train XGBoost regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate the model performance
y_pred = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"XGBoost RMSE: {rmse_xgb:.4f}")


# In[59]:


from xgboost import plot_importance

# Plot the 20 most important features by their frequency
plot_xgb = plot_importance(xgb_model, max_num_features=20, importance_type='weight', height=0.5)
plt.title("XGBoost Regression Top 20 Features by Importance (by frequency)")
plt.xlabel('Importance')
plt.ylabel('Feature')
plot_xgb.grid(False)
plt.show()


# ### Plot the RMSE of the 3 models to compare them

# In[65]:


# RMSE values
models = ['Linear Regression', 'Random Forest Regression', 'XGBoost Regression']
rmse_values = [rmse_lgr, rmse_rf, rmse_xgb]

# Plot
plt.figure(figsize=(8, 6))
bars = plt.bar(models, rmse_values)
plt.ylabel('RMSE')
plt.xlabel('Model')
plt.title('RMSE Comparison of Models')
for bar, rmse in zip(bars, rmse_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.001, f"{rmse:.4f}",
             ha='center', va='bottom', fontsize=10)
plt.show()

