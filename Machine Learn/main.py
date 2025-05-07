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



# Read the dataset
df = pd.read_csv("anime-filtered.csv")

# Drop the unnecessary fields, the ID and Name of the anime
df = df.drop(['anime_id', 'Name', 'Producers', 'Licensors', 'Studios'], axis=1)

# Give a general overview of the dataset
# print(df.head())
# print(df.describe())
# print(df.info())

# Plot the score distribution
# df['Score'].hist(bins=50)
# plt.xlabel('Score')
# plt.ylabel('Count')
# plt.title('Distribution of Anime Scores')
# plt.show()


# Check if an english name is available if not then replace it with the japanese name
df['English name'] = df['English name'].where(
    ~df['English name'].str.strip().isin(['', 'UNKNOWN', np.nan]),
    df['Name']
)
# Replace the empty description with an empty string
df['Synopsis'] = df['Synopsis'].replace(["No description available for this anime.", np.nan], "")

# Use word embeddings on the title and synopsis
tagged_titles = [
    TaggedDocument(words=word_tokenize(title.lower()), tags=[f"TITLE_{i}"])
    for i, title in enumerate(df['English name'])
]

tagged_synopses = [
    TaggedDocument(words=word_tokenize(title.lower()), tags=[f"SYNOPSIS_{i}"])
    for i, title in enumerate(df['Synopsis'])
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

# Split genres by commas
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['Genres'].str.split(', '))
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# Add to main dataframe
df = pd.concat([df, genre_df], axis=1)
df = df.drop(['Genres'], axis=1)


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


def split_premiered(premiered_str):
    if premiered_str.lower() == 'unknown':
        return np.nan, np.nan

    parts = premiered_str.split(' ')
    season = parts[0]

    try:
        year = int(parts[1])
    except (IndexError, ValueError):  # IndexError if parts[1] is missing, ValueError if not an int
        year = np.nan

    return season, year


df[['Season', 'Year']] = df['Premiered'].apply(lambda x: pd.Series(split_premiered(x)))
df = df.drop(['Premiered'], axis=1)

df['Episodes'] = df['Episodes'].replace('Unknown', np.nan)
df['Episodes'] = pd.to_numeric(df['Episodes'], errors='coerce')


df['Duration'] = df['Duration'].replace('Unknown', np.nan)


# Function to convert duration string to total seconds
def parse_duration(duration_str):
    if pd.isnull(duration_str):
        return np.nan

    # Search for numbers before 'hr.', 'min.', and 'sec.'
    hours = re.search(r'(\d+)\s*hr', duration_str)
    minutes = re.search(r'(\d+)\s*min', duration_str)
    seconds = re.search(r'(\d+)\s*sec', duration_str)

    total_seconds = 0
    if hours:
        total_seconds += int(hours.group(1)) * 3600  # 1 hr = 3600 sec
    if minutes:
        total_seconds += int(minutes.group(1)) * 60  # 1 min = 60 sec
    if seconds:
        total_seconds += int(seconds.group(1))  # seconds

    return total_seconds / 60 if total_seconds > 0 else np.nan


# Apply the function
df['Duration_Minutes'] = df['Duration'].apply(parse_duration)
df = df.drop(['Duration'], axis=1)


categorical_cols = ['Type', 'Season', 'Source', 'Rating', 'Status']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)


df = df.fillna(df.median(numeric_only=True))

X = df.drop('Score', axis=1)
y = df['Score']

# Help standardize the features and normalize the score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
score_scaler = MinMaxScaler()
y_scaled = score_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

# LINEAR REGRESSION
from sklearn.ensemble import BaggingRegressor
# Attempt to use bagging
bagging = BaggingRegressor(estimator=SGDRegressor(max_iter=1000, tol=1e-3), n_estimators=20, random_state=42)
bagging.fit(X_scaled, y)
coefs = np.array([est.coef_ for est in bagging.estimators_])
avg_coefs = np.mean(np.abs(coefs), axis=0)

# Find the importance of each feature to see which ones are the best
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': avg_coefs})
top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(30)['feature'].tolist()
X_top = X[top_features]

X_top_scaled = scaler.fit_transform(X_top)

X_train, X_test, y_train, y_test = train_test_split(X_top_scaled, y, test_size=0.2, random_state=42)

# Gradient descent time baby
final_model = SGDRegressor(max_iter=1000, tol=1e-3)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"RMSE: {rmse:.2f}")

tolerance = 1.0
within_margin = np.abs(y_test - y_pred) <= tolerance
percentage_within_margin = within_margin.mean() * 100

print(f"Percentage of predictions within ±{tolerance}: {percentage_within_margin:.2f}%")
