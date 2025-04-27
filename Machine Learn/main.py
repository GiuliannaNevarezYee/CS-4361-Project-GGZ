import numpy as np
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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

    return total_seconds / 60 if total_seconds > 0 else np.nan  # Ensure NaN if no time found


# Apply the function
df['Duration_Minutes'] = df['Duration'].apply(parse_duration)


categorical_cols = ['Type', 'Season', 'Source', 'Rating']
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

print(df.head())
print(df.describe())
print(df.info())
