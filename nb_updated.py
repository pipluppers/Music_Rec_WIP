import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB

from subprocess import check_output
import warnings
warnings.filterwarnings('ignore')


# Load data and Merge the training set with the songs and members files
# Use only 1% of the training set. File is too large
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
songs = pd.read_csv('../input/songs.csv')
members = pd.read_csv('../input/members.csv')

train = train.sample(frac=0.01)
train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
del songs
del members

train.info()


# Use to get an idea of which features have missig values and how many
train.isnull().sum()

# -----------------------------------------------------------------------------------------------------------

# Fill in the missing values for artist names, language of the song, and song length (in seconds)
freq_artist = train.artist_name.dropna().mode()[0]
freq_lang = train.language.dropna().mode()[0]
train['song_length'] = train['song_length'].replace(train['song_length'].values, \
                        train['song_length'].values / 1000)

train['artist_name'] = train['artist_name'].fillna(freq_artist)
train['language'] = train['language'].fillna(freq_lang)
train['song_length'] = train['song_length'].fillna(train.song_length.mean())

# -----------------------------------------------------------------------------------------------------------

# Create new features for the year, month, and day of the registration and expiration dates
train.registration_init_time = pd.to_datetime(train.registration_init_time, format= \
                '%Y%m%d', errors='ignore')
train['registration_year'] = train['registration_init_time'].dt.year.astype(int)
train['registration_month'] = train['registration_init_time'].dt.month.astype(int)
train['registration_day'] = train['registration_init_time'].dt.day.astype(int)

train.expiration_date = pd.to_datetime(train.expiration_date, format= \
                '%Y%m%d', errors='ignore')
train['expiration_year'] = train['expiration_date'].dt.year.astype(int)
train['expiration_month'] = train.expiration_date.dt.month.astype(int)
train['expiration_day'] = train.expiration_date.dt.day.astype(int)

train.head()
#train[['registration_year','target']].groupby(by='registration_year',as_index=
#    False).mean().sort_values(by='registration_year')
#train[['expiration_year','target']].groupby(by='expiration_year', as_index= \
#    False).mean().sort_values(by='expiration_year')

# -----------------------------------------------------------------------------------------------------------

