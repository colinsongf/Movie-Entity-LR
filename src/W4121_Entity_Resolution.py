import numpy as np
import pandas as pd
import re
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.linear_model import LogisticRegression as LogiReg
from sklearn.model_selection import cross_val_score as crossVal

# team member:
def parse_dur(timestr):
    """
    Parse various time string into value in minutes
    timestr: containing hour, minute, and seconds
    deliminator does not matter
    abbreviations are okay, as long as it contains
    h for hour, m for minute, s for second

    :param timestr: containing hour, minute, and secondsd; eliminator does not matter abbreviations are okay,
                    as long as it contains h for hour, m for minute, s for second.
    :return: integer, rounded to the nearest minute
    """
    if isinstance(timestr, str):
        hrs = re.findall(r'[0-9]+[\s]*h', timestr)
        mins = re.findall(r'[0-9]+[\s]*m', timestr)
        secs = re.findall(r'[0-9]+[\s]*s', timestr)
        tarray = [0, 0, 0]  # Default 0
        tscale = [60, 1, 1/60]  # Scale to Minutes
        tinmins = 0  # Default 0
        for tind, tstr in enumerate([hrs, mins, secs]):
            if len(tstr) != 0:
                tarray[tind] = int(re.findall(r'[0-9]+', tstr[0])[0])
                tinmins += tarray[tind] * tscale[tind]
        return round(tinmins)
    else:
        return 0


def read_amazon(amazonfile):
    """Just pandas read csv with fixed arguments"""
    df_read = pd.read_csv(amazonfile, na_values=['NA'], engine='python', index_col=0,
                          usecols=["id", "time", "director", "star"])
    return df_read


def read_rottmt(rottmtfile):
    """Just pandas read csv with fixed arguments"""
    df_read = pd.read_csv(rottmtfile, na_values=['NA'], engine='python', index_col=0,
                          usecols=[0, 1, 2, 4, 5, 6, 7, 8, 9])
    return df_read


def liststr_remover(listin, str_format):
    """Maybe useful later. Not sure yet.
    """
    listout = []
    for element in listin:
        if element != str_format:
            listout.append(element)
    return listout


def entry_parser(ent_amz, ent_tmt):
    """ Further parse formatted entries into values

    Current mechanism: parse duration, director, and stars into scores, respectively;
    Duration: ratio in domain [0,1]; if duration seems strange, 0.5 is forced.
    Director: Fuzzy token set match
    Stars: Average score of element-wise list match
    e.g. For ['John Smith', 'Mary Jane', 'Ulysses Grant'] and
    ['Jane Smith', 'Jason Bourne', 'Ulysses S. Grant', 'Jon Doe', 'Maria June', 'Motoko Kusanagi']
    iterat through the shorter list, find the best token set match, and find the average score.

    :param ent_amz: [Duration (str), Director (str), Stars (str)]
    :param ent_tmt: [Duration (str), Director (str), Stars * 6 (multiple)]
    :return: parsed scores [X_dur, X_director, X_stars]

    """

    # Both have duration at their first column
    dur_amz = parse_dur(ent_amz[0])
    dur_tmt = parse_dur(ent_tmt[0])
    if dur_amz >= 10 and dur_tmt >= 10:
        # Duration ratio: 0 to 1, automatically normalized
        dur_ratio = min(dur_amz, dur_tmt) / max(dur_amz, dur_tmt)
    else:  # abnormal duration values
        dur_ratio = 0.5  # Force the duration ratio to be 0.5

    # column 2. Normalize to 1 (Fuzzywuzzy scores 0 to 100)
    director_ratio = fuzz.token_set_ratio(str(ent_amz[1]), str(ent_tmt[1])) / 100

    # For amazon.csv, stars are grouped in one string
    str_amz = str(ent_amz[2])

    # For rotten_tomatoes.csv, each star is in one column, with blank ones marked as nan
    lstrna_tmt = ent_tmt[2:]
    # Convert them to comparable formats
    lstr_amz = [xstr.strip() for xstr in str_amz.split(',')]
    lstr_amz = sorted(lstr_amz)
    # Remove the nan's
    # this syntax only works in Python3
    lstr_tmt = list(filter(None.__ne__, lstrna_tmt))
    lstr_tmt = sorted(np.str(lstr_tmt))

    # Find the shorter list
    if len(lstr_amz) <= len(lstr_tmt):
        lstr_short = lstr_amz
        lstr_long = lstr_tmt
    else:
        lstr_short = lstr_tmt
        lstr_long = lstr_amz
    ratio_total = 0
    n_entries = 0

    # Iterate through to find matching names
    for xstr in lstr_short:
        ratio_total += process.extractOne(xstr, lstr_long, scorer=fuzz.token_set_ratio)[1]
        n_entries += 1

    # Average & Normalize
    star_ratio = ratio_total / n_entries / 100.0

    return [dur_ratio, director_ratio, star_ratio]

# Main Process Begins
# ----------------------------------------------------------------------------------------------------------------------
# Change this to your directory housing all the csv files.
# os.chdir("C:/Users/cydru/Documents/W4121/W4121_EnRes")

# Read the training set first
# coz it's easy
df_train = pd.read_csv('train.csv', na_values=['NA'], engine='python', index_col=None)
trainlist_amz = list(df_train.iloc[:, 0])
trainlist_tmt = list(df_train.iloc[:, 1])
trainlist_ans = list(df_train.iloc[:, 2])

# read the messy data files
df_rottmt = read_rottmt('rotten_tomatoes.csv')
df_amazon = read_amazon('amazon.csv')
# Extract only relevant entries
# In this case, training entries
sl_amazon = df_amazon.loc[trainlist_amz, :]
sl_rottmt = df_rottmt.loc[trainlist_tmt, :]
l_train = len(trainlist_ans)

# Constructing the trainng input
xmat = list([])
# # Separate the abnormally formated entries from "normal" ones
# abnormals = list([])
for itentry in range(l_train):
    # Extract entry
    entry_amz = list(sl_amazon.iloc[itentry, :])
    entry_tmt = list(sl_rottmt.iloc[itentry, :])
    # Calculate and Record
    xmat.append(entry_parser(entry_amz, entry_tmt))

# Convert to ndarray with numpy for inputting into sklearn
xmat = np.array(xmat)
yvec = np.array(trainlist_ans)

# Cross-validation to find best regulatory term
Cvec = np.power([2]*30, range(30))
Scores = np.arange(30, dtype=np.float64)
for ind, Cval in enumerate(Cvec):
    cross_scores = crossVal(LogiReg(C=Cval), xmat, yvec, scoring='accuracy', cv=20)
    Scores[ind] = cross_scores.mean()
Cbest = Cvec[np.argmax(Scores, axis=0)]
model = LogiReg(C=Cbest)
model = model.fit(xmat, yvec)
print(model.coef_)  # For our information


# Testing Procedure: test.csv
# ----------------------------------------------------------------------------------------------------------------------
df_test = pd.read_csv('test.csv', na_values=['NA'], engine='python', index_col=None)
testlist_amz = list(df_test.iloc[:, 0])
testlist_tmt = list(df_test.iloc[:, 1])
st_amazon = df_amazon.loc[testlist_amz, :]
st_rottmt = df_rottmt.loc[testlist_tmt, :]
l_test = len(testlist_amz)
xmat_test = list([])
for itentry in range(l_test):
    # Extract entry
    entry_amz = list(st_amazon.iloc[itentry, :])
    entry_tmt = list(st_rottmt.iloc[itentry, :])
    # Calculate and Record
    xmat_test.append(entry_parser(entry_amz, entry_tmt))
xmat_test = np.array(xmat_test)
yvec_predict = model.predict(xmat_test)
# ----------------------------------------------------------------------------------------------------------------------
# Write to csv
goldframe = pd.DataFrame(data=yvec_predict, index=None, columns=["gold"])
goldframe.to_csv('gold.csv', sep=',', index=False, index_label=False)

# Testing Procedure: holdout.csv
# ----------------------------------------------------------------------------------------------------------------------
df_test = pd.read_csv('holdout.csv', na_values=['NA'], engine='python', index_col=None)
testlist_amz = list(df_test.iloc[:, 0])
testlist_tmt = list(df_test.iloc[:, 1])
st_amazon = df_amazon.loc[testlist_amz, :]
st_rottmt = df_rottmt.loc[testlist_tmt, :]
l_test = len(testlist_amz)
xmat_test = list([])
for itentry in range(l_test):
    # Extract entry
    entry_amz = list(st_amazon.iloc[itentry, :])
    entry_tmt = list(st_rottmt.iloc[itentry, :])
    # Calculate and Record
    xmat_test.append(entry_parser(entry_amz, entry_tmt))
xmat_test = np.array(xmat_test)
yvec_predict = model.predict(xmat_test)
# ----------------------------------------------------------------------------------------------------------------------
# Write to csv
goldframe = pd.DataFrame(data=yvec_predict, index=None, columns=["gold"])
goldframe.to_csv('gold2.csv', sep=',', index=False, index_label=False)

print('The end is the beginning is the end.')