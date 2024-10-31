# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def trim_dataset(df: pd.DataFrame):
    df_usa = df[df['Country']=='USA'] #Taking only USA shark attacks    
    df_florida = df_usa[df_usa["State"]=='Florida'] #Only Florida shark attacks
    df_florida.reset_index(drop=True, inplace=True) #Reset index
    df_florida.columns = [c.lower().strip() for c in df_florida.columns.values.tolist()] #Column name to lowercase
    idx_columns_to_drop = [1, 3, 4, 7, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    columns_to_drop=[]
    for index in idx_columns_to_drop:
        columns_to_drop.append(df_florida.columns[index])
    
    df_florida = df_florida.drop(columns=columns_to_drop)#Removing useless columns
    
    return df_florida

def standardize_string(column: pd.Series):
  clean_column = column.str.lower().str.strip().apply(str) #convert a column to lowercase strings without starting/ending whitespace
  return clean_column

def standardize_activity(activity: str):
  """
  Place every activity in a common category based on the words in this activity.
  """
  if activity=='nan':
    return 'unknown'

  activities = ['surfing', 'swimming', 'wading', 'fishing', 'standing', 'spearfishing', 'boarding', 'snorkeling', 'diving', 'walking', 'bathing', 'floating']
  standardized_activity = "other"
  for activ in activities:
    if activ in activity:
      standardized_activity = activ
      break

  return standardized_activity

def standardize_type(attack_type: str):
  if attack_type != 'unprovoked' and attack_type != 'provoked':
    return 'unknown'
  return attack_type

def standardize_location(location: list):
  """
  Return a [location, county] list based on the words in this location.
  """

  counties = ['volusia county', 'brevard county', 'martin county', 'duval county', 'palm beach county', 'monroe county', 'indian river county', 'lee county']
  locations = ['new smyrna beach', 'daytona beach', 'ponce inlet', 'melbourne beach', 'cocoa beach', 'jacksonville beach', 'florida keys', 'juno beach', 'ormond beach', 'palm beach', 'riviera beach']
  loc = "other"
  county = "other"

  if len(location)==1:

    if location[0] == 'nan':

      return ['unknown', 'unknown']

    else:
      loc = 'other'
      county = 'other'
      loca = location[0]

      for l in locations:
        if l in loca:
          loc = l
      for c in counties:
        if c in loca:
          county = c


  elif len(location) ==2:

    loc, county = location[0], location[1]

    for l in locations:
      if l in loc:
        loc = l

    for c in counties:
      if c in loc:
        county = c

  elif len(location) > 2:

    loc, county = location[0], location[-1]

    for l in locations:
      if l in loc:
        loc = l

    for c in counties:
      if c in loc:
        county = c

  return [loc, county]

def get_county_from_location(df: pd.DataFrame):
    test_df = pd.DataFrame([standardize_location(x.split(',')) for x in df['location']], columns=['precise_location', 'county'])
    
    for i, (loc, county) in enumerate(test_df.values):
      if loc not in ('other', 'unknown') and county in ('other', 'unknown'):
    
        if loc == 'new smyrna beach' or loc == 'ormond beach' or loc == 'daytona beach': test_df.loc[i, "county"] = 'volusia county'
        if loc == 'florida keys': test_df.loc[i, "county"] = 'monroe county'
        if loc == 'riviera beach' or loc == 'palm beach' or loc == 'juno beach': test_df.loc[i, "county"] = 'palm beach county'
    
    return pd.concat([df, test_df], axis=1, join='inner')

def clear_sex(sex):
    if pd.isna(sex):
        return "F"

    sex = sex.strip().upper()

    if sex == "F":
        return "F"
    elif sex == "N" or sex == "LLI":
        return "F"
    else:
        return "M"

def categorize_age(age):
    if age > 60: return 'Senior'
    if age > 18: return 'Adult'
    if age > 12: return 'Teens'
    else: return 'Child'

def clean_date_prefix(date_value):
  prefixes = ["Reported", "Eearly", "Before", "Fall"]
  for prefix in prefixes:
    if date_value.startswith(prefix):
      return date_value[len(prefix):].strip()
    return date_value

def standardize_time_format(time_str):
    time_str = str(time_str).strip().lower()

    # Define regex patterns for matching time formats
    patterns = [
        r'(\d{1,2})h(\d{2})',  # matches '12h30', '11h15'
        r'(\d{1,2})h',          # matches '9h', '14h'
        r'(\d{1,2}):(\d{2})',   # matches '11:30', '14:05'
        r'(\d{4})hr',           # matches '1600hr', '1100hr'
        r'(\d{4})',             # matches '1600', '1300' (4 digits)
    ]

    for pattern in patterns:
        match = re.fullmatch(pattern, time_str)
        if match:
            if len(match.groups()) == 2:  # e.g. '12h30'
                hours, minutes = match.groups()
                return f"{int(hours):02}h{int(minutes):02}"
            elif len(match.groups()) == 1:  # e.g. '9h' or '1600hr'
                if 'h' in time_str and 'hr' not in time_str:
                    hours = match.group(1)
                    return f"{int(hours):02}h00"
                else:
                    # Handling 4-digit format like '1600'
                    hours = match.group(1)[:2]
                    minutes = match.group(1)[2:]
                    return f"{int(hours):02}h{int(minutes):02}"

    return 'unknown'

def time_category(time):
  time = str(time)
  if time[0:2].isdigit():
    hour = int(time[0:2])
    if hour > 20 or hour < 7:
      return 'night'
    elif hour >= 12:
      return 'afternoon'
    elif hour >= 7:
      return 'morning'

  return 'unknown'

def clean_and_standardize_time(df, column_name):
    series = df[column_name]
    series = series[~series.isin(["?", "", np.nan])]

    cleaned_series = series.apply(standardize_time_format)


    df[column_name] = cleaned_series.reset_index(drop=True)
    df[column_name] = df[column_name].apply(time_category)

    return df

small_species = ['4-5ft blacktip shark', "bull shark, 4'", '"a small shark"', "4' shark", "blacktip shark, 5' to 6'", "spinner shark, 4'?",
                 "2' to 3' shark", "3' to 5' shark", 'small nurse shark', "4' to 5' shark", 'nurse shark, juvenile', "bull shark, 4' to 6'",
                 'juvenile bull shark?', 'juvenile blacktip shark', "4' to 5' blacktip shark", "2' shark", "3' to 4' shark", "lemon shark, 3'",
                 'spinner shark, 4 to 5 feet', "5' shark", "5' to 6' shark", "bull shark, 4' to 5'", "nurse shark, 4'", "blacktip shark, 4'", "nurse shark, 2'",
                 "spinner shark, 5'", "blacktip shark, 4' to 5'", 'juvenile shark', "3' shark", "blacktip or spinner shark, 4'", "bull shark, 5'",
                 '3- to 4-foot shark', "possibly a bull shark, 3'", "lemon shark, 4'", 'a small shark', "3.5' to 4' shark", "lemon shark, 4' to 5'",
                 "bull shark, 4.5'", "4' to 6' shark", 'small bull shark', "1' to 2' shark", 'a small spinner shark', 'nurse shark, juvenile ',
                 "5' to 8' shark", "+3' shark", "2' to 3' juvenile shark", "1.5' to 2' shark", "2.5' shark", "blacktip shark, 5'", "nurse shark, 2' to 3'",
                 "18\" to 24\" shark", "3' to 3.5' shark", "possibly a 1' to 3' blacktip or spinner shark", "bull shark, 4' to bull shark",
                 '"small shark"', '18" to 36" shark', 'small hammerhead shark', "nurse shark, 3'", "1.2 m [4'] shark", "1.2 m [4'] bull shark",
                 "1.2 m to 1.5 m [4' to 5'] shark", "1.5 m to 1.8 m [5' to 6'] shark", 'unidentified species', "106 cm [3.5'] shark",
                 '“small brown shark”', "0.9 m to 1.2 m [3' to 4'] shark", 'a “small” shark', "60 cm [2'] captive shark", 'species unidentified',
                 "3.5' to 4.5' shark", "1' to 2' hammerhead or bonnethead shark", "0.9 m to 1.2 m [3' to 4'] shark; tooth fragment recovered from hand",
                 '1 m shark', "3' blacktip shark", "3' shark", "0.9 m [3'] shark", 'nurse shark, 1m ', "spinner shark, 3' to 4'", "2' to 3.5' shark",
                 "1.5 m [5'] shark", "3' shark, possibly a blacktip or spinner shark", "blacktip shark, 5' to 6'", "0.9 m [3'] shark",
                 "possibly a 1.5 m [5'] blacktip or spinner shark", "spinner shark, 1.2 m to 1.5 m [4' to 5']", "1.2 m to 1.8 m [4' to 6'] shark",
                 "blacktip shark, 2'", "60 cm to 90 cm [2' to 3'] shark", "a 2' shark was seen in the area by witnesses", "nurse shark, 1.5 m [5']",
                 "nurse shark, 1.2 m [4']", "0.9 m [3'] shark, probably a blacktip or spinner shark", 'a "small shark"', "4.5' to 5' shark",
                 "lemon shark, 1.5 m [5'], identified by the surfer", "nurse shark, 0.9 m [3']", "5' spinner shark", 'small blacktip shark',
                 '"juvenile shark"', "1.2 m [4'] shark (spinner shark?)", 'possibly a sand shark', '"a young shark"',
                 "60 cm to 90 cm [2' to 3'] blacktip or spinner shark", "nurse shark, 0.9 m [3']", "1.2 m to 1.5 m [4' to 5'] shark",
                 "1.5 to 1.8 m [5' to 6'] shark", "3 m to 3.7 m [10' to 12'] bull shark", 'thought to involve a small sand shark', "1.2 m [4'] spinner shark",
                 "nurse shark, 3', 20-lb", "4' spinner shark", "1.5 m [5'] blacktip shark", "1.2 m [4'] blacktip shark", 'hammerhead shark?+o2356',
                 "4.5' shark", "3.5' hammerhead shark", "1.2 m to 1.5 m [4' to 5'] bull, sandbar or dusky shark", 'a small hammerhead shark',
                 'nurse shark, 106 cm, 28-lb, male', 'nurse shark, 1 m', "nurse shark, 2.5'", "1.2 m [4'] hammerhead shark",
                 "nurse shark, 0.94 m to 1.2 m [3' to 4']", "1.7 m [5.5'] shark", "nurse shark, 1.1 m [3.5']",
                 'nurse shark, 60 cm [24"], identified by Dr. L.P.L. Schultz on photograph', "tiger shark, 1.5 m to 1.8 m [5' to 6']",
                 "hammerhead shark, 5'", "0.7 m [2.5'] shark", "nurse shark, 1.5 m [5'] identified by Dr. E. Clark on description of shark",
                 "1.2 m [4'] shark", 'lemon shark, 1164 mm, immature male, identified by V.G. Springer', "1.2 m [4'], possibly larger shark"]

medium_species = ['bull shark 6ft', '6ft shark', "bull shark, 6'", 'mako shark', 'juvenile nurse shark', "lemon shark  6'", "6' shark",
                  'blacktip or spinner shark', "7' to 8' shark", "blacktip shark, 6'", "blacktip shark, 6' to 7'", "lemon shark, 8'", "6.5' shark",
                  'spinner shark', "8' shark", "bull shark, 5' to 7'", "bull shark, 7'", "spinner shark, 7'", 'spinner shark or blacktip shark',
                  "bull shark, 8'", 'blacktip or spinner shark?', 'nurse shark?', "bull shark, 6' to 8'", "possibly a 5' to 6' bull shark",
                  "6' shark, possibly a blacktip or spinner shark", "sandbar shark, 8'", "lemon shark, 6'", "spinner shark, 6'", "bull shark, 8'",
                  "nurse shark, 6'", 'possibly a spinner shark', "possibly a 6' bull shark", "lemon shark, 6' female", "6' to 8' shark",
                  '1.8 m blacktip "reef" shark', "1.8 m [6'] bull shark", "1.8 m [6'] blacktip shark", "1.8 m [6'] caribbean reef shark",
                  "8' great hammerhead shark", "2.1 m to 2.4 m [7' to 8'] shark", "bull shark, 2.3 m [7.5']", 'possibly a juvenile blacktip or spinner shark',
                  "2.1 to 2.4 m [7' to 8'] shark", "bull shark, 2.4 m [8']", "1.8 m [6'] shark", 'unidentified', "bull shark, 2.1 m [7']",
                  "blacktip shark, 1.8 m [6']", "1.8 m [6'] shark, possibly a blacktip", 'possibly a blacktip shark',
                  "1.8 m to 2.1 m [6' to 7'] hammerhead shark", "2.1 m [7'] shark", ">1.8 m [6'] shark",
                  "1.8 m to 2.4 m [6' to 8'] shark, tooth fragments recovered", 'lemon shark', "6' to 7' blacktip shark",
                  "2.4 m [8'] shark", 'lemon shark, 30-lb', "2.1 m [7'], 140-lb reef shark", "1.8 m [6'] shark",
                  "1.8 m to 2.4 m [6' to 8'] shark", '150-lb shark', "1.8 m to 2.1 m [6' to 7'] spinner or blacktip shark",
                  "lemon shark, 1.8 m to 2.4 m [6' to 8'], tooth fragment recovered", '1.8 m silky shark',
                  "nurse shark, 2.1 m [7'] identified by Dr. E. Clark on color & tooth impressions",
                  'tiger shark?', "1.8 m [6'] blacktip shark or spinner shark", "nurse shark, 2.1 m [7']",
                  'possibly a small hammerhead shark', "6' to 8' bull shark", "1.8 m [6'] shark, species identity questionable",
                  "1.8 m to 2.4 m [6' to 8'] hammerhead shark", 'bull shark, 8', "2.1 m [7'] lemon shark or bull shark",
                  "tiger shark, 6'", "7.5' shark", "silky shark, 1.9 m [6.5']",
                  "said to involve a 2.4 m [8'] hammerhead shark",
                  'lemon shark, 1.8 m [6\'] male, n. breviostris, identified by w.a. stark ii, later the same day a 6\'8" pregnant female lemon shark bit the bow of the boat']

large_species = ['bu.ll', "bull shark, 9'", 'nurse shark', 'blacktip shark?', "9' shark", 'bull shark', 'white shark', 'blacktip shark',
                 'tiger shark', 'blue shark, 8 to 9 feet', 'possibly a bull shark', "bull shark, 12'", "nurse shark, 10'", "9.5' shark?",
                 "possibly a 10' bull shark", "bull shark, 10'", "14' shark",
                 "thought to involve a 2.7 m [9'], 400-lb bull shark", "blacktip shark, 2.4 m to 3 m [8' to 10']", 'hammerhead shark',
                 "lemon shark, 2.7 m [9']", "3 m [10'] bull shark", "2.7 m [9'] bull or lemon shark",
                 'said to involve a tiger shark or a hammerhead shark', '"sandshark"', 'c. maculpinnis or c. limbatus',
                 "12' to 14' shark", "10' shark", "lemon shark, 9'", "2.7 m [9'] silky shark", 'spinner or blacktip sharks',
                 "white shark, 10' to 12'", '500-lb shark', 'hammerhead shark, 500-lb', "tiger shark, 3 m [10']",
                 "mako shark, 1.8 m to 2.1 m [6' to 7'] with hook & wire leader caught in mouth",
                 'a hammerhead shark, then 8 to 10 other sharks were said to be involved', "3.7 m [12'] shark",
                 "tiger shark, 3.7 m [12']", "tiger shark 4.3 m [14']", "2.7 m [9'] bull shark, identified by Capt. W. Gray",
                 "4 m [13'] shark", 'said to involve a large mako shark', 'a sand shark', 'possibly c. leucas',
                 'tiger shark, 4.5 to 5.5 m [14\'9" to 18\'], 2000-lb', '650-lb shark', 'possibly a bull shark or tiger shark',
                 "hammerhead shark, 2.4 m [8'], according to lifeguard Sam Barrows", '9-foot shark',
                 "3.7 m [12'], 1200-lb shark. shark caught & its jaw exhibited at the Carnegie Museum",
                 "reported to involve a 3.7 m [12'] shark, possibly a white shark", "13' shark", "3 m [10'] shark",
                 "12' shark"]

not_specified = ['not specified', 'injury most likely caused by barracuda, not a shark',
                 'no shark involvement - it was a publicity stunt', 'possibly a blacktip or spinner shark',
                 'no shark involvement', 'shark involvement probable, but not confirmed',
                 'shark involvement prior to death was not confirmed', 'possibly a spinner shark',
                 'invalid', 'unknown, but it was reported that a shark tooth was recovered from the wound',
                 ' ', '15 cm to 20 cm [6" to 8"] bite diameter just below left knee', 'questionable incident',
                 'questionable incident - shark bite may have precipitated drowning', 'unidentified',
                 'shark involvement not confirmed; officials considered barracuda',
                 'shark involvement prior to death unconfirmed', 'questionable']

# Define the classification function
def classify_size(species):
    species = species.lower().strip()  # Normalize input
    if species in small_species:
        return 'small'
    elif species in medium_species:
        return 'medium'
    elif species in large_species:
        return 'large'
    elif species in not_specified:
        return 'unknown'
    else:
        return 'unknown'
    