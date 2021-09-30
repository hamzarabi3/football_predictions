from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd 
from joblib import dump, load
import os 
import numpy as np

from warnings import filterwarnings

filterwarnings('ignore')

#-----0----------------functions for adding previous match results and goals------------------------------------/
def get_previous_match_result(match_index,team_name,date,df):
  try:
    filter1=df.query('away_team==@team_name and date<@date')
    filter1=filter1.sort_values(by='date')
    filter1=filter1.iloc[-match_index]
    if filter1['FTR']=='A':
      filter1['FTR']='W'
    elif filter1['FTR']=='H':
      filter1['FTR']='L'
  except :
    filter1=df.query('away_team=="me playing in the kitchen"')

  try:
    filter2=df.query('home_team==@team_name and date<@date')
    filter2=filter2.sort_values(by='date')
    filter2=filter2.iloc[-match_index]
    if filter2['FTR']=='A':
      filter2['FTR']='L'
    elif filter2['FTR']=='H':
      filter2['FTR']='W'
  except:
    filter2=df.query('away_team=="me playing in the kitchen"')

              
  if len(filter1)==0 and  len(filter2)>0:
    return filter2['FTR']
  if len(filter1)>0 and len(filter2)==0:
    return filter1['FTR']
  if len(filter1)==0 and len(filter2)==0:
    return None

  if filter1.date <filter2.date:
    return filter2['FTR']
  return filter1['FTR']

def get_previous_match_goals(match_index,team_name,date,df):
  try:
    filter1=df.query('away_team==@team_name and date<@date')
    filter1=filter1.sort_values(by='date')
    filter1=filter1.iloc[-match_index]
    date1=filter1.date
    filter1=filter1['away_goals']
  except:
    filter1=None
  try:
    filter2=df.query('home_team==@team_name and date<@date')
    filter2=filter2.sort_values(by='date')
    filter2=filter2.iloc[-match_index]
    date2=filter2.date

    filter2=filter2['home_goals']
  except:
    filter2=None
              
  if filter1 is None and filter2 is not None:
    return filter2
  if filter1 is not None and filter2 is None:
    return filter1
  if filter1 is None and filter2 is None:
    return None
  if date1 <date2:
    return filter2
  return filter1


#-----1----------reading data file and CONFIGURATIONS--------------------------------------------------------------------------------------------------------/

# lookback=5
# training=True
def generate_input_files(lookback=5,frac=1):
  """
  Takes all_leagues.csv file and outputs train_features.csv, and train_targets.csv
  """
  data_folder='data'
  models_folder='models'

  data_file=os.path.join(data_folder,'all_leagues.csv')
  

  league=pd.read_csv(data_file).sample(frac=frac)
  league.date=pd.to_datetime(league.date)
  league.reset_index(inplace=True)
  league.drop_duplicates(inplace=True)
  

  targets_file=os.path.join(data_folder,"targets.csv")
  features_file=os.path.join(data_folder,'features.csv')

  #-----2-------Adding previous matches results(win, lose, draw), lookback argument determins how many past matches to consider as input to the model----------------/
  print('Adding history of results')
  for x in range(1,lookback+1):
    print(f'{x}',end='\t')
    #add home team previous match results
    new_columnh=f'HMFTR_{x}'
    print('.',end='\t')
    league[new_columnh]=league.apply(lambda raw:get_previous_match_result(x,raw.home_team,raw.date,league),axis=1)
    league[new_columnh].fillna('NA',inplace=True)

    new_columna=f'AMFTR_{x}'
    print('.')
    league[new_columna]=league.apply(lambda raw:get_previous_match_result(x,raw.away_team,raw.date,league),axis=1)
    league[new_columna].fillna('NA',inplace=True)
      

  #------3-------------Adding previous matches goals---------------/

  print('Adding history of goals')

  for x in range(1,lookback+1):
    print(f'{x}',end='\t')
    #add home team previous match results
    new_columnh=f'HMFTG_{x}'
    print('.',end='\t')
    league[new_columnh]=league.apply(lambda raw:get_previous_match_goals(x,raw.home_team,raw.date,league),axis=1)
    league[new_columnh].fillna(0,inplace=True)

    new_columna=f'AMFTG_{x}'
    print('.')
    league[new_columna]=league.apply(lambda raw:get_previous_match_goals(x,raw.away_team,raw.date,league),axis=1)
    league[new_columna].fillna(0,inplace=True)

  #------4----------Create our prediction objectives from goals data----------------/
  print('Creating target columns')

  league['total_goals_more_than_3']=league.apply(lambda raw: (raw['away_goals']+raw['home_goals'])>=3,axis=1).astype('int')
  league['btts']=league.apply(lambda raw: raw['away_goals']>0 and raw['home_goals']>0,axis=1).astype('int')
  league['total_goals_more_than_2']=league.apply(lambda raw: (raw['away_goals']+raw['home_goals'])>=2,axis=1).astype('int')
  league['away_team_wins']=league.apply(lambda raw: raw['FTR']=='A',axis=1).astype('int')
  league['home_team_wins']=league.apply(lambda raw: raw['FTR']=='H',axis=1).astype('int')
  league['draw']=league.apply(lambda raw: raw['FTR']=='D',axis=1).astype('int')

  #------5---------Create additional calendar features from the datetime column------/

  print('Adding calendar features')

  league['match_week']=league.date.dt.week
  league['match_week_day']=league.date.dt.dayofweek
  try:
    league.drop(['FTR','home_goals','away_goals'],inplace=True,axis=1)
  except KeyError:
    pass

  #------6--------Encode categorical features, and scale numerical features----------------/
  print('Extracting categoricals')
  categoricals_df=league.select_dtypes(include=['object'])
  try:
    categoricals_df.drop(['total_goals_more_than_3','btts','total_goals_more_than_2','away_team_wins','home_team_wins','draw'],axis=1,inplace=True)
  except:
    pass

  categoricals_df.fillna('NA',inplace=True)
  print(f'Categoricals shape is {categoricals_df.shape}')
  print('Extracting numericals')

  numerical_df=league.select_dtypes(include=['number'])
  try:
    numerical_df.drop(['total_goals_more_than_3','btts','total_goals_more_than_2','away_team_wins','home_team_wins','draw'],axis=1,inplace=True)
  except:
    pass
  for c in numerical_df.columns:
    if 'Unnamed' in c:
      numerical_df.drop([c],axis=1,inplace=True)

  numerical_df.fillna(0,inplace=True)
  print(f'Numericals shape is {numerical_df.shape}')

  print('Encoding categoricals')
  ohe=OneHotEncoder(handle_unknown='ignore',sparse=False)
  categoricals_df=pd.DataFrame(ohe.fit_transform(categoricals_df),index=categoricals_df.index)
  ohe_file=os.path.join(models_folder,'ohe_file.joblib')
  dump(ohe,ohe_file)
  print(f'encoder saved in {ohe_file}')
  print('Scaling numericals')

  scaler=MinMaxScaler()
  numerical_df=pd.DataFrame(scaler.fit_transform(numerical_df),index=numerical_df.index,columns=numerical_df.columns)
  scaler_file=os.path.join(models_folder,'scaler.joblib')
  dump(scaler,scaler_file)
  print(f'scaler saved in {scaler_file}')

  numerical_df=pd.DataFrame(scaler.transform(numerical_df),index=categoricals_df.index,columns=numerical_df.columns)

  #------7--------save categoricals, numericals and target data in different files-----------/

  targets=league[['total_goals_more_than_3','btts',"total_goals_more_than_2",'away_team_wins','home_team_wins','draw']]
  targets.to_csv(targets_file)
  categoricals_df=categoricals_df.join(numerical_df)
  categoricals_df.to_csv(features_file)



generate_input_files(frac=0.03) 
