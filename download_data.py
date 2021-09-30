# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 13:39:39 2021

"""
import pandas as pd
import os 
#we create a dictionary dict_countries similar to the one we used in the scraping section 
#but in this case we need to specify the names considered in the link to download the csv e.g. "SP1" (Spanish League)
dict_countries = {
              'Spanish La Liga':'SP1', 'Spanish Segunda Division':'SP2',
              'German Bundesliga':'D1', 
              'German Bundesliga 2':'D2',
              'Italian Serie A':'I1', 
              'Italian Serie B':'I2',
              'English Premier League':'E0', 'English League 1':'E2', 'English League 2':'E3',
              'French Ligue 1': 'F1', 'French Ligue 2':'F2',
              'Dutch Eredivisie':'N1',
              'Belgian First Division A':'B1',
              'Portuguese Primeira Liga':'P1',
              'Turkish Super League':'T1',
              'Greek Super League':'G1'
             }

#dict_historical_data contains data of the past 5 years. we'll use it to manage 2 dataframes: df_historical_data and df_profile
dict_historical_data = {} 

#to download all the leagues we loop through the dictionary
for league in dict_countries:
    frames = []
    for i in range(10, 21):
        try:
            df = pd.read_csv("http://www.football-data.co.uk/mmz4281/"+str(i)+str(i+1)+"/"+dict_countries[league]+".csv")
        except: #Italian Serie B (0xa0 utf-8)
            df = pd.read_csv("http://www.football-data.co.uk/mmz4281/"+str(i)+str(i+1)+"/"+dict_countries[league]+".csv", encoding='unicode_escape')
        df = df.assign(season=i)
        frames.append(df)
    df_frames = pd.concat(frames)
    df_frames = df_frames.rename(columns={'Date':'date', 'HomeTeam':'home_team', 'AwayTeam':'away_team',
                        'FTHG': 'home_goals', 'FTAG': 'away_goals'})
    dict_historical_data[league] = df_frames


MLS2020 = pd.read_csv("https://www.football-data.co.uk/new/USA.csv")
Swed2020 = pd.read_csv("https://www.football-data.co.uk/new/SWE.csv")
Nor2020 = pd.read_csv("https://www.football-data.co.uk/new/NOR.csv")
LigaMx2020 = pd.read_csv("https://www.football-data.co.uk/new/MEX.csv")
Arg = pd.read_csv("https://www.football-data.co.uk/new/ARG.csv")
AUT = pd.read_csv("https://www.football-data.co.uk/new/AUT.csv")
BRA = pd.read_csv("https://www.football-data.co.uk/new/BRA.csv")
CHN = pd.read_csv("https://www.football-data.co.uk/new/CHN.csv")
FIN = pd.read_csv("https://www.football-data.co.uk/new/FIN.csv")
    
Other_Leagues = pd.concat([MLS2020, Swed2020, Nor2020, LigaMx2020, Arg, AUT, BRA, CHN, FIN])\


Other_Leagues.Date=pd.to_datetime(Other_Leagues.Date)
Other_Leagues = Other_Leagues.rename(columns={'Date':'date', 'Home':'home_team', 'Away':'away_team',
                        'HG': 'home_goals', 'AG': 'away_goals','Res':'FTR'})

leagues=list(dict_historical_data.keys())
sets=[set(dict_historical_data[l].columns) for l in leagues]+[set(Other_Leagues.columns)]
common_columns=set.intersection(*sets)

data_file=os.path.join('data','all_leagues.csv')

