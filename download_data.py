import pandas as pd
import os

from pandas.core import frame

data_folder='data'
league='E0'

frames=[]
#split into two chunks
#before 2000

start_season0=93
end_season0=99
for i in range(start_season0, end_season0):
    start="0"+str(i) if i <10 else str(i)
    end="0"+str(i+1) if i+1<10 else str(i+1)
    try:
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/"
            + start
            + end
            + "/"
            + league
            + ".csv"
        )
        df = df.assign(season=i)
        print(f'season {start}/{end} shape is {df.shape}')

        frames.append(df)
    except:
        print(f'season {start}/{end} not read')
#1999/2000 season:
start=str(99)
end='00'
try:
    df = pd.read_csv(
        "http://www.football-data.co.uk/mmz4281/"
        + start
        + end
        + "/"
        + league
        + ".csv"
    )
    df = df.assign(season=i)
    print(f'season {start}/{end} shape is {df.shape}')
    frames.append(df)
except:
    print(f'season {start}/{end} not read')


start_season1=0
end_season1=21

#after 2000
for i in range(start_season1, end_season1):
    start="0"+str(i) if i <10 else str(i)
    end="0"+str(i+1) if i+1<10 else str(i+1)
    try:
        df = pd.read_csv(
            "http://www.football-data.co.uk/mmz4281/"
            + start
            + end
            + "/"
            + league
            + ".csv"
        )
        df = df.assign(season=i)
        print(f'season {start}/{end} shape is {df.shape}')
        frames.append(df)
    except:
        print(f'season {start}/{end} shape is {df.shape}')


sets=[set(frame.columns) for frame in frames]
to_keep=list(set.intersection(*sets))
frames=[frame[to_keep] for frame in frames]
for frame in frames:
    print(frame.shape)
df_frames = pd.concat(frames)

save_as=os.path.join(data_folder,f'match_results_{league}_{start_season0}_{end_season1}.csv')
df_frames.to_csv(save_as)
    