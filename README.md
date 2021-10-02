This direcotory contains all trained models and training data. 
an example inference is also contained in here in data/predictions.csv

## Installing the environement :
if you have anaconda installed, open anaconda prompt and run this command


`conda env create --name footaball_betting --file=conda_env.yml`
## To do the training
run `download_data.py`, then `prepare_train_data.py` and then `train.py`
## To do testing on new matches
you should have pretrained models in models folder and all_leagues.csv file to use as match history
put a fixtures.csv file with columns: date, home_team, away_team
