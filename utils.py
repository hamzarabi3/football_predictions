# -----0----------------functions for adding previous match results and goals------------------------------------/



def get_previous_match_result(match_index, team_name, date, df):
    try:
        filter1 = df.query("away_team==@team_name and date<@date")
        filter1 = filter1.sort_values(by="date")
        filter1 = filter1.iloc[-match_index]
        if filter1["FTR"] == "A":
            filter1["FTR"] = "W"
        elif filter1["FTR"] == "H":
            filter1["FTR"] = "L"
    except:
        filter1 = df.query('away_team=="me playing in the kitchen"')

    try:
        filter2 = df.query("home_team==@team_name and date<@date")
        filter2 = filter2.sort_values(by="date")
        filter2 = filter2.iloc[-match_index]
        if filter2["FTR"] == "A":
            filter2["FTR"] = "L"
        elif filter2["FTR"] == "H":
            filter2["FTR"] = "W"
    except:
        filter2 = df.query('away_team=="me playing in the kitchen"')

    if len(filter1) == 0 and len(filter2) > 0:
        return filter2["FTR"] 
    if len(filter1) > 0 and len(filter2) == 0:
        return filter1["FTR"]
    if len(filter1) == 0 and len(filter2) == 0:
        return None

    if filter1.date < filter2.date:
        return filter2["FTR"]
    return filter1["FTR"]


def get_previous_match_goals(match_index, team_name, date, df):
    try:
        filter1 = df.query("away_team==@team_name and date<@date")
        filter1 = filter1.sort_values(by="date")
        filter1 = filter1.iloc[-match_index]
        date1 = filter1.date
        filter1 = filter1["away_goals"]
    except:
        filter1 = None
    try:
        filter2 = df.query("home_team==@team_name and date<@date")
        filter2 = filter2.sort_values(by="date")
        filter2 = filter2.iloc[-match_index]
        date2 = filter2.date

        filter2 = filter2["home_goals"]
    except:
        filter2 = None

    if filter1 is None and filter2 is not None:
        return filter2
    if filter1 is not None and filter2 is None:
        return filter1
    if filter1 is None and filter2 is None:
        return None
    if date1 < date2:
        return filter2
    return filter1
