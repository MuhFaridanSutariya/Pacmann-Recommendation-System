import pandas as pd

def load_data():
    df1 = pd.read_csv('data/tourism_rating.csv')
    df2 = pd.read_csv('data/tourism_with_id.csv')
    df3 = pd.read_csv('data/user.csv')

    raw = pd.merge(df1, df3, on='User_Id')

    df = pd.merge(df2, raw, on='Place_Id').drop(columns=['Unnamed: 11', 'Unnamed: 12', 'Lat', 'Coordinate', 'Long', 'Time_Minutes'])

    return df

def get_place_encodings(df):
    place_ids = df['Place_Id'].unique().tolist()
    user_ids = df['User_Id'].unique().tolist()

    user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

    place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
    place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}
    
    return place_to_place_encoded, place_encoded_to_place, user_to_user_encoded