import pandas as pd
#import matplotlib.pyplot as plt
from utility._utili import make_short_form, get_season, is_eid_ul_adha, is_monsoon, is_ramadan, extract_medicine_type, insert_medicine_type

# Wedding dict()
wedding_seasons = {
    3: 1,  # March
    4: 1,  # April
    5: 1,  # May
    9: 1,  # September
    10: 1,  # October
    11: 1  # November
}

def preprocess_fn(data):
    """
    Implement your preprocess function here
    """

    data.columns = ['hospital', 'store', 'year', 'month','item', 'transaction', 'y', 'received']

    data['hospital'] = data['hospital'].apply(make_short_form)
    data['store'] = data['store'].str.replace('OPD Pharmacy', 'OPD', regex=False)
    data['store'] = data['store'].str.replace('Repeat Medicine Store', 'Repeat', regex=False)
    data['item'] = data['item'].str.replace(' ', '.', regex=False).str.lower()
    data['unique_id'] = data['hospital'].astype(str) + '_' + data['store'].astype(str) + '_' + data['item'].astype(str)
    data['ds'] = pd.to_datetime(data[['year', 'month']].assign(day=1))

    # filter pos
    data = data[data['transaction']=='POS']

    # remove received
    data['y'] = data['y']-data['received']
    data = data[['ds', 'unique_id', 'y']]

    data['ds'] = pd.to_datetime(data['ds'])
    date_range = pd.date_range(start='2020-01-01', end='2024-05-01', freq='MS')
    all_combinations = pd.DataFrame([(date, unique_id) for date in date_range for unique_id in data['unique_id']], columns=['ds', 'unique_id'])
    data_complete = pd.merge(all_combinations, data, on=['ds', 'unique_id'], how='left')
    data_complete['y'].fillna(0, inplace=True)

    data_complete = data_complete.drop_duplicates(subset=['ds', 'unique_id'])

    # Convert 'ds' column to datetime type
    data_complete['ds'] = pd.to_datetime(data_complete['ds'])
    data_complete['year'] = data_complete['ds'].dt.year
    data_complete['month'] = data_complete['ds'].dt.month
    data_complete['quarter'] = data_complete['ds'].dt.quarter
    # Extract Season based on Pakistan Seasons
    data_complete['season'] = data_complete['month'].apply(get_season)
    
    # Extract month of Eid_ul_Adha
    data_complete['eid_ul_adha'] = data_complete.apply(lambda row: is_eid_ul_adha(row['year'], row['month']), axis=1)

    # Extract Ramdan month
    data_complete['ramadan'] = data_complete.apply(lambda row: is_ramadan(row['year'], row['month']), axis=1)

    # Extract Moonson Season
    data_complete['monsoon'] = data_complete['month'].apply(is_monsoon)

    # Extract wedding Seanson
    data_complete['wedding_season'] = data_complete['month'].map(wedding_seasons).fillna(0).astype(int)

    data_complete = pd.get_dummies(data_complete, columns=['season'])

    df = data_complete
    #df_ls.append(data_complete)

    #df = pd.concat(df_ls[0], df_ls[1])
    df_types = ['cap', 'syringe', 'tab', 'SYRINJE', 'crm', 'drip', 'drp', 'gel', 'inj', 'sach', 'syp', 'advance', 'inj', 'milk']
    df['unique_id'], df['medicine_type'] = zip(*df['unique_id'].apply(lambda x: extract_medicine_type(x, df_types)))
    df['unique_id'] = df.apply(lambda row: insert_medicine_type(row['unique_id'], row['medicine_type']), axis=1)
    
    print("Preprocess Done...")
    return df