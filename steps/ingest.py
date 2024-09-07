import pandas as pd
#from database.sqlScript import Sql

def ingest_fn():
    # host = ""
    # password = ""
    # dbname = ""
    # user = ""
    # trust = ""
    # # Injection data from database
    # sql = Sql(host,password, dbname, user,trust)
    # sql.enable_connection()
    # df = sql.load_data()
    dfs = []
    df = pd.read_csv("/home/hamza/BSCS/TSF/data/saadHospitalPharmacy.csv")
    print("Injest Done.........")
    return df
