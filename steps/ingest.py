import pandas as pd
#from database.sqlScript import Sql

def ingest_fn(data_path):
    # host = ""
    # password = ""
    # dbname = ""
    # user = ""
    # trust = ""
    # # Injection data from database
    # sql = Sql(host,password, dbname, user,trust)
    # sql.enable_connection()
    # df = sql.load_data()
    df = pd.read_csv(data_path)
    print("Injest Done.........")
    return df
