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

    df.to_csv("./data/raw_data.csv")
    #return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/saadHospitalPharmacy.csv")
    args = parser.parse_args()
    ingest_fn(args.data_path)
