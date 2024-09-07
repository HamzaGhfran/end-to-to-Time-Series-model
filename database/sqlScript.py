import sql
import pyodbc
import pandas as pd
class Sql:
    """
    class is created to build connection with Microsoft SQL database.
    Class contain methods and connection attributes

    enable_connection(): Method is used to enable connection with database
    run_query(): Method is used for run different queries on database
    load_data(): Method is used to load data from database to panda dataframe
    close_connection(): Method is used for end connection with database
    """


    def __init__(self, host,password, dbname, user,trust):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.trust = trust
        self.connection = None
        self.cursor = None
    
    def enable_connection(self):
        try:
            connectionString = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.host};DATABASE={self.dbname};UID={self.user};PWD={self.password};TrustServerCertificate={self.trust}'
            self.connection = pyodbc.connect(connectionString)
            print(self.connection)
            return 'connection establish successfully'
        except Exception as e:
            return "the error '{e}' occured"
        
    def run_query(self, query):
        try:
            self.cursor = self.connection.cursor()
            self.cursor.execute(query)
            records = self.cursor.fetchall()
            return records
        except Exception as e:
             return "the error '{e}' occured during query process"
    
    def load_data(self, query):
        try:
            df = pd.read_sql_query(query, self.connection)
            return df
        except Exception as e:
            return "the error '{e}' occured during data load in pandas"

        
    def close_connection(self):
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            return 'Connection closed'
        except Exception as e:
            return "error '{e}' occured during close connection"