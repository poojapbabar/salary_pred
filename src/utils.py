import os 
import sys 
import pickle 
import sqlite3
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging

class ConnectDB:
    def __init__(self):
        self.connection = None

    def establish_connection(self, host, username, password, database):
        try:
            self.connection = sqlite3.connect(database)
            print("Connection established successfully")
        except Exception as e:
            print("Error establishing connection:", e)

    def retrieve_data(self, query):
        try:
            if self.connection:
                cursor = self.connection.cursor()
                cursor.execute(query)
                data = cursor.fetchall()
                cursor.close()
                return data
            else:
                print("Connection not established. Please establish connection first.")
        except Exception as e:
            print("Error retrieving data:", e)

def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        for i, (name, model) in enumerate(models.items()): 
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score
        return report
    except Exception as e: 
        raise CustomException(e, sys)

def save_function(file_path, obj): 
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open (file_path, "wb") as file_obj: 
        pickle.dump(obj, file_obj)

def load_obj(file_path):
    try: 
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object function in utils")
        raise CustomException(e, sys)

def main():
    # Load your data or define X_train, y_train, X_test, y_test
    # Assuming you have loaded your data using pandas or another method
    # X_train, X_test, y_train, y_test = load_data_or_define_here()

    db = ConnectDB()
    db.establish_connection(host='localhost', username='root', password='shree.2002', database='Salary_predication.db')
    sql_query = "SELECT * FROM Salary_predication;"  # Modify this query as needed
    data = db.retrieve_data(sql_query)
    
    # Process the fetched data as needed
    for row in data:
        print(row)  # Example: Print each row fetched from the database

    # # Example models
    # models = {"model1": YourModel1(), "model2": YourModel2()}  # Replace with your actual models
    # performance_report = model_performance(X_train, y_train, X_test, y_test, models)
    # print(performance_report)

if __name__ == "__main__":
    main()
