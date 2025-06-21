import os 
import sys
import numpy as np
import pandas as pd
import dill

from src.exceptions import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}
        fitted_models={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            random=RandomizedSearchCV(model,para,cv=5,n_iter=15,n_jobs=-1,verbose=2,refit=True)
            random.fit(x_train,y_train)
            best_model=random.best_estimator_
            y_train_pred=best_model.predict(x_train)
            y_test_pred=best_model.predict(x_test)
            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score
            fitted_models[list(models.keys())[i]]=best_model
        return report,fitted_models
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)