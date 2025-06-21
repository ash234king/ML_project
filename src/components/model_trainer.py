import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={
                "Linear Regression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "ElasticNet":ElasticNet(),
                "K-Nearest Neighbours":KNeighborsRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "Gradient Boost regressor":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(),
            }
            params={
                "Decision Tree":{
                    'criterion' : ['squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter' : ['best','random'],
                    'max_features': ['sqrt','log2']
                },
                "Linear Regression": {
                    'fit_intercept':[True,False]
                },
                "Ridge":{
                    'alpha': [0.1,0.2,0.01,1,0.03],
                    'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga']
                },
                "Lasso":{
                    'alpha': [0.1,0.2,0.01,1,0.03],
                    'selection':['cyclic','random']
                },
                "ElasticNet":{
                    'alpha': [0.1,0.2,0.01,1,0.03],
                    'l1_ratio':[0.2,0.1,0.3,0.4,0.5],
                    'selection':['random','cyclic']
                },
                "K-Nearest Neighbours":{
                    'weights':['uniform','distance'],
                    'algorithm':['auto','ball_tree','kd_tree','brute'],
                    'metric':['minkowski']
                },
                "Random Forest Regressor":{
                    'n_estimators':[10,20,40,50,100],
                    'criterion':['squared_error','absolute_error','friedman_mse','poisson'],
                    'max_features':['sqrt','log2'],
                },
                "Adaboost Regressor":{
                    'learning_rate':[0.1,0.2,0.01,1,0.5],
                    'loss':['linear','square','exponential'],
                    'n_estimators':[2,8,32,64,88]
                },
                "Gradient Boost regressor":{
                    'loss':['squared_error','absolute_error','huber','quantile'],
                    'learning_rate':[0.1,0.2,0.01,1,0.5],
                    'n_estimators':[2,8,32,64,88],
                    'criterion':['friedman_mse', 'squared_error'],
                    'max_features':['sqrt','log2']
                },
                "XGBRegressor":{
                    'booster':['gbtree','gblinear','dart'],
                    'n_estimators':[2,8,32,64,88],
                    'learning_rate':[0.1,0.2,0.01,1,0.5]
                },
                "CatBoosting Regressor":{
                    'learning_rate':[0.1,0.2,0.01,1,0.5],
                    'loss_function':['RMSE','MAE','MAPE','Quantile'],
                    'od_type':['IncToDec','Iter'],
                    'boosting_type':['Ordered','Plain']
                }
            }
            model_report,fitted_models=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                             models=models,param=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=fitted_models[best_model_name]
            if(best_model_score<0.6):
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
            
