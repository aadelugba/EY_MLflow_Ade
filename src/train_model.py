# Import modules and libraries needed
from distutils.command.config import config
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.tracking import MlflowClient as mf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope
import numpy as np
import pandas as pd
from joblib import dump
from src.utils.config import get_configs
from src.utils.models import BasicModel

# try the following using Rapids functions cudf, and cuml --> it enables us run on GPUs which is far faster

def train_model(params, test_size = 0.3, registered_model_name= None):
    '''
    This function takes params, fits a model, registers it with mlflow and returns a loss function for hyperopt parameter tuning.

        Parameters:
            params (dict): Parameters to tune
            test_size (float): Test set size
            registered_model_name (str): Name for the registered model

        Return:
            loss (dict): Dictionary of loss function
    '''

    # Get configs
    configs = get_configs()
    
    # Get config parameters to initialise model class
    data_path = configs['DATA_PATH']
    id_col = configs['ID_COL']
    label_col = configs['LABEL_COL']
    model_type = configs['MODEL_TYPE']

    # Initialise and fit model
    model = BasicModel(data_path, id_col, label_col, test_size=test_size, model=model_type, **params)
    model.fit()
    
    # Log model with mlflow
    mlflow.sklearn.log_model(model, artifact_path=configs['ARTIFACT_PATH'], registered_model_name=registered_model_name)
     
    # Log metrics
    if test_size > 0.0:
        
        y_pred = model.predict()
        acc, rec, prec = model.scoring(y_pred)   
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        
    else:
        acc = np.nan
        prec = np.nan
        rec = np.nan
    
    # Return loss function
    return {'loss': -acc, 'status': STATUS_OK}


def get_run_id(loss_function):
    '''
    This function takes a loss function and run trials to get the best parameters of the model built.

        Parameters:
            loss_function (dict): Loss function

        Return:
            best (dict): Dictionary of best parameters
    '''

    # Get configs
    configs = get_configs()

    # Configure search space
    search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
            'max_features': hp.uniform("max_features", 0.0, 1.0),
            'n_estimators': scope.int(hp.quniform('n_estimators', 100, 200, 50))
            }
    #configs['SEARCH_PARAMS']

    # Set algo as tpe
    algo = tpe.suggest

    # Define spark trials object
    trials = Trials()

    mlflow.end_run() # close out any run in progress

    # Run mlflow with the hyper parameter tuning job
    with mlflow.start_run() as run:

        # Set mlflow tags
        mlflow.set_tags({"model_type": configs['MODEL_NAME'], "resource_type": configs['WORK_TYPE']})

        # Get best hyper parameters using hyperopt
        best = fmin(
                    fn = loss_function,
                    space = search_space,
                    algo = algo,
                    trials = trials,
                    max_evals = 2,
                )
        
        # Set mlflow tags
        mlflow.set_tag("best_params", str(best))

        # get bets params
        best_params = {"max_features": best["max_features"], "n_estimators": int(best["n_estimators"]), "max_depth" : int(best["max_depth"])}
        
        mlflow.log_params(best_params)

        # Train the model
        train_model(best_params)

        # Get run id and return
        run_id = mlflow.active_run().info.run_id

    mlflow.end_run()

    return run_id