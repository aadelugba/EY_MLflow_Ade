import yaml
import os
from hyperopt import  hp
from hyperopt.pyll import scope

# Create configurations file
def create_configs():
    '''
    This is to create the configurations for training the model

    Notes on Model Type: 'dt' = DecisionTreeClassifier(),
                         'svc' = SVC(),
                         'et' = ExtraTreeClassifier(),
                         'gp' = GaussianProcessClassifier(),
                         'rf' = RandomForestClassifier(),
                         'knn': KNeighborsClassifier(),
                         'lr': LogisticRegression()  
     
    '''
    config_settings = [
        {
        'DATA_PATH' : 'data/diabetes.csv',
        'MODEL_NAME' : 'Classification Model',
        'MODEL_TYPE' : 'rf',
        'MLFOW_TRACKING_URI' : 'http://localhost:5000',
        'ARTIFACT_PATH' : 'Model_Artifacts',
        'WORK_TYPE' : "POC Work",
        'ID_COL' : 'PatientID',
        'LABEL_COL' : 'Diabetic',
        # 'SEARCH_PARAMS' : {
        #     'max_depth': scope.int(hp.quniform('max_depth', 1, 10, 1)),
        #     'max_features': hp.uniform("max_features", 0.0, 1.0),
        #     'n_estimators': scope.int(hp.quniform('n_estimators', 100, 200, 50))
        #     }
    
        }
    ]

    with open("config_settings.yaml", 'w') as yamlfile:
        configs = yaml.dump(config_settings, yamlfile)


# Invoke the configutions file
def get_configs():
    if not os.path.exists("config_settings.yaml"):
        create_configs()
    with open("config_settings.yaml", "r") as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)[0]
    return configs


