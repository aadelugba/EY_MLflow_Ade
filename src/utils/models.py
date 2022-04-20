from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.model_selection import KFold,train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import _tree
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def transformation_pipeline(model):
    '''
    This is the transformation pipeline that takes in the model and defines transformation steps/pipeline to be applied to the training data.

    Input Param:
        ML Model

    Return Value:
        Pipeline object indicating all the steps required to prep data and build model
        
    '''

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                          ('scaler', StandardScaler())]
                                          )

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))]
                                              )

    preprocessor = ColumnTransformer(
                            transformers=[('num', numeric_transformer, make_column_selector(dtype_exclude="category")),
                                          ('cat', categorical_transformer, make_column_selector(dtype_include="category"))]
                                          )

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model)]) 

    return model_pipeline


class BasicModel:
    '''
    Class for creating a model

    Input Params:
        Data path
        Id columns to be excluded from model training
        Label column to enable distinction of features (X) from label (y)
        Test size
        Model to be used (string representation)
        Params dictionary to be unpacked

    Return Value:
        Initialised model

    '''

    def __init__(self, data_path, id_col, label_col, test_size = 0.3, model='rf', **params):

        # Initialise df, test size and params
        self.df = pd.read_csv(data_path).drop(id_col, axis=1)
        self.test_size = test_size
        self.params = params

        # Get features and labels
        self.X = self.df.drop([label_col], axis = 1)
        self.y = self.df[label_col]
    
        # Train Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 123)

        # Initialise sklearn model to use
        if model == 'dt':
            self.model = DecisionTreeClassifier(**self.params)
        elif model == 'svc':
            self.model = SVC(**self.params)
        elif model == 'et':
            self.model = ExtraTreeClassifier(**self.params)
        elif model == 'gp':
            self.model = GaussianProcessClassifier(**self.params)
        elif model == 'rf':
            self.model = RandomForestClassifier(**self.params)
        elif model == 'knn':
            self.model = KNeighborsClassifier(**self.params)
        elif model == 'lr':
            self.model = LogisticRegression(**self.params)  


    def fit(self):
        '''
        Fit function to fit model to Training set defined in the constructor above.
        It first applies the transformation pipeline to the model
        Then uses the transformed pipeline/model to fit the training set

        '''

        pipelines = transformation_pipeline(self.model)

        self.trained_model = pipelines.fit(self.X_train, self.y_train)
    

    def predict(self):
        '''
        Predict function to predict the Response/Dependent variable based on the Regressor/Explanatory/Independent/Manipulated/Predictor imputted.

        Input Params:
            N/A
        
        Return Value:
            Predicted values

        '''

        predictions = self.trained_model.predict(self.X_test)

        return predictions


    def scoring(self, pred):
        '''
        Scoring function after applying 10 fold cross validation

        Input Params:
            N/A

        Return Value:
            Accuracy, Recall and Precision scores

        '''
        acc = accuracy_score(self.y_test, pred)
        rec = recall_score(self.y_test, pred)
        prec = precision_score(self.y_test, pred)

        return acc, rec, prec