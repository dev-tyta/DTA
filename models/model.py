# importing python libraries
import pandas as pd
import pickle as pkl
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("./dataset/diabetes.csv")

