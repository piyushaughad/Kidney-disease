import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For handling imbalanced datasets by creating synthetic samples
from imblearn.over_sampling import SMOTE

# Statistical tools for diagnosing multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# For feature scaling
from sklearn.preprocessing import StandardScaler

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

# Model selection and hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV

# Evaluation metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,confusion_matrix

import streamlit as st


st.set_page_config(page_title="Chronical kidney disease ML Dashboard", layout="wide")
st.title("🎗️ Chronical kidney disease Machine Learning Classification Dashboard")
st.markdown("**Created by Piyush Aughad**")
