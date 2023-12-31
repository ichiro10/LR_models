import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso ,ElasticNet
from sklearn.dummy import DummyRegressor

# Define a list of baseline strategies and their respective arguments
baseline_strategies = [
    {"strategy": "constant", "constant": 50},
    {"strategy": "quantile", "quantile": 0.75},
    {"strategy": "mean"},
    {"strategy": "median"}
]

# Define a function to create and evaluate dummy regressors
def evaluate_dummy_regressor(strategy_args, X_train, y_train, X_test, y_test):
    baseline = DummyRegressor(**strategy_args)
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    predictionVStrue(y_test, y_pred, mse, r2, 'baseline', strategy_args)


def predictionVStrue(y_true, y_pred, mse, r2 ,name ,strategy):
    plt.scatter(y_true, y_pred, label=f'{name},{strategy}\nMSE: {mse:.2f}, R-squared: {r2:.2f}')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red', label='Perfect Prediction')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.legend()
    plt.title("Regression Model Predictions")
    plt.show()    


def residual_plot(y_true, y_pred, model_name):
        plt.plot(y_pred, y_true - y_pred, "*")
        plt.plot(y_pred, np.zeros_like(y_pred), "-")
        plt.legend(["Data", "Perfection"])
        plt.title("Residual Plot of " + model_name)
        plt.xlabel("Predicted Value")
        plt.ylabel("Residual")
        plt.show()   


def Preparing_models(X_train, X_test, y_train):
        # Créez une liste pour stocker les modèles
        models = []

        # Étape 2 : Initialisez et entraînez différents modèles
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        models.append(("Linear Regression", linear_model))
        

        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        models.append(("Ridge Regression", ridge_model))
        
        alpha_values = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

        for i, a in enumerate(alpha_values):
                lasso_model = Lasso(alpha=a)
                lasso_model.fit(X_train, y_train)
                models.append(("Lasso Regression"+"alpha=" +str(a), lasso_model))

        # elastic net
        alpha_values = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        for i, a in enumerate(alpha_values):
                ElasticNet_model = ElasticNet(alpha = a, l1_ratio=0.4)
                ElasticNet_model.fit(X_train, y_train)
                models.append(("ElasticNet Regression"+"alpha=" +str(a), ElasticNet_model))
        

        return models

def data_preprocessing(data):
        print(data.info())
        print(data.nunique())
        print(data.describe(include='all'))

        #data_viz(data)
        #missed values 
        print(data.isnull().sum())

        #categorical data 
        dummies = pd.get_dummies(data["RAD"],dtype=int ,prefix="RAD")
        #Concatenate the original DataFrame with the dummy variables
        data = pd.concat([data, dummies], axis=1)

        # Drop the original categorical column 
        data = data.drop(columns=['RAD'])
        
        #outliers 
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=100)

        for col in data.columns:
            data[col] = data[col] = qt.fit_transform(pd.DataFrame(data[col]))
        

        for col in data:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                whisker_width = 1.5
                lower_whisker = q1 - (whisker_width * iqr)
                upper_whisker = q3 + whisker_width * iqr
                data[col] = np.where(data[col] > upper_whisker, upper_whisker, np.where(data[col] < lower_whisker, lower_whisker, data[col]))
       
        params = data.drop('MEDV', axis=1)
        target = data['MEDV']

        scalar = StandardScaler()
        scalar.fit(params)
        scaled_inputs = scalar.transform(params)

        return scaled_inputs, target

def data_viz(data):
      for column in data.columns : 
        plt.figure(figsize = (14,4))
        sns.histplot(data[column])
        plt.title(column)
        plt.show()    

      for column in data:
                sns.boxplot(data = data, x = column)
                plt.show()    

      params = data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',  'TAX',
        'PTRATIO', 'B', 'LSTAT','RAD']]
      plt.figure(figsize=(10,10))
      sns.set_theme()
      sns.heatmap(params.corr(),annot=True, fmt="0.1g", cmap='PiYG')
      plt.show()            

      
data = pd.read_csv('c:/Users/ghamm/OneDrive/Bureau/UQAC/TP/AA/boston.csv')
plt.figure(figsize=(12, 6))

scaled_inputs, target = data_preprocessing(data)
#split the data
X_train, X_test, y_train, y_test = train_test_split(scaled_inputs, target, test_size=0.2, random_state = 42)

# Loop through the baseline strategies and evaluate dummy regressors
for strategy_args in baseline_strategies:
    evaluate_dummy_regressor(strategy_args, X_train, y_train, X_test, y_test)


models= Preparing_models(X_train, X_test, y_train)
for name, model in models:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    predictionVStrue(y_test, y_pred, mse, r2,name, strategy='')
    residual_plot(y_test, y_pred, name)
    # Print results
    print(f"Model: {name}")
    print("MSE and R2 score:")
    print(f"MSE: {mse}, R^2: {r2}")
    print("Coefficients:")
    print(model.coef_)
    print("Intercept:")
    print(model.intercept_)
    print("\n")
