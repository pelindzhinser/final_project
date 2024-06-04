# Final Project
# "Predicting Real Estate Data in St. Petersburg"

### Project Description
**Project goal**: real estate price forecasting in St. Petersburg

This project aims to analyze real estate data in St. Petersburg based on data from Yandex.Real Estate listings containing apartment listings from 2016 to mid-August 2018. 
In the course of the work was carried out: 
1) EDA data analysis plotting on the obtained data. 
2) Training of the model for predicting apartment prices
3) Transferring the data to a remote machine and creating a web application using Flask for easy access to the forecasts.
4) Using Docker to containerize the application, ensuring it can run on different environments without modification. 

## Baseline data and statistics ##

**Data source:** [realty.yandex.ru](https://realty.yandex.ru)

**EDA**

The data initially contained 429187 rows and 17 columns. Below is the info() about our raw table: 

![alt text](images/info.png)


Column information:
1. **offer_id**: Unique identifier of the real estate offer/offer.
2. **first_day_exposition**: The date the offer was first posted.
3. **last_day_exposition**: The date the offer was last posted.
4. **last_price**: The last price of the property.
5. **floor**: The floor on which the property is located.
6. **open_plan**: Flag indicating if the layout is open.
7. **rooms**: The number of rooms in the property.
8. **studio**: Flag indicating if the property is a studio. 9. area: The total area of the property.
9. **area**: The total area of the property.
10. **kitchen_area**: The area of the kitchen.
11. **living_area**: Living area.
12. **agent_fee**: Agent's commission.
13. **renovation**: Renovation costs.
14. **offer_type**: 2 - RENT, 1 - SELL 
15. **category_type**: Category type.
16. **unified_address**: The unified address of the property.
17. **building_id**: Unique building identifier.

**We divided the table into two: the first with all rental information, the second with sales information.** 

Total rent data size: 171186

Total sell data size: 258001

### DataFrame about rent 
Prepare dataframe with rent data in city limits: 

```
rent_df = spb_df[spb_df.offer_type == 2].copy()
rent_df_spb = rent_df[rent_df.unified_address.str.contains('Россия, Санкт-Петербург')].copy()
 ````

The shape of this table is (156054, 17)

Description of important numerical attributes: 

![alt text](images/description1.png)

For better analytics, 2 columns were added: **price_per_sq_m** and **house_price_sqm_median** 

Outliers in the 'last price' column: 
![alt text](images/outliers.png)
Conclusion: we need to logarithm this column

To evaluate the **“last price”** column, let's introduce the **“visualize_property”** function, which is designed to visualize a single numeric attribute in a DataFrame . It creates plots to help understand the distribution and characteristics of the attribute.

1. Histogram plot: Shows the distribution of the feature values using a histogram, which helps you see how the feature values are distributed over a range.

2. QQ plot: This plot is used to compare the distribution of the trait values with a normal distribution. If the points on the plot lie close to a straight line, it indicates a normal distribution.

3. Box plot: It shows the distribution of the feature values as well as the presence of outliers.

After plotting these plots, the function also outputs skewness and kurtosis, which are measures of skewness and kurtosis respectively.

```
def visualize_property(df, feature):
    fig, axs = plt.subplots(3, figsize=(8, 15))
    # Histogram plot
    axs[0].set_title('Histogram')
    df[feature].hist(ax=axs[0])
    # QQ plot 
    axs[1].set_title('QQ')
    stats.probplot(df[feature], plot=axs[1])
    # Box plot
    axs[2].set_title('Box plot')
    sns.boxplot(df[feature], ax=axs[2], orient='h')
    print("Skewness: %f" % df[feature].skew())
    print("Kurtosis: %f" % df[feature].kurt())
```

Before we applied log transformation on the **“last price”** column this function showed these results:

Skewness and kurtosis decreased, distribution became normal.

![alt text](<images/функция .png>)
![alt text](images/функция1.png)

The QQ has gotten better too

Before:
![alt text](<images/Снимок экрана 2024-06-03 в 14.32.08.png>)

After:

![alt text](<images/Снимок экрана 2024-06-03 в 14.31.46.png>)



Before:
![alt text](<images/Снимок экрана 2024-06-03 в 14.35.50.png>)

After: 
![alt text](<images/Снимок экрана 2024-06-03 в 14.37.38.png>)

**Missing values**
![alt text](<images/Снимок экрана 2024-06-03 в 14.39.51.png>)


**Deleting columns that have no meaning to us:**
'offer_id', 'price_per_sq_m' , 'house_price_sqm_median','category_type', 'offer_type'

Explore relations between 'last_price' and 'area'
![alt text](<images/Снимок экрана 2024-06-03 в 14.46.46.png>)



## Machine learning model

**Necessary libraries for modeling:**

```
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn import metrics 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

Before creating the model, make sure all the data is prepared. The data looks like this:

![alt text](<images/Снимок экрана 2024-06-03 в 14.58.40.png>)

**Splitting data:** 

```
X = df[['open_plan', 'rooms', 'area', 'renovation']]
y = df['last_price']
df = pd.concat([X, y], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df[['open_plan', 'rooms', 'area', 'renovation']], df['last_price'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2, разделение 60-20-20

df_training = pd.concat([X_train, y_train], axis=1)
df_validation = pd.concat([X_val, y_val], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)
```

Training sample size: (93234, 5)

Validation sample size: (31079, 5)

Test sample size: (31079, 5)

**Standardization:**

```
from sklearn.preprocessing import StandardScaler

y_train = np.array(y_train).reshape(-1, 1)
y_val = np.array(y_val).reshape(-1, 1)

sc_X = StandardScaler()
sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_val = sc_X.transform(X_val)  # используйте transform для валидационного набора данных
y_train = sc_y.fit_transform(y_train)
y_val = sc_y.transform(y_val.reshape(-1, 1))  # используйте transform для валидационного набора данных
```
**Model training using decision trees:** RandomForestRegressor model, which is an ensemble method based on decision trees. The model has been trained on a training dataset for predicting apartment prices.

    MAE: 0.36450180897542317
    MSE: 0.3659391242188636
    RMSE: 0.6049290241167666

![alt text](<images/Снимок экрана 2024-06-03 в 15.44.12.png>)

**Model training with CatBoost:** CatBoostRegressor model, which is a gradient bousting method. This model has also been trained on a training dataset to predict the cost of apartments.

    MAE: 0.3643808046785357
    MSE: 0.36586152077406886
    MSE: 0.604864878112516

![alt text](<images/Снимок экрана 2024-06-03 в 15.45.51.png>)


Hyperparameter tuning with GridSearchCV: GridSearchCV to tune the hyperparameters of the RandomForestRegressor model. This means that different combinations of hyperparameter values were tried and the performance of the model on each set of values was evaluated. The best hyperparameters were then used for the final model.

```
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 20, 30, 50, 100, 200],
    'bootstrap': [True, False],
    'max_depth': [5, 10, 15], 
    'min_samples_split': [2, 3, 4], 
    'max_features': [1, 2, 3]
}

random_forest_model = RandomForestRegressor()

grid_search = GridSearchCV(random_forest_model, param_grid, cv=5)
grid_search.fit(X_train, y_train.ravel())

best_params = grid_search.best_params_
print(best_params)
```

###   Web service
This command updates the list of available packages from the software repositories: 

```
sudo apt-get update 
```

This command installs the Python package manager, pip, on your system. Pip is used to install, upgrade, and manage Python packages: 
```
sudo apt install python3-pip
```
Both of these commands require sudo privileges, which means that you must be logged in as a user with administrator privileges to run them.

**Installing Python Virtual Environment**

1. Install Virtualenv package using pip
2. Create a virtual environment for project:

```
pip install virtualenv
virtualenv venv
```
**Activating Virtual Environment**

1. Activate the virtual environment:

```
source venv/bin/activate
```
2. Verify that the virtual environment is active by checking the command prompt:


```
(venv) $
```

**Installing Flask**

```
pip install Flask
```

**Running Flask App**

```
from flask import Flask, request 
import joblib
import numpy

MODEL_PATH = 'mlmodels/model.pkl'
SCALER_X_PATH = 'mlmodels/scaler_x.pkl'
SCALER_Y_PATH = 'mlmodels/scaler_y.pkl'

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
sc_x = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

@app.route('/predict_price', methods = ['GET'])
def predict(): 
    args = request.args
    open_plan = args.get('open_plan', default = -1, type = int)
    rooms = args.get('rooms', default = -1, type = int)
    area = args.get('area', default = -1, type = int)
    renovation = args.get('renovation', default = -1, type = int)
    
    x = numpy.array([open_plan, rooms, area, renovation]).reshape(1,-1)
    x = sc_x.transform(x)
    
    result = model.predict(x)
    result = sc_y.inverse_transform(result.reshape(1,-1))
    
    return str(result[0][0])

if __name__ == '__main__': 
    app.run(debug = True, port = 7778, host = '0.0.0.0')
```

To open this on a virtual machine, you need to enter this code: 

``` 
python app.py 
```

![alt text](<images/Снимок экрана 2024-06-03 в 21.20.04.png>)
![alt text](<images/Снимок экрана 2024-06-03 в 21.21.41.png>)


This Python code uses the Flask microframework to create a simple web application that can predict the price of a property based on four features:

• Number of open plan rooms

• Number of rooms

• Area in square meters

• Renovation status (0 for no renovation, 1 for renovated)

The code loads a pre-trained machine learning model and two scalers (for the input features and the output target) from disk.

The predict() function handles incoming GET requests and extracts the four feature values from the request arguments. It then reshapes the input data into a format that the model expects and applies the input scaler to normalize the data.

The model is used to make a prediction, and the output is inverse-transformed using the output scaler to obtain the predicted price in the original units.

The predicted price is returned as a string in the response.

If the script is run directly (not imported as a module), it starts a Flask development server on port 7778 and listens on all network interfaces (0.0.0.0)

### Dockerfile

```
FROM ubuntu:20.04
MAINTAINER Pelin Dzhinser
RUN apt-get update -y
COPY . /opt/gsom_predictor
WORKDIR /opt/gsom_predictor
RUN apt install -y python3-pip
RUN pip3 install -r requirements.txt
CMD python3 app.py
```

1. FROM ubuntu:20.04:
   - Uses the base image of Ubuntu 20.04. The base image provides the file system structure and base environment for your application.

2. MAINTAINER Pelin Dzhinser:
   - This instruction is deprecated in Docker 1.13. It was previously used to specify a supporting person or organization. It is recommended that you use the LABEL metadata instead.

3.RUN apt-get update -y:
   - Updates the local Ubuntu package index to ensure that the latest versions of packages are installed.

4. COPY . /opt/gsom_predictor:
   - Copies all files from the current host directory (where the Dockerfile resides) inside the image to the /opt/gsom_predictor directory. This allows you to include the necessary files for your application in the image.

5. WORKDIR /opt/gsom_predictor:
   - Sets the working directory inside the /opt/gsom_predictor container. All subsequent instructions will be executed in this directory.

6. RUN apt install -y python3-pip:
   - Installs the python3-pip (Python package manager) package inside the container.

7. RUN pip3 install -r requirements.txt:
   - Installs the Python packages listed in the requirements.txt file. 

8. CMD python3 app.py:
   - Specifies the command that will be executed by default when the app.py container is started.

### How to open the port in remote VM

```
ssh <login>@<your_vm_address>
```

### How to run app using docker and which port it uses

```
docker run --network host -d pelindzhinser/gsom_e2e24:v.0.1
sudo ufw allow 7778
```

Example of work: 

![alt text](<Снимок экрана 2024-06-04 в 07.50.24.png>)
