import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

_df = pd.read_csv("StudentsPerformance.csv")

_df["gender"] = _df["gender"].map({
    "female": 0,
    "male": 1
})

_df["lunch"] = _df["lunch"].map({
    "standard": 1,
    "free/reduced": 0
})

_df["test preparation course"] = _df["test preparation course"].map({
    "none": 0,
    "completed": 1
})

_df["parental level of education"] = _df["parental level of education"].map({
    "some high school": .1,
    "high school": .3,
    "some college": .5,
    "associate's degree": .7,
    "bachelor's degree": .9,
    "master's degree": 1
})

_df = pd.get_dummies(
    _df,
    columns=["race/ethnicity"],
    dtype=int
)

df = _df.drop(columns=["reading score", "writing score"])

X = df.drop(columns=["math score"])
y = df["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)

results = model.predict(X_test)

MSE = mean_squared_error(y_test, results)

print(MSE)
