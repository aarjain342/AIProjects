import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from matplotlib import pyplot as plt

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

_df["reading score"] = _df["reading score"].apply(lambda x: x/100)
_df["writing score"] = _df["writing score"].apply(lambda x: x/100)
_df["math score"] = _df["math score"].apply(lambda x: x/100)

df = _df

predicting_column = "math score"

X = df.drop(columns=[predicting_column])
y = df[predicting_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=128)

model = LinearRegression()
model.fit(X_train, y_train)

results = model.predict(X_test)

RMSE = root_mean_squared_error(y_test,results)

plt.plot(range(len(y_test.values)), y_test, '--b')
plt.plot(range(len(results)), results, '--r')
plt.legend(["Actual", "Predicted"])
plt.show()

print(f"Error: {RMSE}")

for feature, val in zip(X.columns, model.coef_):
    print(f"Parameter: {feature} | Weight: {val}")


