import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import root_mean_squared_error

def normalize(df):
    df = df.drop(columns=["Name"])

    df["Pclass"] = df["Pclass"].map({
        1: .3,
        2: .6,
        3: .9,
    })

    df["Sex"] = df["Sex"].map({
        "male": 0,
        "female": 1
    })

    df["Age"] = df["Age"].apply(lambda x: x/100)
    df["Fare"] = df["Fare"].apply(lambda x: x/100)

    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        dtype=int
    )

    # Temporary, see how model performs without these two variables
    df = df.drop(columns=["Ticket", "Cabin"])

    df = df.fillna(0)

    return df

df1 = pd.read_csv("train.csv")
df2 = pd.read_csv("test.csv")
df1 = normalize(df1)
df2 = normalize(df2)

X_train = df1.drop(columns=["Survived"])
y_train = df1["Survived"]
X_test = df2
y_test = pd.DataFrame()

model = LogisticRegression(max_iter=10000).fit(X_train, y_train)

results = model.predict(X_test)

resultsdf = pd.DataFrame(results, columns=[ "Survived"], index=range(892, 892+ len(results)))

print(resultsdf.head())

resultsdf.index.name = "PassengerId"

resultsdf.to_csv("submission.csv")

print(results)


