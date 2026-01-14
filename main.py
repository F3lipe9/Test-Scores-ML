import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


data = pd.read_csv("student_admission_record_dirty.csv")

def cleanData(df):

    df = df.drop(columns=(["Name", "City", "Gender", "Age"]))

    df = df[df["Admission Status"].isnull() == False]
    df = df[df["Admission Test Score"].isnull() == False]
    df = df[df["High School Percentage"].isnull() == False]
    df = df[df["Admission Test Score"] > 0]
    df = df[df["High School Percentage"] > 0]
    df = df[df["Admission Test Score"] < 100]
    df = df[df["High School Percentage"] < 100]
    df = df[~((df["High School Percentage"] < 50) & (df["Admission Status"] == 1))]
    df = df[~((df["Admission Test Score"] < 50) & (df["Admission Status"] == 1))]
    df = df[~((df["High School Percentage"] > 90) & (df["Admission Status"] == 0))]
    df = df[~((df["Admission Test Score"] > 90) & (df["Admission Status"] == 0))]


    df["Score*Percent"] = df["Admission Test Score"] * df["High School Percentage"]
    df["ScoreMinusPercent"] = df["Admission Test Score"] - df["High School Percentage"]
    df["ScoreRatio"] = df["Admission Test Score"] / df["High School Percentage"]
    df["PercentSquared"] = df["High School Percentage"] ** 2
    df["ScoreSquared"] = df["Admission Test Score"] ** 2

    df = df.drop_duplicates()

    df["Admission Status"] = df["Admission Status"].map({"Accepted":1, "Rejected":0})
    return df


data = cleanData(data)


data.to_csv("output.csv", index=False)

x = data.drop(columns=(["Admission Status"]))
y = data["Admission Status"]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2, random_state=500, stratify=y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=9999))
])

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
print(accuracy_score(y_test, y_pred))




