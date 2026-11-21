import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

df = pd.read_csv('dataset/anime-dataset-2023.csv')
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')
df = df.dropna(subset=['Name', 'Score', 'Popularity'])
df = df[df['Score'] > 0]

original = df.copy()

le_genres = LabelEncoder()
le_type = LabelEncoder()
df['Genres'] = le_genres.fit_transform(df['Genres'].astype(str))
df['Type'] = le_type.fit_transform(df['Type'].astype(str))

x = df[['Genres', 'Type', 'Popularity']]
y = df['Score']
category = pd.cut(y, bins=[0, 5, 7, 10], labels=['Low', 'Average', 'High'])

x_train, x_test, y_train, y_test, c_train, c_test = train_test_split(
    x, y, category, test_size=0.2, random_state=3
)

rf = RandomForestRegressor(n_estimators=100, random_state=3)
rf.fit(x_train, y_train)
print("RF  MAE:", mean_absolute_error(y_test, rf.predict(x_test)))
print("RF  R2 :", r2_score(y_test, rf.predict(x_test)))

dt = DecisionTreeClassifier(random_state=3)
dt.fit(x_train, c_train)
print("DT  Accuracy:", accuracy_score(c_test, dt.predict(x_test)))

genres = [g.strip().lower() for g in input("Genre(s), comma separated: ").split(',')]
a_type = input("Type (TV, Movie, OVA...): ").strip().lower()
min_score = float(input("Minimum score (1-10): "))

m = original[
    (original['Type'].str.lower() == a_type) &
    (original['Score'] >= min_score) &
    original['Genres'].apply(lambda g: all(x in str(g).lower() for x in genres))
].copy()

if m.empty:
    print("No anime found.")
else:
    f = pd.DataFrame({
        'Genres': le_genres.transform(m['Genres'].astype(str)),
        'Type': le_type.transform(m['Type'].astype(str)),
        'Popularity': m['Popularity'].values
    })
    m['Predicted_Score'] = rf.predict(f).round(2)
    m['Predicted_Category'] = dt.predict(f)
    print(m.sort_values('Predicted_Score', ascending=False)
            .head(10)[['Name', 'Score', 'Predicted_Score', 'Predicted_Category']]
            .to_string(index=False))