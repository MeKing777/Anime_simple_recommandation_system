from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
load = lambda f: joblib.load(os.path.join(BASE, 'models', f))

data = pd.read_csv(os.path.join(BASE, 'dataset', 'anime-dataset-2023.csv'))
data['Score'] = pd.to_numeric(data['Score'], errors='coerce')
data['Popularity'] = pd.to_numeric(data['Popularity'], errors='coerce')
data = data.dropna(subset=['Name', 'Genres', 'Type', 'Score', 'Popularity'])

rf_model = load('anime_random_forest.pkl')
dt_model = load('anime_decision_tree.pkl')
le_genres = load('le_genres.pkl')
le_type = load('le_type.pkl')

TYPES = sorted(t for t in data['Type'].unique() if t.strip())
GENRES = sorted({g.strip() for val in data['Genres'] for g in val.split(',') if g.strip()})


@app.route('/')
def home():
    return render_template('index.html', types=TYPES, genres=GENRES)


@app.route('/recommend', methods=['POST'])
def recommend():
    genres = [g.strip() for g in request.form.getlist('genre') if g.strip()]
    anime_type = request.form.get('anime_type', '').strip()

    try:
        min_score = float(request.form.get('min_score', 1))
    except ValueError:
        return "Please enter a valid score.", 400

    if not genres or not anime_type:
        return "Please enter genre and anime type.", 400
    if not 1 <= min_score <= 10:
        return "Score must be between 1 and 10.", 400

    m = data[
        (data['Type'].str.strip().str.lower() == anime_type.lower()) &
        (data['Score'] >= min_score) &
        data['Genres'].apply(lambda s: all(g.lower() in s.lower() for g in genres))
    ].copy()

    recommendations = []
    if not m.empty:
        f = pd.DataFrame({
            'Genres': le_genres.transform(m['Genres']),
            'Type': le_type.transform(m['Type']),
            'Popularity': m['Popularity'].values
        })
        m['Predicted_Score'] = rf_model.predict(f)
        m['Predicted_Category'] = dt_model.predict(f)
        top = m.sort_values('Predicted_Score', ascending=False).head(10)

        recommendations = [{
            'name': r.Name,
            'genre': r.Genres,
            'type': anime_type,
            'actual_score': round(r.Score, 2),
            'predicted_score': round(r.Predicted_Score, 2),
            'category': r.Predicted_Category
        } for r in top.itertuples()]

    return render_template(
        'result.html',
        recommendations=recommendations,
        genre=', '.join(genres),
        anime_type=anime_type,
        min_score=min_score
    )


if __name__ == '__main__':
    app.run(debug=True)