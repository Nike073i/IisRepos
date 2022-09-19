from flask import Flask, request, render_template, redirect, url_for
import classification
app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/error')
def error():
    return render_template('error.html')


@app.route('/get_classified')
def get_classified():
    try:
        test_size = float(request.args.get('size'))
        precision = int(request.args.get('precision'))
    except ValueError:
        return redirect(url_for('error'))

    # Проверка входных условий
    score, feature_importances = classification.estimate(test_size, precision)
    return render_template('classified.html', data=dict(score=score, feature_importances=feature_importances))
