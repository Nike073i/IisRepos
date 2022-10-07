from flask import Flask, request, render_template, url_for
import classification

app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='')


@app.route('/')
def home():
    return render_template('index.html')


def error(message, return_url):
    return render_template('error.html', data=dict(error_message=message, return_url=return_url))


@app.route('/classified-settings')
def classified_settings():
    return render_template('classified_settings.html')


@app.route('/classified', methods=['POST'])
def get_classified():
    try:
        test_size = float(request.form['size'])
        precision = int(request.form['precision'])
    except ValueError as e:
        return error(e, url_for('classified-settings'))

    # Проверка входных условий
    score, feature_importances = classification.estimate(test_size, precision)
    return render_template('classified.html', data=dict(score=score, feature_importances=feature_importances))
