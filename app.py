""" Create flask application. """
import io
import base64
import json
import os
import sys

PARENT_FOLDER = str(os.path.dirname(os.path.realpath(__file__)).rsplit('/', 1)[0])
sys.path.insert(0, PARENT_FOLDER)
ROOT_FOLDER = PARENT_FOLDER.rsplit('/', 1)[0]

from flask import Flask, request, render_template
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from linear_regression import LinearReg


matplotlib.use('Agg')
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return "Welcome!"


@app.route('/predict', methods=['GET'])
def predict():
    batch_size = request.args.get('batch_size')
    train_per_cent = request.args.get('train_per_cent')
    lr = LinearReg()
    df = lr.predict(batch_size, train_per_cent)
    return df.to_html()


@app.route('/predict_real_time', methods=['POST'])
def predict_real_time_with_metrics():
    data_list = json.loads(request.data)
    lr = LinearReg()
    df1, df2 = lr.predict_real_time_with_metrics(data_list)
    return render_template('tables.html', tables = [df1.to_html(classes='data'), df2.to_html(classes='data')],
                           titles=['na', 'Actual vs Predicted', 'Metrics'])


@app.route('/plot', methods=['GET'])
def plot():
    batch_size = request.args.get('batch_size')
    train_per_cent = request.args.get('train_per_cent')
    lr = LinearReg()
    lr.predict(batch_size, train_per_cent)
    df = lr.get_actual_v_predicted_df()
    
    fig = df.plot(kind='bar', figsize=(10, 4)).get_figure()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.xlabel('Minimum Temperature')
    plt.ylabel('Maximum Temperature')
    
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('ascii')

    return render_template('image.html', image=pngImageB64String)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
