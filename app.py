from flask import Flask, render_template, request
from algorithms import TSP
import os
import glob
import matplotlib.pyplot as plt

IMG_FOLDER = os.path.join('static', 'graphs')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        files = glob.glob('static/graphs/*')
        for f in files:
            os.remove(f)
        n = int(request.form['num'])

        if request.form.get('circular') and not request.form.get('3d'):
            graph = TSP(n, circular=True)
        elif not request.form.get('circular') and request.form.get('3d'):
            graph = TSP(n, three_dim=True)
        elif request.form.get('circular') and request.form.get('3d'):
            graph = TSP(n, circular=True, three_dim=True)
        else:
            graph = TSP(n)

        if request.form.get('random'):
            graph.draw('Random')

        if request.form.get('greedy'):
            graph.use_greedy()
            graph.draw('Greedy')

        if request.form.get('2-approx'):
            graph.use_2_approximation()
            graph.draw('2-Approximation')

        if request.form.get('mst'):
            graph.use_mst()
            graph.draw('Minimal Spanning Tree')

        files = glob.glob('static/graphs/*')
        return render_template('index.html', images=files)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
