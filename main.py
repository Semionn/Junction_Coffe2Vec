from flask import Flask, render_template
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json
import logging
from operator import itemgetter
from coffe2vec import CoffeeDB
import datetime

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
db = CoffeeDB()


def create_plot():
    N = 200
    x = np.linspace(0, 1, N)
    y = x * 10 + np.random.randn(N)
    df = pd.DataFrame({'x': x, 'y': y})  # creating a sample dataframe
    data = [
        go.Bar(
            x=df['x'],  # assign x as the dataframe column 'x'
            y=df['y']
        )
    ]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def create_purchases_with_weather_plot():
    weather = db.select_weather()
    make_from_date = lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    weather_dates_str = list(map(itemgetter(db.WEATHER_DATE_IDX), weather))
    weather_dates = list(map(make_from_date, weather_dates_str))
    weather_temp = list(map(itemgetter(db.WEATHER_TEMP_IDX), weather))
    x = weather_dates
    y = weather_temp
    df = pd.DataFrame({'x': x, 'y': y})
    data = [
        go.Scatter(
            x=df['x'],
            y=df['y']
        )
    ]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/')
def welcome():
    bar = create_plot()
    # db.rebuild_db()
    return render_template('index.html', plot=bar)


@app.route('/customer/main')
def customer_main():
    bar = create_purchases_with_weather_plot()
    return render_template('main_customer_page.html', plot=bar)


# @app.route('/')
# def index():
#     bar = create_plot()
#     return render_template('index_figma_test.html', plot=bar)

# @app.route('/')
# def index():
#     bar = create_plot()
#     return render_template('index_figma2_page.html', plot=bar)

if __name__ == '__main__':
    app.run()
