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

#
# def create_plot():
#     N = 200
#     x = np.linspace(0, 1, N)
#     y = x * 10 + np.random.randn(N)
#     df = pd.DataFrame({'x': x, 'y': y})  # creating a sample dataframe
#     data = [
#         go.Bar(
#             x=df['x'],  # assign x as the dataframe column 'x'
#             y=df['y']
#         )
#     ]
#     graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
#     return graphJSON


def create_plot():
    weather_df = db.select_weather_df()
    make_from_date = lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    weather_dates_str = weather_df['dt']
    weather_dates = list(map(make_from_date, weather_dates_str))
    weather_temp = weather_df['temp']
    x = weather_dates
    y = weather_temp
    df = pd.DataFrame({'x': x, 'y': y})
    weather_data = [
        go.Scatter(
            x=df['x'],
            y=df['y']
        ),
    ]
    weatherGraphJSON = json.dumps(weather_data, cls=plotly.utils.PlotlyJSONEncoder)
    return weatherGraphJSON


def create_purchases_with_weather_plot():
    weather_df = db.select_weather_df()
    make_from_date = lambda d: datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    weather_dates_str = weather_df['dt']
    weather_dates = list(map(make_from_date, weather_dates_str))
    weather_temp = weather_df['temp']

    weather_data = [
        go.Scatter(
            x=weather_dates,
            y=weather_df['temp'],
            name='temperature'
        ),
    ]
    weather_precip_data = [
        go.Scatter(
            x=weather_dates,
            y=weather_df['precip_prob'],
            name='Precipitation probability'
        ),
    ]
    revenue_dt = db.select_revenue_by_day_df()
    data = [
        go.Scatter(
            x=revenue_dt['dt'],
            y=revenue_dt['sum_revenue'],
            name='Total revenue'
        ),
    ]
    weatherGraphJSON = json.dumps(weather_data, cls=plotly.utils.PlotlyJSONEncoder)
    weatherPrecipGraphJSON = json.dumps(weather_precip_data, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON, weatherGraphJSON, weatherPrecipGraphJSON


@app.route('/')
def welcome():
    db.init_connection()
    # db.rebuild_db()
    plot1, weather_plot, weather_precip = create_purchases_with_weather_plot()
    db.close_connection()
    return render_template('index.html', plot=plot1, weather_plot=weather_plot, weather_precip=weather_precip)


@app.route('/customer/main')
def customer_main():
    db.init_connection()
    plot1, weather_plot, _ = create_purchases_with_weather_plot()
    db.close_connection()
    return render_template('main_customer_page.html', plot=plot1, weather_plot=weather_plot)


@app.route('/manager/main')
def manager_main():
    db.init_connection()
    plot1, weather_plot, _ = create_purchases_with_weather_plot()
    db.close_connection()
    return render_template('main_manager_page.html', plot=plot1, weather_plot=weather_plot)


if __name__ == '__main__':
    app.run()
