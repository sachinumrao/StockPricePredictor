import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pandas_datareader as pdr
from fbprophet import Prophet
import datetime
from datetime import timedelta

app = dash.Dash()
app.title = 'Stock Predictor'

external_styles = ['https://codepen.io/chriddyp/pen/bWLwgP.csc']
app = dash.Dash(__name__, external_stylesheets=external_styles)

app.layout = html.Div([
    html.Div([dcc.Input(id='input-box', type='text')], className='row'),
    html.Div([html.Button('Submit', id='button')], className='row'),
    html.Div([
        dcc.Graph(
            id='figure-1'
        )
    ], className='ten columns offset-by-one')
])

@app.callback(
    dash.dependencies.Output('figure-1', 'figure'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')]
)
def update_output(n_clicks, value):
    if value is None:
        value = 'AAPL'

    ticker = value
    forecast_period = 22
    curr_date = datetime.datetime.now()

    df = pdr.get_data_yahoo(ticker)
    df['ds'] = df.index
    df['ds'] = pd.to_datetime(df['ds'], format='%YYYY-%mm-%dd')
    df1 = pd.DataFrame()
    df1[['ds', 'y']] = df[['ds', 'Close']]

    train_period_len = None
    if df1.shape[0] > 252*3 :
        train_period_len = 252*3
    else:
        train_period_len = df1.shape[0]

    df2 = df1.tail(train_period_len)
    
    model = Prophet()
    model.fit(df2)

    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    forecast2 = forecast.tail(forecast_period)

    f_df = forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    raw_df = pd.DataFrame()
    raw_df[['ds', 'yhat']] = df1[['ds', 'y']]
    raw_df['yhat_lower'] = raw_df['yhat']
    raw_df['yhat_upper'] = raw_df['yhat']

    past_window = 126
    a_df = raw_df.tail(past_window)
    df3 = pd.DataFrame()
    df3 = pd.concat([a_df, f_df])

    figure = {
        'data':[
            {
                'x' : df3['ds'],
                'y' : df3['yhat_upper'],
                'type' : 'Line',
                'name' : 'Upper Confidence'
            },
            {
                'x' : df3['ds'],
                'y' : df3['yhat_lower'],
                'type' : 'Line',
                'name' : 'Lower Confidence'
            },
            {
                'x' : df3['ds'],
                'y' : df3['yhat'],
                'type' : 'Line',
                'name' : 'Forecast'
            },
            {
                'x' : a_df['ds'],
                'y' : a_df['yhat'],
                'type' : 'Line',
                'name' : 'Actual'
            }
        ],
        'layout':{
            'title' : ticker + ' Stock price Predictor',
            'xaxis' : dict(
                title='Date',
                titlefont=dict(
                    family = ' Courier New, monospace',
                    size= 20,
                    color = '#7f7f7f'
                )
            ),
            'yaxis' : dict(
                title = 'Stock Price',
                titlefont = dict(
                    family = 'Helvetica, monospace',
                    size = 20,
                    color = '#7f7f7f'
                )
            )
        }
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)