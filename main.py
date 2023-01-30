import pandas as pd
import numpy as np
#import pandas_datareader.data as web
from datetime import date
import matplotlib.pyplot as plt
import scipy.optimize as opt
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
import plotly

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus']=False
pd.set_option('display.max_columns',None)

begin = date(2022,12,31)
end = date.today()
interval = (end-begin).days*24
risk_free_rate = 0.04/365

tnames = [' BTC ','  ETH ',' BNB ','  USDT ',' ADA ','  XRP ',' SOL ','  DOGE ',' SHIB ','  DOT ',' DAI ','  MATIC ',' ATOM ',' UNI ','  APE ']
tickers = ['BTC-USD','ETH-USD','BNB-USD','USDT-USD','ADA-USD','XRP-USD','SOL-USD','DOGE-USD','SHIB-USD','DOT-USD','DAI-USD','MATIC-USD','ATOM-USD','UNI-USD','APE-USD']
df = pd.DataFrame()


'''
for i in tickers:
    df[i] = yf.download(i,start = '2022-12-31',end = date.today(), interval = '1h',)['Adj Close']
data = np.log(df/df.shift(1))
df_dc = df/df.shift(1)-1
data.to_csv('Database.csv')
data_corr = data.corr()'''


#main code

data = pd.read_csv('Database.csv')
returns_annual = data.mean(numeric_only=True)*interval #calculate mean return
cov_annual = data.cov()*interval #covariance matrix
number_of_assets = len(data.columns)-1


weight = []
optimal_search = []
here_to_find_ratio = []
weights = []
for stock in range(1000000): #MCS
    next_i = False
    while True:
        weight = np.random.random(number_of_assets)
        weight = np.round(weight / (np.sum(weight)),4)
        returns = np.round(np.dot(weight, returns_annual),4)
        volatility = np.round(np.sqrt(np.dot(weight.T, np.dot(cov_annual, weight))),4)  # portfolio volatility in interval
        sharpe = (returns - risk_free_rate) / volatility

        for re,vo in optimal_search:
            if (re > returns) & (vo < volatility):
                next_i = True
                break
        if next_i:
            break
        here_to_find_ratio.append([returns, volatility, sharpe, weight,tnames])
        weights.append(weight)
        optimal_search.append([returns,volatility])
        #weight.append(weight)
here_to_find_ratio = pd.DataFrame(here_to_find_ratio, columns = ['returns','volatilities','SPI','weights','assets'])

fig = px.scatter(here_to_find_ratio, x = 'volatilities', y = 'returns', color = 'SPI', hover_data = ['assets','weights'], title = 'Efficient Frontier')
def save_fig(fig, name = 'output.html'):
    html = fig.to_html()
    with open(name, 'w') as f:
        f.write(html)
    print(f'successful html output:{name}')
save_fig(fig,'efficient_frontier.html')
fig.show()

