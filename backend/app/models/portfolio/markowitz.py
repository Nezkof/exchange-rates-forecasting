import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

class MarkowitzMethod:
   def __init__(self, path, tickers):
      self.path = path
      self.tickers = tickers

   def prepare_data(self):
      self.daily_returns = pd.read_csv(self.path, index_col=0, usecols=["DATE"] + self.tickers)
      self.mean_annual_returns = (1 + self.daily_returns.mean())**252 - 1
      self.cov = self.daily_returns.cov()*252

   def build_portfolios(self, samples_amount, tickers_amount):
      self.mean_variance_pairs = []
      self.weights_list = []
      self.tickers_list = []

      for i in tqdm(range(samples_amount)):
         next_i = False
         while True:
            assets = np.random.choice(list(self.daily_returns.columns), tickers_amount, replace=False)
            weights = np.random.rand(tickers_amount)
            weights = weights/sum(weights)

            portfolio_E_Variance = 0
            portfolio_E_Return = 0
            for i in range(len(assets)):
                  portfolio_E_Return += weights[i] * self.mean_annual_returns.loc[assets[i]]
                  for j in range(len(assets)):
                     portfolio_E_Variance += weights[i] * weights[j] * self.cov.loc[assets[i], assets[j]]

            for R,V in self.mean_variance_pairs:
                  if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                     next_i = True
                     break
            if next_i:
                  break

            self.mean_variance_pairs.append([portfolio_E_Return, portfolio_E_Variance])
            self.weights_list.append(weights)
            self.tickers_list.append(assets)
            break

      self.mean_variance_pairs = np.array(self.mean_variance_pairs)

      self.risk_free_rate = 0

   def visualize_results(self):
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=self.mean_variance_pairs[:,1]**0.5, y=self.mean_variance_pairs[:,0], 
                           marker=dict(color=(self.mean_variance_pairs[:,0]-self.risk_free_rate)/(self.mean_variance_pairs[:,1]**0.5), 
                                       showscale=True, 
                                       size=7,
                                       line=dict(width=1),
                                       colorscale="RdBu",
                                       colorbar=dict(title="Sharpe<br>Ratio")
                                       ), 
                           mode='markers',
                           text=[str(np.array(self.tickers_list[i])) + "<br>" + str(np.array(self.weights_list[i]).round(2)) for i in range(len(self.tickers_list))]))
      fig.update_layout(template='plotly_white',
                        xaxis=dict(title='Annualised Risk (Volatility)'),
                        yaxis=dict(title='Annualised Return'),
                        title='Sample of Random Portfolios',
                        width=850,
                        height=500)
      fig.update_xaxes(range=[0.18, 0.35])
      fig.update_yaxes(range=[0.05,0.29])
      fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
      fig.show()

   def optimize(self, samples_amount, tickers_amount):
      self.prepare_data()
      self.build_portfolios(samples_amount, tickers_amount)
      self.visualize_results()




