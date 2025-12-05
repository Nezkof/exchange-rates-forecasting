import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm


class MarkowitzMethod:
    def __init__(self, path, tickers, risk_threshold, rows_num):
        self.path = path
        self.tickers = tickers
        self.risk_threshold = risk_threshold
        self.rows_num = rows_num

    def prepare_data(self):
        self.daily_returns = pd.read_csv(
            self.path, index_col=0, usecols=["DATE"] + self.tickers, nrows=self.rows_num
        )
        self.mean_annual_returns = (1 + self.daily_returns.mean()) ** 252 - 1
        self.cov = self.daily_returns.cov() * 252
        print(self.daily_returns.shape)

    def build_portfolios(self, samples_amount, tickers_amount):
        self.mean_variance_pairs = []
        self.weights_list = []
        self.tickers_list = []

        for i in tqdm(range(samples_amount)):
            next_i = False
            while True:
                assets = np.random.choice(
                    list(self.daily_returns.columns), tickers_amount, replace=False
                )
                weights = np.random.rand(tickers_amount)
                weights = weights / sum(weights)

                portfolio_E_Variance = 0
                portfolio_E_Return = 0
                for i in range(len(assets)):
                    portfolio_E_Return += (
                        weights[i] * self.mean_annual_returns.loc[assets[i]]
                    )
                    for j in range(len(assets)):
                        portfolio_E_Variance += (
                            weights[i] * weights[j] * self.cov.loc[assets[i], assets[j]]
                        )

                for R, V in self.mean_variance_pairs:
                    if (R > portfolio_E_Return) & (V < portfolio_E_Variance):
                        next_i = True
                        break
                if next_i:
                    break

                self.mean_variance_pairs.append(
                    [portfolio_E_Return, portfolio_E_Variance]
                )
                self.weights_list.append(weights)
                self.tickers_list.append(assets)
                break

        self.mean_variance_pairs = np.array(self.mean_variance_pairs)

        self.risk_free_rate = 0

    def visualize_results(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.mean_variance_pairs[:, 1] ** 0.5,
                y=self.mean_variance_pairs[:, 0],
                marker=dict(
                    color=(self.mean_variance_pairs[:, 0] - self.risk_free_rate)
                    / (self.mean_variance_pairs[:, 1] ** 0.5),
                    showscale=True,
                    size=7,
                    line=dict(width=1),
                    colorscale="RdBu",
                    colorbar=dict(title="Sharpe<br>Ratio"),
                ),
                mode="markers",
                text=[
                    str(np.array(self.tickers_list[i]))
                    + "<br>"
                    + str(np.array(self.weights_list[i]).round(2))
                    for i in range(len(self.tickers_list))
                ],
            )
        )
        fig.update_layout(
            template="plotly_white",
            xaxis=dict(title="Очікуваний ризик (Волатильність)"),
            yaxis=dict(title="Очікуваний прибуток"),
            title="Набір портфелів",
            width=850,
            height=500,
        )
        fig.update_xaxes(range=[0.18, 0.35])
        fig.update_yaxes(range=[0.05, 0.29])
        fig.update_layout(coloraxis_colorbar=dict(title="Sharpe Ratio"))
        fig.show()

    def get_portfolios_below_risk(self):
        all_risks = self.mean_variance_pairs[:, 1] ** 0.5
        indices_below_threshold = np.where(all_risks < self.risk_threshold)[0]
        filtered_weights = [self.weights_list[i] for i in indices_below_threshold]
        return filtered_weights

    def get_chart_data(self, capital):
        all_risks_raw = self.mean_variance_pairs[:, 1] ** 0.5
        all_returns_raw = self.mean_variance_pairs[:, 0]

        all_risks = np.nan_to_num(all_risks_raw, nan=0.0, posinf=0.0, neginf=0.0)
        all_returns = np.nan_to_num(all_returns_raw, nan=0.0, posinf=0.0, neginf=0.0)

        all_sharpe_ratios = np.nan_to_num(
            (all_returns - self.risk_free_rate) / all_risks,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        portfolios_data = []

        for i in range(len(self.weights_list)):
            current_risk = float(all_risks[i])
            if current_risk < self.risk_threshold:
                try:
                    tickers = self.tickers_list[i].tolist()
                except AttributeError:
                    tickers = self.tickers_list[i]

                original_weights = np.array(self.weights_list[i])
                capital_amounts = original_weights * capital

                weights_as_capital = capital_amounts.tolist()

                portfolios_data.append(
                    {
                        "risk": current_risk,
                        "return": float(all_returns[i]),
                        "sharpe_ratio": float(all_sharpe_ratios[i]),
                        "tickers": tickers,
                        "distribution": weights_as_capital,
                    }
                )

        min_risk = float(np.min(all_risks)) if len(all_risks) > 0 else 0.0
        max_risk = float(np.max(all_risks)) if len(all_risks) > 0 else 1.0
        min_return = float(np.min(all_returns)) if len(all_returns) > 0 else 0.0
        max_return = float(np.max(all_returns)) if len(all_returns) > 0 else 1.0

        metadata = {
            "x_axis_label": "Очікуваний ризик (Волатильність)",
            "y_axis_label": "Очікуваний прибуток",
            "color_bar_label": "Коефіцієнт шарпа",
            "chart_title": f"Портфелі з ризиком менше за {self.risk_threshold * 100:.0f}%",
            "x_range": [min_risk, max_risk],
            "y_range": [min_return, max_return],
        }

        return {"portfolios": portfolios_data, "metadata": metadata}

    def optimize(self, samples_amount, tickers_amount):
        self.prepare_data()
        self.build_portfolios(samples_amount, tickers_amount)
