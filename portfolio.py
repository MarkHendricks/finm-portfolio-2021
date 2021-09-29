import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

def tangency_weights(returns,dropna=True,scale_cov=1):
    if dropna:
        returns = returns.dropna()

    covmat_full = returns.cov()
    covmat_diag = np.diag(np.diag(covmat_full))
    covmat = scale_cov * covmat_full + (1-scale_cov) * covmat_diag

    weights = np.linalg.solve(covmat,returns.mean())
    weights = weights / weights.sum()

    return pd.DataFrame(weights, index=returns.columns)


def performanceMetrics(returns,annualization=1, quantile=.05):
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Mean'] = returns.mean() * annualization
    metrics['Vol'] = returns.std() * np.sqrt(annualization)
    metrics['Sharpe'] = (returns.mean() / returns.std()) * np.sqrt(annualization)

    metrics['Min'] = returns.min()
    metrics['Max'] = returns.max()

    metrics[f'VaR ({quantile})'] = returns.quantile(quantile)
    metrics[f'CVaR ({quantile})'] = (returns[returns < returns.quantile(quantile)]).mean()

    return metrics



def maximumDrawdown(returns):
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max

    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()
    summary = pd.DataFrame({'Max Drawdown': max_drawdown, 'Bottom': end_date})

    # The rest of this code is to get the peak and Recover dates.
    # It is tedious, and I recommend skipping the rest of this code unless you are
    # already comfortable with Python and Pandas.

    # get the date at which the return recovers to previous high after the drawdown
    summary['Recover'] = None
    for col in returns.columns:
        idx = returns.index[(returns.index >= end_date[col]).argmax()]
        check_recover = (cum_returns.loc[idx:, col] > rolling_max.loc[idx, col])
        if check_recover.any():
            summary.loc[col, 'Recover'] = check_recover.idxmax()
    summary['Recover'] = pd.to_datetime(summary['Recover'])

    # get the date at which the return peaks before entering the max drawdown
    summary.insert(loc=1, column='Peak', value=0)
    for col in returns.columns:
        df = rolling_max.copy()[[col]]
        df.columns = ['max']
        df['max date'] = df.index
        df = df.merge(df.groupby('max')[['max date']].first().reset_index(), on='max')
        df.rename(columns={'max date_y': 'max date', 'max date_x': 'date'}, inplace=True)
        df.set_index('date', inplace=True)

        summary.loc[col, 'Peak'] = df.loc[end_date[col], 'max date']

    summary['Peak'] = pd.to_datetime(summary['Peak'])
    summary['Peak to Recover'] = (summary['Recover'] - summary['Peak'])

    return summary



def get_ols_metrics(regressors, targets, annualization=1):
    # ensure regressors and targets are pandas dataframes, as expected
    if not isinstance(regressors, pd.DataFrame):
        regressors = regressors.to_frame()
    if not isinstance(targets, pd.DataFrame):
        targets = targets.to_frame()

    # align the targets and regressors on the same dates
    df_aligned = targets.join(regressors, how='inner', lsuffix='y ')
    Y = df_aligned[targets.columns]
    X = df_aligned[regressors.columns]

    reg = pd.DataFrame(index=targets.columns)
    for col in Y.columns:
        y = Y[col]
        model = LinearRegression().fit(X, y)
        reg.loc[col, 'alpha'] = model.intercept_ * annualization
        reg.loc[col, regressors.columns] = model.coef_
        reg.loc[col, 'r-squared'] = model.score(X, y)

        # sklearn does not return the residuals, so we need to build them
        yfit = model.predict(X)
        residuals = y - yfit

        # if intercept =0, numerical roundoff will nonetheless show nonzero Info Ratio
        num_roundoff = 1e-12
        if np.abs(model.intercept_) < num_roundoff:
            reg.loc[col, 'Info Ratio'] = None
        else:
            reg.loc[col, 'Info Ratio'] = (model.intercept_ / residuals.std()) * np.sqrt(annualization)

    return reg


def forecasts_performance(forecasts, target, adjust_vol=True):
    # Function assumes that the date on the forecasts refers to the date of the period
    # being forecast--not the date it is constructed. So a forecast value with date stamp
    # of 2021-06-30 is a forecast constructed the previous period.

    # function to translate forecasts to weights
    wts = forecasts * 100
    # by taking position proportional to forecast, we go longer when more bull-ish
    # and less long (or even short) when more bear-ish.

    # simulate strategy returns
    strategy = wts * target.values
    strategy.dropna(inplace=True)

    # for comparability, rescale the strategy to have same vol as passive
    if adjust_vol:
        sigma = target.std().values[0]
        strategy /= strategy.std()
        strategy *= sigma

    return strategy