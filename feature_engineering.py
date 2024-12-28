import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FeatureEngineer:
    def __init__(self, df, date_col, target_col):
        """
        Initialize the FeatureEngineer class.
        
        Args:
            df (pd.DataFrame): Input data frame containing time series data.
            date_col (str): Name of the column containing date information.
            target_col (str): Name of the target column (e.g., revenue).
        """
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.df.sort_values(by=self.date_col, inplace=True)

    def add_date_features(self):
        """Add date-based features like day of week, month, and year."""
        self.df['DayOfWeek'] = self.df[self.date_col].dt.dayofweek
        self.df['Month'] = self.df[self.date_col].dt.month
        self.df['Year'] = self.df[self.date_col].dt.year
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        print("Date-based features added.")
        return self.df

    def add_rolling_features(self, window=7):
        """
        Add rolling aggregates such as moving averages and standard deviations.
        
        Args:
            window (int): The size of the rolling window.
        """
        self.df[f'{self.target_col}_RollingMean_{window}'] = self.df[self.target_col].rolling(window).mean()
        self.df[f'{self.target_col}_RollingStd_{window}'] = self.df[self.target_col].rolling(window).std()
        print(f"Rolling features with window {window} added.")
        return self.df

    def add_lag_features(self, lags=[1, 7, 30]):
        """
        Add lagged features based on the target variable.
        
        Args:
            lags (list): List of lag periods to create features for.
        """
        for lag in lags:
            self.df[f'{self.target_col}_Lag_{lag}'] = self.df[self.target_col].shift(lag)
        print(f"Lag features for lags {lags} added.")
        return self.df

    def add_differencing(self, periods=1):
        """
        Add differenced features to make the series stationary.
        
        Args:
            periods (int): Number of periods for differencing.
        """
        self.df[f'{self.target_col}_Diff_{periods}'] = self.df[self.target_col].diff(periods)
        print(f"Differencing with period {periods} added.")
        return self.df

    def preprocess(self, rolling_window=7, lags=[1, 7, 30], differencing_periods=1):
        """
        Execute all feature engineering steps in sequence.
        
        Args:
            rolling_window (int): Window size for rolling features.
            lags (list): List of lag periods to create lag features for.
            differencing_periods (int): Number of periods for differencing.
        """
        self.add_date_features()
        self.add_rolling_features(window=rolling_window)
        self.add_lag_features(lags=lags)
        self.add_differencing(periods=differencing_periods)
        print("Preprocessing complete.")
        return self.df

class SpikeNormalizer:
    def __init__(self, df, column):
        """
        Initialize the SpikeNormalizer class.

        Args:
            df (pd.DataFrame): Input DataFrame containing time series data.
            column (str): Name of the column with spikes to normalize.
        """
        self.df = df.copy()
        self.column = column

    def winsorize(self, lower_quantile=0.05, upper_quantile=0.95):
        """Normalize spikes using Winsorization."""
        lower_limit = self.df[self.column].quantile(lower_quantile)
        upper_limit = self.df[self.column].quantile(upper_quantile)
        self.df['Winsorized'] = self.df[self.column].clip(lower=lower_limit, upper=upper_limit)
        return self.df['Winsorized']

    def log_transform(self):
        """Normalize spikes using log transformation."""
        self.df['LogTransformed'] = np.log1p(self.df[self.column])  # log(1 + x) for stability
        return self.df['LogTransformed']

    def smooth(self, window=7):
        """Smooth spikes using rolling average."""
        self.df['Smoothed'] = self.df[self.column].rolling(window=window, min_periods=1).mean()
        return self.df['Smoothed']

    def z_score_normalize(self):
        """Normalize spikes using Z-score normalization."""
        mean = self.df[self.column].mean()
        std = self.df[self.column].std()
        self.df['ZScore'] = (self.df[self.column] - mean) / std
        return self.df['ZScore']

    def impute_spikes(self, lower_bound=None, upper_bound=None):
        """Impute spikes by replacing them with interpolated values."""
        if lower_bound is None:
            lower_bound = self.df[self.column].quantile(0.05)
        if upper_bound is None:
            upper_bound = self.df[self.column].quantile(0.95)
        
        is_outlier = (self.df[self.column] < lower_bound) | (self.df[self.column] > upper_bound)
        self.df['Imputed'] = self.df[self.column].where(~is_outlier, np.nan).interpolate()
        return self.df['Imputed']

    def normalize_all(self, lower_quantile=0.05, upper_quantile=0.95, smooth_window=7):
        """
        Apply all normalization methods and store results in the DataFrame.
        """
        self.winsorize(lower_quantile, upper_quantile)
        self.log_transform()
        self.smooth(smooth_window)
        self.z_score_normalize()
        self.impute_spikes()
        print("All normalization methods applied.")

    def visualize_results(self):
        """Visualize the original data and all normalized results."""
        plt.figure(figsize=(14, 8))
        plt.plot(self.df[self.column], label='Original', color='black', linestyle='--', alpha=0.7)
        
        if 'Winsorized' in self.df:
            plt.plot(self.df['Winsorized'], label='Winsorized')
        if 'LogTransformed' in self.df:
            plt.plot(self.df['LogTransformed'], label='Log Transformed')
        if 'Smoothed' in self.df:
            plt.plot(self.df['Smoothed'], label='Smoothed')
        if 'ZScore' in self.df:
            plt.plot(self.df['ZScore'], label='Z-Score Normalized')
        if 'Imputed' in self.df:
            plt.plot(self.df['Imputed'], label='Imputed')

        plt.title('Spike Normalization Methods Comparison')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
