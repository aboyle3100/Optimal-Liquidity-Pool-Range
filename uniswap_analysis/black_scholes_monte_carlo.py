import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import math

class Simulator():
    def __init__(self, S0, mu, sigma, T):
        """_summary_

        Args:
            S0 (_type_): _description_
            mu (_type_): _description_
            sigma (_type_): _description_
            T (_type_): _description_
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.data = []
        pass
    
    def standard_monte_carlo(self, N, M):
        """Returns M sample stock price paths for a black scholes monte carlo simulation

        Args:
            S0 (float/int): initial stock price
            mu (float): drift term mean change in stock price per time step
            sigma (float): stock price volatility
            T (int): maturity date days  
            N (int): number of time steps
            M (int): number of simulations

        Returns:
            float array: stock price for M sample paths until maturity date
        """
        
        dt = self.T / N
        S = np.zeros((M,N+1))
        S[:,0] = self.S0
        for t in range(1, N+1):
            # draw M random standard normal values 
            Z = np.random.standard_normal(M)
            # compute the simulated stock prices at time t based on previous price
            S[:,t] = S[:,t-1] * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma*np.sqrt(dt)*Z)
            
        return S
    
    
    def set_initial_price(self, new_S0):
        self.S0 = new_S0
    
    def get_security_data(self, ticker, start='2010-01-01', end='2023-01-01'):
        data = yf.download(ticker, start=start, end=end)
        # log returns
        data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # Drop the first row with NaN return
        data = data.dropna()
        
        return data
    
    def markov_switching_monte_carlo(self,N, M, results):
        """_summary_

        Args:
            N (_type_): _description_
            M (_type_): _description_
            results (_type_): _description_

        Returns:
            _type_: _description_
        """
        dt = self.T / N
        
        num_regimes = results.k_regimes
        start = math.factorial(num_regimes)
        
        mu = results.params[start:start+num_regimes]
        # sigma = np.sqrt(results.params[results.k_regimes:2*results.k_regimes])
        raw_sigma = np.clip(results.params[start+num_regimes:start + 2*num_regimes], 1e-6, None)
        sigma = np.sqrt(raw_sigma)
        P = results.model.regime_transition_matrix(results.params).squeeze()
        
        P = P.T
        
        S = np.zeros((M,N+1))
        S[:,0] = self.S0
        
        for i in range(M):
            regime_path = [np.random.choice(results.k_regimes)]# initial regime
        
            for t in range(1,N+1):
                
                
                current_regime = regime_path[-1]
                next_regime = np.random.choice(results.k_regimes, p=P[current_regime])
                regime_path.append(next_regime)
                
            for t in range(1,N+1):
                z = np.random.standard_normal()
                reg = regime_path[t]
                
                if np.isnan(S[i, t-1]) or np.isnan(mu[reg]) or np.isnan(sigma[reg]):
                    print(f"NaN at i={i}, t={t}, reg={reg}, S_prev={S[i, t-1]}, mu={mu[reg]}, sigma={sigma[reg]}")
                    break                
                
                S[i, t] = S[i, t-1] * np.exp((mu[reg] - 0.5 * sigma[reg]**2) * dt + sigma[reg] * np.sqrt(dt) * z)
        return S
    
    def estimate_apy_for_range(self, monte_data, lower, upper, fee_rate=0.003, volume=1e6, initial_liquidity=1e6):
        """
        Estimate the annualized APY for a given price range from simulated paths.

        Args:
            monte_data (ndarray): Simulated price paths, shape (M, N+1)
            lower (float): Lower bound of pool price range
            upper (float): Upper bound of pool price range
            fee_rate (float): Uniswap fee tier (e.g., 0.003 for 0.3%)
            volume (float): Assumed daily volume in the pool
            initial_liquidity (float): Value of capital provided at the start

        Returns:
            float: Estimated annualized APY (as a decimal, e.g., 0.08 = 8%)
        """
        M, N_plus_1 = monte_data.shape
        days = N_plus_1 - 1  # time steps ≈ daily
        in_range = np.logical_and(monte_data >= lower, monte_data <= upper)
        time_in_range_fraction = in_range.sum(axis=1) / days

        avg_time_in_range = time_in_range_fraction.mean()
        expected_fees = avg_time_in_range * volume * fee_rate * days

        final_prices = monte_data[:, -1]
        initial_price = monte_data[:, 0].mean()
        price_ratio = final_prices / initial_price
        IL = 1 - (2 * np.sqrt(price_ratio) / (1 + price_ratio))
        expected_IL = IL.mean()

        net_gain = expected_fees - (expected_IL * initial_liquidity)
        apy = net_gain / initial_liquidity *10

        return apy
    
    def markov_switching_model(self, data, num_regimes=2):
        """Generates a fitted markov regime change model  

        Args:
            ticker (string): ticker representation of a stock findable on yahoo finance
            num_regimes (int, optional): number of volatilities to use. Defaults to 2.
            start (str, optional): start date to pull data from. Defaults to '2010-01-01'.
            end (str, optional): end date to pull stock data from. Defaults to '2023-01-01'.

        Returns:
            MarkovRegression: fitted markov chain regression model
        """
            
        model = MarkovRegression(data['Returns'], k_regimes=num_regimes, switching_variance=True)
        results = model.fit()
        return results
        
    def plot_data(self, data,title='S&P 500 Log Returns'):
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Returns'], label='Log Returns')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.legend()
        plt.show()
    
    def compute_payoff(self,K,S,option="call"):
        """Computes the payoff value for a stock price at maturity date

        Args:
            K (float): strike price of the option 
            S (float): stock price at maturity date
            option (str, optional): Put or call option. Defaults to "call".

        Returns:
            float: payoff of one stock price at maturity date
        """
        if option == "call":
            return np.maximum(S-K, 0)
        elif option == "put":
            return np.maximum(K-S, 0)
        else: 
            return None
    
    

    def find_optimal_pool_range(self, monte_data, fee_rate=0.003, volume=1e6, width_choices=None):
        """
        Optimize pool range by balancing fee income and impermanent loss.

        Args:
            monte_data (ndarray): Simulated price paths, shape (M, N+1)
            fee_rate (float): Uniswap fee tier, e.g., 0.003 for 0.3%
            volume (float): Assumed daily volume for LP share
            width_choices (list of float): Widths as % around final price to evaluate

        Returns:
            tuple: (optimal_lower_bound, optimal_upper_bound, best_net_utility)
        """
        if width_choices is None:
            width_choices = np.linspace(0.05, 0.50, 10)  # e.g., 5% to 50%

        final_prices = monte_data[:, -1]
        initial_price = monte_data[:, 0].mean()

        best_range = None
        best_utility = -np.inf

        for w in width_choices:
            lower = final_prices.mean() * (1 - w)
            upper = final_prices.mean() * (1 + w)

            in_range = np.logical_and(monte_data >= lower, monte_data <= upper)
            time_in_range_fraction = in_range.sum(axis=1) / monte_data.shape[1]

            avg_time_in_range = time_in_range_fraction.mean()
            expected_fees = avg_time_in_range * volume * fee_rate

            # Impermanent loss formula vs HODL
            price_ratio = final_prices / initial_price
            IL = 1 - (2 * np.sqrt(price_ratio) / (1 + price_ratio))
            expected_IL = IL.mean()

            utility = expected_fees - expected_IL * volume  # Adjust scale if needed

            if utility > best_utility:
                best_utility = utility
                best_range = (lower, upper)

        return (*best_range, best_utility)    
        
    def plot_paths(self,S):
        time_steps = np.arange(S.shape[1])
        plt.figure(figsize=(10, 6))

        for i in range(S.shape[0]):
            if np.isnan(S[i, :]).any():
                print(f"NaN in path {i}, skipping.")
                continue
            plt.plot(time_steps, S[i, :], lw=0.8, alpha=0.6)

        plt.xlabel("Time Step")
        plt.ylabel("Stock Price")
        plt.title("Monte Carlo Simulation with Markov Switching")
        plt.grid(True)
        plt.show()
        

        
# S0 = 100  # Initial stock price
# mu = 0.05  # Drift
# sigma = 0.2  # Volatility
# T = 1  # Time to maturity (in years)
# N = 252  # Number of time steps
# M = 10000  # Number of simulations

# sim = Simulator(S0,mu, sigma, T)
# sample_paths = sim.standard_monte_carlo(N,M)
# sim.plot_paths(sample_paths)=