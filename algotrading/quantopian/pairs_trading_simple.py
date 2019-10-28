# FYI Poor performing overall strategy, just for tutorial
# Only works in https://www.quantopian.com/algorithms IDE
import numpy as np


# Initialize
def initialize(context):
    schedule_function(check_pairs, date_rules.every_day(), time_rules.market_close(minutes=60))
    
    context.aa = sid(45971)
    context.ua = sid(28051)
    
    context.long_on_spread = False
    context.shorting_spread = False

    
# check_pairs
def check_pairs(context, data):
    
    aa = context.aa
    ua = context.ua
    
    prices = data.history([aa, ua], 'price', 30, '1d')
    
    short_prices = prices.iloc[-1:]
    ma1 = np.mean(short_prices[aa] - short_prices[ua])
    ma30 = np.mean(prices[aa] - prices[ua])
    std30 = np.std(prices[aa] - prices[ua])
    
    if std30 > 0:
        zscore = (ma1 - ma30) / std30
        
        # since we are taking AA's perspective:
        # high zscore = overvalued AA / undervalued UA
        if zscore > 1.0 and not context.shorting_spread:
            order_target_percent(aa, -0.5)
            order_target_percent(ua, 0.5)
            context.shorting_spread = True
            context.long_on_spread = False
        # low zscore = undervalued AA / overvalued UA
        elif zscore < -1.0 and not context.long_on_spread:
            order_target_percent(aa, 0.5)
            order_target_percent(ua, -0.5)
            context.shorting_spread = False
            context.long_on_spread = True
        elif abs(zscore) < 0.1:
            order_target_percent(aa, 0)
            order_target_percent(ua, 0)
            context.shorting_spread = False
            context.long_on_spread = False
            
        record(zscore=zscore)
