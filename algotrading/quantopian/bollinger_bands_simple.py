# FYI very simple bollinger band strategy, erratic performance, just to understand platform
# Only works in https://www.quantopian.com/algorithms IDE
import numpy as np


def initialize(context):
    
    context.jnj = sid(4151)
    
    schedule_function(check_bands, date_rules.every_day())


def check_bands(context, data):
    
    jnj = context.jnj
    
    # price = prices.iloc[-1]
    price = data.current(jnj, 'price')
    prices = data.history(jnj, 'price', 20, '1d')
    
    # ma20 = np.mean(prices)
    # std20 = np.std(prices)
    ma20 = prices.mean()
    std20 = prices.std()
    
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    
    if price <= bb_lower:
        order_target_percent(jnj, 1)
    elif price >= bb_upper:
        order_target_percent(jnj, -1)
    else:
        pass
        
    record(
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        ma20=ma20,
        price=price
    )
