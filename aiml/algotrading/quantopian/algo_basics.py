from zipline.api import sid

def initialize(context):
    '''
    context = programs states, stored like a python dict
    '''
    context.techs = [sid(24), sid(1900), sid(16841), sid(3766)]
    # context.aapl = sid(24)
    # context.csco = sid(1900)
    # context.amzn = sid(16841)
    # context.ibm = sid(3766)
    
    # schedule_function(open_position, date_rules.week_start(), time_rules.market_open())
    # schedule_function(close_position, date_rules.week_end(), time_rules.market_close(minutes=30))
    
    schedule_function(rebalance, date_rules.every_day(), time_rules.market_open())
    schedule_function(record_vars, date_rules.every_day(), time_rules.market_close())
    
    
def handle_data(context, data):
    '''
    called at the end of each interval (usually 1 minute) specified
    '''

    # get price history per period of call
    # price_history = data.history(
    #     context.techs,
    #     fields='price',
    #     bar_count=5,    # Number of time periods defined in freq returned per call
    #     frequency='1d'
    # )
    # print(price_history)
    
    # Example trade when can trade
    # tgt_pct_list = [0.27, 0.2, 0.53, 0]
    # if data.can_trade(context.techs):
    #     for stock, tgt in zip(context.techs, tgt_pct_list):
    #         order_target_percent(stock, tgt)
    
    # Manual offsetting
    # order_target_percent(context.aapl, 0.27)
    # order_target_percent(context.csco, 0.20)
    # order_target_percent(context.amzn, 0.53)
    
def open_position(context, data):
    order_target_percent(context.techs[0], 0.27)
    
def close_position(context, data):
    order_target_percent(context.techs[0], 0)
    
def rebalance(context, data):
    order_target_percent(context.techs[2], 0.5)
    order_target_percent(context.techs[3], -0.5)

def record_vars(context, data):
    record(amzn_close=data.current(context.techs[2], 'close'))  # parameter in bracket is for labelling, not an actual parameter
    record(ibm_close=data.current(context.techs[3], 'close'))  # parameter in bracket is for labelling, not an actual parameter
    