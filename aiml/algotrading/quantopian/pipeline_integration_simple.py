from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import AverageDollarVolume, SimpleMovingAverage
from quantopian.pipeline.filters.morningstar import Q1500US


def initialize(context):
    
    schedule_function(rebalance, date_rules.week_start(), time_rules.market_close(hours=1))
    
    my_pipeline = make_pipeline()
    attach_pipeline(my_pipeline, 'my_pipeline')

    
def before_trading_start(context, data):
    context.output = pipeline_output('my_pipeline')
    context.longs = context.output[context.output['Longs']].index.tolist()
    context.shorts = context.output[context.output['Shorts']].index.tolist()
    context.longs_weight, context.shorts_weight = compute_weights(context)
    

def rebalance(context, data):
    # check securities in current portfolio if exist in computed longs/shorts in before_trading_starts
    for security in context.portfolio.positions:
        if (
            security not in context.longs 
            and security not in context.shorts 
            and data.can_trade(security)
        ):
            order_target_percent(security, 0)
    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(security, context.longs_weight)
    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(security, context.shorts_weight)


def compute_weights(context):
    if len(context.longs) == 0:
        longs_weight = 0
    else:
        longs_weight = 0.5 / len(context.longs)
    if len(context.shorts) == 0:
        shorts_weight = 0
    else:
        shorts_weight = 0.5 / len(context.shorts)
    return longs_weight, shorts_weight

    
def make_pipeline():
    
    # PREPARE MASKS
    ## CATEGORICAL MASKS
    universe_filter = Q1500US()
    sector_filter = morningstar.asset_classification.morningstar_sector_code.latest.eq(309)  # 309 is energy sector
    # exchange_filter = morningstar.share_class_reference.exchange_id.latest.eq('NYS')
    base_filter = (
        universe_filter
        & sector_filter
        # & exchange_filter
    )
    ## NUMERIC/FACTOR MASKS
    volume_ma30 = AverageDollarVolume(window_length=30)
    volume_ma30_filter = volume_ma30.percentile_between(95, 100)
    # mkt_cap_filter = morningstar.Fundamentals.market_cap.latest < 100000000
    base_filter = (
        base_filter
        & volume_ma30_filter
        # & mkt_cap_filter
    )
    close_ma30 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=30, mask=base_filter)
    close_ma10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10, mask=base_filter)
    pct_diff = (close_ma10 - close_ma30) / close_ma30
    shorts_filter = pct_diff < 0
    longs_filter = pct_diff > 0
    base_filter = (
        base_filter
        & (shorts_filter | longs_filter)
    )
    
    return Pipeline(
        columns={
            'Close MA30': close_ma30, 
            'Pct Diff': pct_diff,
            'Longs': longs_filter,
            'Shorts': shorts_filter
        },
        screen=base_filter
    )
