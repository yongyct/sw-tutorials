# Simplistic example, to combine with other quantitative analysis
# See: https://www.quantopian.com/docs/data-reference/sentdex
# See: https://www.quantopian.com/docs/data-reference/psychsignal
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.data.sentdex import sentiment
# from quantopian.pipeline.data.psychsignal import aggregated_twitter_withretweets_stocktwits, stocktwits, twitter_noretweets, twitter_withretweets


def initialize(context):
    schedule_function(rebalance, date_rules.every_day())
    attach_pipeline(make_pipeline(), 'pipeline')

    
def before_trading_start(context, data):
    port = pipeline_output('pipeline')
    context.longs = port[(port['sentiment'] > 2)].index.tolist()
    context.shorts = port[(port['sentiment'] < -0.5)].index.tolist()
    context.longs_weight, context.shorts_weight = compute_weights(context)


def compute_weights(context):
    if len(context.longs) == 0:
        longs_weight = 0
    else:
        longs_weight = 0.5 / len(context.longs)
    if len(context.shorts) == 0:
        shorts_weight = 0
    else:
        shorts_weight = -0.5 / len(context.shorts)
    return longs_weight, shorts_weight
    
    
def make_pipeline():
    dollar_vol = AverageDollarVolume(window_length=20)
    is_liq = dollar_vol.top(1000)
    
    # impact = sentiment.sentiment_signal.latest
    sentiment_score = sentiment.sentiment_signal.latest
    
    return Pipeline(
        columns={
            # 'impact': impact,
            'sentiment': sentiment_score
        },
        screen=is_liq
    )


def rebalance(context, data):
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
