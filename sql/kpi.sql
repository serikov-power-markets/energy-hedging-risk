-- Monthly KPIs of the hedge programme per strategy and delivery month (DuckDB).
-- hedge ratio = hedged MWh / actual MWh; open volume = |actual - hedge|; unit costs in EUR/MWh.
WITH hourly AS (
    SELECT p.strategy, h.ts_utc, date_trunc('month', h.ts_local) AS delivery_month,
           h.actual_mwh, p.hedge_mw, m.price
    FROM book_hourly h
    JOIN position_hourly p USING (ts_utc)
    JOIN market_hourly m USING (ts_utc)
), trade_cost AS (
    SELECT strategy, delivery_month, sum(mwh * price) AS hedge_cost_eur, sum(mwh) AS hedge_mwh
    FROM trade_months
    GROUP BY ALL
)
SELECT h.strategy, h.delivery_month,
       sum(h.actual_mwh)                                         AS actual_mwh,
       sum(h.hedge_mw) / sum(h.actual_mwh)                       AS hedge_ratio,
       sum(abs(h.actual_mwh - h.hedge_mw)) / sum(h.actual_mwh)   AS open_share,
       t.hedge_cost_eur / nullif(t.hedge_mwh, 0)                 AS avg_hedge_price,
       sum(h.actual_mwh * h.price) / sum(h.actual_mwh)           AS load_weighted_spot,
       (coalesce(t.hedge_cost_eur, 0) + sum((h.actual_mwh - h.hedge_mw) * h.price))
           / sum(h.actual_mwh)                                   AS unit_cost
FROM hourly h
LEFT JOIN trade_cost t USING (strategy, delivery_month)
GROUP BY h.strategy, h.delivery_month, t.hedge_cost_eur, t.hedge_mwh
ORDER BY h.strategy, h.delivery_month;
