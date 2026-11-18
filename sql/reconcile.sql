-- Reconciliation: hourly position rebuilt from the trade blotter in SQL must equal the position
-- computed in Python (position_hourly). Returns the rows that differ by more than 1 kW.
WITH blotter_hours AS (
    SELECT t.strategy, m.ts_utc, sum(t.mw) AS mw
    FROM trades t
    JOIN market_hourly m
      ON m.ts_local >= t.delivery_start
     AND m.ts_local <  t.delivery_end + INTERVAL 1 DAY
     AND (t.kind = 'base' OR m.is_peak)
    GROUP BY ALL
)
SELECT p.strategy, p.ts_utc, p.hedge_mw, coalesce(b.mw, 0) AS blotter_mw
FROM position_hourly p
LEFT JOIN blotter_hours b USING (strategy, ts_utc)
WHERE abs(p.hedge_mw - coalesce(b.mw, 0)) > 0.001
ORDER BY p.strategy, p.ts_utc;
