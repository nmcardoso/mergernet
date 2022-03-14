-- SELECT p.ID, p.RA, p.DEC, p.r_auto, model.PROB_GAL, model.CLASS
-- FROM dr3.all_dr3 AS p
-- INNER JOIN dr3.vac_star_galaxy_quasar AS model ON p.ID = model.ID
-- WHERE p.Field LIKE 'HYDRA%' AND model.PROB_GAL > 0.9

-- SELECT TOP 10 p.ID, p.RA, p.DEC, p.r_auto, model.PROB_GAL, model.CLASS,
--   distance(POINT('ICRS', p.RA, p.DEC), POINT('ICRS', model.RA, model.DEC)) AS dist
-- FROM dr3.all_dr3 AS p
-- INNER JOIN dr3.vac_star_galaxy_quasar AS model ON 1=CONTAINS(
--   POINT('ICRS', p.RA, p.DEC),
--   CIRCLE('ICRS', model.RA, model.DEC, 0.00028)
-- )
-- WHERE p.Field LIKE 'HYDRA%' AND model.PROB_GAL > 0.9 AND p.r_auto < 17

SELECT p.ID, p.RA, p.DEC, p.r_auto, model.PROB_GAL
FROM idr3.r_band AS p
INNER JOIN idr3_vacs.star_galaxy_quasar AS model ON p.ID = model.ID
WHERE (p.Field NOT LIKE 'STRIPE%') AND (model.PROB_GAL > 0.8)
  AND (p.r_auto > 13.5) AND (p.r_auto < 17) AND (p.e_r_auto < 0.1)
  AND (model.model_flag = 0) -- AND (p.PhotoFlag_r = 0)
