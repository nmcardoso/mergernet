SELECT FLOOR(mag_r/0.05)*0.05 AS bins, COUNT(*) AS cnt 
FROM ls_dr9.tractor 
WHERE type != 'PSF' AND mag_r > 14 AND mag_r < 20 
GROUP BY 1
ORDER BY 1
