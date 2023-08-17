SELECT
P.objID AS objID, 
P.targetObjID AS targetObjID, 
P.specObjID AS specObjID, 
P.ra AS ra, 
P.dec AS dec, 
P.modelMag_r AS modelMag_r, 
P.modelMag_g AS modelMag_g, 
P.modelMag_u AS modelMag_u,
P.modelMag_i AS modelMag_i,
P.modelMag_z AS modelMag_z,
P.z AS z,

L.objID AS objID_n, 
L.targetObjID AS targetObjID_n, 
L.specObjID AS specObjID_n, 
L.ra AS ra_n, 
L.dec AS dec_n, 
L.modelMag_r AS modelMag_r_n, 
L.modelMag_g AS modelMag_g_n, 
L.modelMag_u AS modelMag_u_n,
L.modelMag_i AS modelMag_i_n,
L.modelMag_z AS modelMag_z_n,
L.z AS z_n,

N.type AS type,
N.mode AS mode,
N.distance AS separation

FROM SpecPhoto AS P
JOIN Neighbors AS N ON P.targetObjID = N.ObjID
JOIN SpecPhoto AS L ON L.targetObjID = N.NeighborObjID
WHERE
P.targetObjID < L.targetObjID
AND ABS(P.z - L.z) < 0.01
AND P.class = 'GALAXY'
AND L.class = 'GALAXY'
AND P.modelMag_r BETWEEN 15 AND 17
AND N.distance < 0.5
AND ABS(P.modelMag_r - L.modelMag_r) < 1.5
--AND ABS((P.modelMag_u - P.modelMag_g) - (L.modelMag_u - L.modelMag_g)) < 0.075
--AND ABS((P.modelMag_g - P.modelMag_r) - (L.modelMag_g - L.modelMag_r)) < 0.075
--AND ABS((P.modelMag_r - P.modelMag_i) - (L.modelMag_r - L.modelMag_i)) < 0.075
--AND ABS((P.modelMag_i - P.modelMag_z) - (L.modelMag_i - L.modelMag_z)) < 0.075






---------





SELECT
P.objID AS objID, 
P.targetObjID AS targetObjID, 
P.specObjID AS specObjID, 
P.ra AS ra, 
P.dec AS dec, 
P.modelMag_r AS modelMag_r, 
P.modelMag_g AS modelMag_g, 
P.modelMag_u AS modelMag_u,
P.modelMag_i AS modelMag_i,
P.modelMag_z AS modelMag_z,
P.z AS z,

L.objID AS objID_n, 
L.targetObjID AS targetObjID_n, 
L.specObjID AS specObjID_n, 
L.ra AS ra_n, 
L.dec AS dec_n, 
L.modelMag_r AS modelMag_r_n, 
L.modelMag_g AS modelMag_g_n, 
L.modelMag_u AS modelMag_u_n,
L.modelMag_i AS modelMag_i_n,
L.modelMag_z AS modelMag_z_n,
L.z AS z_n,

N.type AS type,
N.mode AS mode,
N.distance AS separation

FROM SpecPhoto AS P
JOIN Neighbors AS N ON P.targetObjID = N.ObjID
JOIN SpecPhoto AS L ON L.targetObjID = N.NeighborObjID
WHERE
P.targetObjID < L.targetObjID
--AND ABS(P.z - L.z) < 0.01
AND P.class = 'GALAXY'
AND L.class = 'STAR'
AND P.modelMag_r BETWEEN 15 AND 17
AND N.distance < 0.3
--AND ABS(P.modelMag_r - L.modelMag_r) < 1.5
--AND ABS((P.modelMag_u - P.modelMag_g) - (L.modelMag_u - L.modelMag_g)) < 0.075
--AND ABS((P.modelMag_g - P.modelMag_r) - (L.modelMag_g - L.modelMag_r)) < 0.075
--AND ABS((P.modelMag_r - P.modelMag_i) - (L.modelMag_r - L.modelMag_i)) < 0.075
--AND ABS((P.modelMag_i - P.modelMag_z) - (L.modelMag_i - L.modelMag_z)) < 0.075





-------



SELECT
G.objID AS objID, 
G.ra AS ra, 
G.dec AS dec, 
G.modelMag_r AS modelMag_r, 
G.modelMag_g AS modelMag_g, 
G.modelMag_u AS modelMag_u,
G.modelMag_i AS modelMag_i,
G.modelMag_z AS modelMag_z,

S.objID AS objID_n,
S.ra AS ra_n, 
S.dec AS dec_n, 
S.modelMag_r AS modelMag_r_n, 
S.modelMag_g AS modelMag_g_n, 
S.modelMag_u AS modelMag_u_n,
S.modelMag_i AS modelMag_i_n,
S.modelMag_z AS modelMag_z_n,

N.type AS type,
N.mode AS mode,
N.distance AS separation

FROM GalaxyTag AS G
JOIN Neighbors AS N ON G.objID = N.ObjID
JOIN StarTag AS S ON S.objID = N.NeighborObjID
WHERE
G.objID < S.objID
--AND ABS(P.z - L.z) < 0.01
--AND P.class = 'GALAXY'
--AND L.class = 'STAR'
AND G.type = 3
AND S.type = 6
AND G.modelMag_r BETWEEN 15 AND 17
AND N.distance < 0.3
