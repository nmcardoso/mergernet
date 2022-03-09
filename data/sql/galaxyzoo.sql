--------------------------------------------------------------------------------
-- ZOO 1 QUERY
--------------------------------------------------------------------------------
SELECT
zoo.*,

sp.petroMag_g, sp.petroMag_u, sp.petroMag_r, sp.petroMag_i, sp.petroMag_z,
sp.modelMag_g, sp.modelMag_u, sp.modelMag_r, sp.modelMag_i, sp.modelMag_z,
sp.z

FROM zooSpec AS zoo
LEFT JOIN SpecPhotoAll AS sp
ON zoo.specobjid = sp.specObjID;



--------------------------------------------------------------------------------
-- ZOO 2 QUERY
--------------------------------------------------------------------------------
SELECT
zoo.specobjid, zoo.dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,

zoo.t04_spiral_a08_spiral_fraction,
zoo.t04_spiral_a08_spiral_weighted_fraction,
zoo.t04_spiral_a08_spiral_debiased,

zoo.t05_bulge_prominence_a12_obvious_fraction,
zoo.t05_bulge_prominence_a12_obvious_weighted_fraction,
zoo.t05_bulge_prominence_a12_obvious_debiased,

zoo.t05_bulge_prominence_a13_dominant_fraction,
zoo.t05_bulge_prominence_a13_dominant_weighted_fraction,
zoo.t05_bulge_prominence_a13_dominant_debiased,

zoo.t08_odd_feature_a21_disturbed_fraction,
zoo.t08_odd_feature_a21_disturbed_weighted_fraction,
zoo.t08_odd_feature_a21_disturbed_debiased,

zoo.t08_odd_feature_a22_irregular_fraction,
zoo.t08_odd_feature_a22_irregular_weighted_fraction,
zoo.t08_odd_feature_a22_irregular_debiased,

zoo.t08_odd_feature_a24_merger_fraction,
zoo.t08_odd_feature_a24_merger_weighted_fraction,
zoo.t08_odd_feature_a24_merger_debiased,

sp.petroMag_g, sp.petroMag_u, sp.petroMag_r, sp.petroMag_i, sp.petroMag_z,
sp.modelMag_g, sp.modelMag_u, sp.modelMag_r, sp.modelMag_i, sp.modelMag_z,
sp.z

FROM zoo2MainSpecz AS zoo
LEFT JOIN SpecPhotoAll AS sp
ON zoo.specobjid = sp.specObjID;



--------------------------------------------------------------------------------
-- ZOO 2 QUERY
--------------------------------------------------------------------------------
SELECT
zoo.specobjid, zoo.dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,

zoo.t04_spiral_a08_spiral_fraction,
zoo.t04_spiral_a08_spiral_weighted_fraction,
zoo.t04_spiral_a08_spiral_debiased,

zoo.t05_bulge_prominence_a12_obvious_fraction,
zoo.t05_bulge_prominence_a12_obvious_weighted_fraction,
zoo.t05_bulge_prominence_a12_obvious_debiased,

zoo.t05_bulge_prominence_a13_dominant_fraction,
zoo.t05_bulge_prominence_a13_dominant_weighted_fraction,
zoo.t05_bulge_prominence_a13_dominant_debiased,

zoo.t08_odd_feature_a21_disturbed_fraction,
zoo.t08_odd_feature_a21_disturbed_weighted_fraction,
zoo.t08_odd_feature_a21_disturbed_debiased,

zoo.t08_odd_feature_a22_irregular_fraction,
zoo.t08_odd_feature_a22_irregular_weighted_fraction,
zoo.t08_odd_feature_a22_irregular_debiased,

zoo.t08_odd_feature_a24_merger_fraction,
zoo.t08_odd_feature_a24_merger_weighted_fraction,
zoo.t08_odd_feature_a24_merger_debiased,

sp.petroMag_u,
sp.petroMag_g,
sp.petroMag_r,
sp.petroMag_i,
sp.petroMag_z,

sp.modelMag_u,
sp.modelMag_g,
sp.modelMag_r,
sp.modelMag_i,
sp.modelMag_z,

sp.run,
sp.rerun,
sp.camcol,
sp.field,
sp.obj,

pdr7.petroMag_u AS dr7_petroMag_u,
pdr7.petroMag_g AS dr7_petroMag_g,
pdr7.petroMag_r AS dr7_petroMag_r,
pdr7.petroMag_i AS dr7_petroMag_i,
pdr7.petroMag_z AS dr7_petroMag_z,

pdr7.modelMag_u AS dr7_modelMag_u,
pdr7.modelMag_g AS dr7_modelMag_g,
pdr7.modelMag_r AS dr7_modelMag_r,
pdr7.modelMag_i AS dr7_modelMag_i,
pdr7.modelMag_z AS dr7_modelMag_z,

pdr7.run AS dr7_run,
pdr7.rerun AS dr7_rerun,
pdr7.camcol AS dr7_camcol,
pdr7.field AS dr7_field,
pdr7.obj AS dr7_obj,

sp.z
FROM zoo2MainSpecz AS zoo INTO mydb.zoo2
LEFT JOIN SpecPhotoAll AS sp ON (zoo.dr8objid = sp.specObjID)
LEFT JOIN PhotoObjDR7 AS pdr7 ON (zoo.dr7objid = pdr7.dr7objid)
-- LEFT JOIN PhotoObj AS p ON (zoo.dr8objid = p.objid)
-- LEFT JOIN PhotoObjDR7 AS pdr7 ON (zoo.dr7objid = pdr7.dr7objid)
-- LEFT JOIN SpecPhotoObj AS s ON (zoo.specobjid = s.specObjID)



--------------------------------------------------------------------------------
-- ZOO1 QUERY
--------------------------------------------------------------------------------
SELECT
zoo.specobjid, zoo.objid AS dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,
zoo.p_el, zoo.p_cw, zoo.p_acw, zoo.p_edge, zoo.p_dk, zoo.p_mg, zoo.p_cs,
zoo.p_el_debiased, zoo.p_cs_debiased, zoo.spiral, zoo.elliptical, zoo.uncertain,

COALESCE(p.petroMag_u, pdr7.petroMag_u) AS petroMag_u,
COALESCE(p.petroMag_g, pdr7.petroMag_g) AS petroMag_g,
COALESCE(p.petroMag_r, pdr7.petroMag_r) AS petroMag_r,
COALESCE(p.petroMag_i, pdr7.petroMag_i) AS petroMag_i,
COALESCE(p.petroMag_z, pdr7.petroMag_z) AS petroMag_z,

COALESCE(p.modelMag_u, pdr7.modelMag_u) AS modelMag_u,
COALESCE(p.modelMag_g, pdr7.modelMag_g) AS modelMag_g,
COALESCE(p.modelMag_r, pdr7.modelMag_r) AS modelMag_r,
COALESCE(p.modelMag_i, pdr7.modelMag_i) AS modelMag_i,
COALESCE(p.modelMag_z, pdr7.modelMag_z) AS modelMag_z,

COALESCE(p.run, pdr7.run) AS run,
COALESCE(p.rerun, pdr7.rerun) AS rerun,
COALESCE(p.camcol, pdr7.camcol) AS camcol,
COALESCE(p.field, pdr7.field) AS field,
COALESCE(p.obj, pdr7.obj) AS obj,

s.z
FROM
zooSpec AS zoo INTO mydb.zoo1
LEFT JOIN PhotoObj AS p ON (zoo.objid = p.objid)
LEFT JOIN PhotoObjDR7 AS pdr7 ON (zoo.dr7objid = pdr7.dr7objid)
LEFT JOIN SpecObj AS s ON (zoo.specobjid = s.specObjID)
















SELECT
zoo.specobjid, zoo.objid AS dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,
zoo.p_el, zoo.p_cw, zoo.p_acw, zoo.p_edge, zoo.p_dk, zoo.p_mg, zoo.p_cs,
zoo.p_el_debiased, zoo.p_cs_debiased, zoo.spiral, zoo.elliptical, zoo.uncertain
FROM
zooSpec AS zoo INTO mydb.zoo1




SELECT
zoo.specobjid, zoo.dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,

zoo.t04_spiral_a08_spiral_fraction,
zoo.t04_spiral_a08_spiral_weighted_fraction,
zoo.t04_spiral_a08_spiral_debiased,

zoo.t05_bulge_prominence_a12_obvious_fraction,
zoo.t05_bulge_prominence_a12_obvious_weighted_fraction,
zoo.t05_bulge_prominence_a12_obvious_debiased,

zoo.t05_bulge_prominence_a13_dominant_fraction,
zoo.t05_bulge_prominence_a13_dominant_weighted_fraction,
zoo.t05_bulge_prominence_a13_dominant_debiased,

zoo.t08_odd_feature_a21_disturbed_fraction,
zoo.t08_odd_feature_a21_disturbed_weighted_fraction,
zoo.t08_odd_feature_a21_disturbed_debiased,

zoo.t08_odd_feature_a22_irregular_fraction,
zoo.t08_odd_feature_a22_irregular_weighted_fraction,
zoo.t08_odd_feature_a22_irregular_debiased,

zoo.t08_odd_feature_a24_merger_fraction,
zoo.t08_odd_feature_a24_merger_weighted_fraction,
zoo.t08_odd_feature_a24_merger_debiased
FROM
zoo2MainSpecz AS zoo INTO mydb.zoo2




SELECT
zoo.specobjid, zoo.objid AS dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,
zoo.p_el, zoo.p_cw, zoo.p_acw, zoo.p_edge, zoo.p_dk, zoo.p_mg, zoo.p_cs,
zoo.p_el_debiased, zoo.p_cs_debiased, zoo.spiral, zoo.elliptical, zoo.uncertain
FROM zooSpec as zoo
UNION
SELECT
zoo.specobjid, zoo.objid AS dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,
zoo.p_el, zoo.p_cw, zoo.p_acw, zoo.p_edge, zoo.p_dk, zoo.p_mg, zoo.p_cs
FROM zooNoSpec as zoo
INTO mydb.union








SELECT
zoo.specobjid, zoo.objid AS dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,
zoo.p_el, zoo.p_cw, zoo.p_acw, zoo.p_edge, zoo.p_dk, zoo.p_mg, zoo.p_cs,
n.z, sp.modelMag_r
FROM zooNoSpec AS zoo
CROSS APPLY dbo.fGetNearestSpecObjEq(zoo.ra, zoo.dec, 0.025) AS n
LEFT JOIN SpecPhotoAll AS sp ON n.specObjID = sp.specObjID
