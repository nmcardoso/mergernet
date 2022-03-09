--------------------------------------------------------------------------------
-- GALAXYZOO 2 (MAIN) WITH SPECTRA
-- CONTEXT: DR17
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

p.petroMag_g, p.petroMag_u, p.petroMag_r, p.petroMag_i, p.petroMag_z,
p.modelMag_g, p.modelMag_u, p.modelMag_r, p.modelMag_i, p.modelMag_z,
p.run, p.rerun, p.camcol, p.field, p.obj, s.z
FROM zoo2MainSpecz AS zoo
LEFT JOIN dr7.PhotoObjAll AS p ON zoo.dr7objid = p.objid
LEFT JOIN dr8.SpecObjAll AS s ON zoo.specobjid = s.specObjID



--------------------------------------------------------------------------------
--
--------------------------------------------------------------------------------




SELECT
n.specObjID, zoo.dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,

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

p.petroMag_g, p.petroMag_u, p.petroMag_r, p.petroMag_i, p.petroMag_z,
p.modelMag_g, p.modelMag_u, p.modelMag_r, p.modelMag_i, p.modelMag_z,
p.run, p.rerun, p.camcol, p.field, p.obj, n.z
FROM zoo2MainPhotoz AS zoo
LEFT JOIN dr7.PhotoObjAll AS p ON zoo.dr7objid = p.objid
CROSS APPLY dbo.fGetNearestSpecObjEq(zoo.ra, zoo.dec, 0.025) AS n
