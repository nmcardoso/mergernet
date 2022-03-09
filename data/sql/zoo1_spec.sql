--------------------------------------------------------------------------------
-- GALAXYZOO 1 WITH SPECTRA
-- CONTEXT: DR17
--------------------------------------------------------------------------------

SELECT
zoo.specobjid, zoo.objid AS dr8objid, zoo.dr7objid, zoo.ra, zoo.dec,
zoo.p_el, zoo.p_cw, zoo.p_acw, zoo.p_edge, zoo.p_dk, zoo.p_mg, zoo.p_cs,
zoo.p_el_debiased, zoo.p_cs_debiased, zoo.spiral, zoo.elliptical, zoo.uncertain,

p.petroMag_g, p.petroMag_u, p.petroMag_r, p.petroMag_i, p.petroMag_z,
p.modelMag_g, p.modelMag_u, p.modelMag_r, p.modelMag_i, p.modelMag_z,
p.run, p.rerun, p.camcol, p.field, p.obj, s.z
FROM zooSpec AS zoo
LEFT JOIN dr7.PhotoObjAll AS p ON zoo.dr7objid = p.objid
LEFT JOIN dr8.SpecObjAll AS s ON zoo.specobjid = s.specObjID
