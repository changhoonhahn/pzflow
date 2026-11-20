-- RECONSTRUCTED COMPATIBLE ADQL, NOT RECOVERED ORIGINAL REQUEST PAYLOADS.
-- Endpoint: https://irsa.ipac.caltech.edu/TAP/sync
-- POST REQUEST=doQuery LANG=ADQL FORMAT=csv QUERY=<statement>.
-- Respect service MAXREC/overflow; chunk requests or use an async TAP job.
-- Preserve full source precision and IDs. Current server catalogs may differ
-- from the cached snapshots; compare against sources.json and local ID sets.

-- Broadband inputs and selection fields. Original cached extract: 711918 rows.
SELECT id, ra, dec,
       cfht_u_flux_aper2, cfht_u_fluxerr_aper2,
       hsc_g_flux_aper2, hsc_g_fluxerr_aper2,
       hsc_r_flux_aper2, hsc_r_fluxerr_aper2,
       hsc_i_flux_aper2, hsc_i_fluxerr_aper2,
       hsc_z_flux_aper2, hsc_z_fluxerr_aper2,
       hsc_y_flux_aper2, hsc_y_fluxerr_aper2,
       lp_zpdf, lp_type, flag_combined
FROM cosmos2020_classic
WHERE flag_combined = 0 AND lp_type = 0 AND lp_zpdf > 0;

-- Magnitude, total-aperture offset, and reported interval summaries.
SELECT id, hsc_i_mag_aper2, total_off2, lp_zpdf, lp_zpdf_l68, lp_zpdf_u68
FROM cosmos2020_classic
WHERE flag_combined = 0 AND lp_type = 0 AND lp_zpdf > 0;

-- r-band values for the spectroscopic-overlap diagnostic.
SELECT id, hsc_r_mag_aper2, hsc_r_flux_aper2, hsc_r_fluxerr_aper2
FROM cosmos2020_classic
WHERE flag_combined = 0 AND lp_type = 0 AND lp_zpdf > 0;

-- Safe EBV superset for the eligible parent. The cached EBV extract has 170172
-- rows from a narrower original request that was not retained. A superset is
-- compatible because the EAZY workflow joins by Classic ID, never row number.
SELECT id, ebv_mw
FROM cosmos2020_classic
WHERE flag_combined = 0 AND lp_type = 0 AND lp_zpdf > 0;
