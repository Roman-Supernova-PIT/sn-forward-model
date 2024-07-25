"""
Query RomanDESCSims Rubin data processing on NERSC. 
Requires LSST Science Pipelines to query images and get data Ids
Goal to get filepaths to feed to rest of forward-modeling package
which doesn not require Sicence Pipelines.
"""
from lsst.daf.butler import Butler, Timespan
from lsst import sphgeom

import astropy.time
from astropy.table import Table

transient_id = 30328322
ra, dec = 8.52941151,-43.0266337
mjd_start, mjd_end = 62300.0, 62600.0

repo = "/global/cfs/cdirs/lsst/production/gen3/roman-desc-sims/repo"
collections = ["u/descdm/preview_data_step1_w_2024_12"]

# The Step3/coadd collection is:
# collection = ["u/descdm/preview_data_step3_2877_19_w_2024_12"]

butler = Butler(repo, collections=collections)
butler.registry.queryCollections()

level = 10  # the resolution of the HTM grid
pixelization = sphgeom.HtmPixelization(level)

htm_id = pixelization.index(
    sphgeom.UnitVector3d(
        sphgeom.LonLat.fromDegrees(ra, dec)
    )
)
start_time = astropy.time.Time(mjd_start, format="mjd")
end_time = astropy.time.Time(mjd_end, format="mjd")

before_timespan = Timespan(None, start_time)
during_timespan = Timespan(start_time, end_time)
after_timespan = Timespan(end_time, None)


before_dataset_refs = butler.registry.queryDatasets("calexp", htm20=htm_id,
                                                    band="r",
                                     where="visit.timespan OVERLAPS my_timespan",
                                     bind={"my_timespan": before_timespan})
during_dataset_refs = butler.registry.queryDatasets("calexp", htm20=htm_id,
                                                    band="r",
                                     where="visit.timespan OVERLAPS my_timespan",
                                     bind={"my_timespan": during_timespan})
after_dataset_refs = butler.registry.queryDatasets("calexp", htm20=htm_id,
                                                    band="r",
                                     where="visit.timespan OVERLAPS my_timespan",
                                     bind={"my_timespan": after_timespan})

# Realize into list
before_dataset_refs = list(before_dataset_refs)
during_dataset_refs = list(during_dataset_refs)
after_dataset_refs = list(after_dataset_refs)

# Extract visit, band, detector
rows = \
[(transient_id, dr.dataId['instrument'], dr.dataId['visit'], dr.dataId['band'], dr.dataId['detector']) for dr in during_dataset_refs]
ddr_table = Table(rows=rows, names=("transient_id", "instrument", "visit", "band", "detector"))

before_dataset_refs[
[f"{transient_id} {dr.dataId['instrument']} {dr.dataId['visit']} {dr.dataId['band']} {dr.dataId['detector']}" for dr in during_dataset_refs]
before_dataset_refs[

# Get URL (On NERSC these are filepaths)
bdr = [butler.getURI(dr).geturl() for dr in before_dataset_refs]
ddr = [butler.getURI(dr).geturl() for dr in during_dataset_refs]
adr = [butler.getURI(dr).geturl() for dr in after_dataset_refs]
