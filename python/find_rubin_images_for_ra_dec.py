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


def get_table(htm_id, timespan, band="r", dataset_type="calexp"):
    """
    Get table of dataset, list of filepaths

    transient_id, instrument, visib, band, detector
    """
    dataset_refs = butler.registry.queryDatasets(
        dataset_type,
        htm20=htm_id,
        band=band,
        where="visit.timespan OVERLAPS my_timespan",
        bind={"my_timespan": timespan}
    )
    # Extract visit, band, detector
    rows = \
    [(transient_id, dr.dataId['instrument'], dr.dataId['visit'], dr.dataId['band'], dr.dataId['detector']) for dr in dataset_refs]
    if len(rows) > 0:
        dr_table = Table(rows=rows, names=("transient_id", "instrument", "visit", "band", "detector"))
    else:
        dr_table = Table()

    # Get URL (On NERSC these are filepaths)
    dr_filepaths = [butler.getURI(dr).geturl() for dr in dataset_refs]

    return dr_table, dr_filepaths


bdr, bdr_filepaths = get_table(htm_id, before_timespan, band="r", dataset_type="calexp")
ddr, ddr_filepaths = get_table(htm_id, during_timespan, band="r", dataset_type="calexp")
adr, adr_filepaths = get_table(htm_id, after_timespan, band="r", dataset_type="calexp")
bdr.write(f"{transient_id}_image_info_before.csv", overwrite=True)
ddr.write(f"{transient_id}_image_info_during.csv", overwrite=True)
adr.write(f"{transient_id}_image_info_after.csv", overwrite=True)
