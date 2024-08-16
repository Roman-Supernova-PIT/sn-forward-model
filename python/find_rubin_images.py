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


def load_transient_info():
    names = ("transient_id", "ra", "dec", "mjd_start", "mjd_end")
    rows = (
        (30328322, 8.52941151, -43.0266337, 62300.0, 62600.0),
        (30300185, 8.53301717, -43.0415779, 62300.0, 62600.0),
        (41024123441, 8.535177, -43.041163, 60000.0, 70000.0),
        (20202893, 8.037774, -42.752337, 62236.0, 62707.0),
        (120000631, 8.595502038478454, -43.37404862401809, 63550.0, 63570.0),
        (120001196, 8.536407954475163, -43.21273817150576, 63550.0, 63570.0),
        (120002335, 8.492362151687624, -42.556687375246604, 63550.0, 63570.0),
        (120002788, 8.57487346783137, -43.46253917213952, 63550.0, 63570.0),
        (120003124, 8.408964293617913, -43.18411887823555, 63550.0, 63570.0),
        (120003660, 8.384744075725367, -42.836192769229676, 63550.0, 63570.0),
        (120004108, 8.469384560370557, -43.248344795186775, 63550.0, 63570.0),
        (120004970, 8.327411377167403, -42.62340899762345, 63550.0, 63570.0),
        (120005864, 8.398214489025749, -43.00826717221937, 63550.0, 63570.0),
        (120009480, 8.416564960246284, -43.090615140770645, 63550.0, 63570.0),
        (50130157, 8.414873040573653, -42.99870024490142, 63207.0, 63269.0),
        (50130277, 8.586162270818827, -43.133063263302674, 62383.0, 62689.0),
        (50132692, 8.6756901031846, -43.330377785632, 62415.0, 62666.0),
        (50137499, 8.474545189929705, -42.98452065295462, 62567.0, 62898.0),
        (50138070, 8.365749122707813, -43.32403670199172, 62704.0, 62854.0),
        (50139783, 8.597135505534567, -43.160325335401744, 63011.0, 63267.0),
        (110000220, 8.47397564535685, -43.15524333319279, 62323.0, 62586.0),
        (110002468, 8.391780192235615, -43.31772497932507, 62543.0, 62780.0),
        (110003217, 8.361546860887035, -42.532323540431406, 63235.0, 63269.0),
        (110003236, 8.447869862105966, -42.79325618686947, 62694.0, 62970.0),
    )
    transients = Table(rows=rows, names=names)
    return transients


def get_butler():
    repo = "/global/cfs/cdirs/lsst/production/gen3/roman-desc-sims/repo"
    collections = ["u/descdm/preview_data_step1_w_2024_12"]

    # The Step3/coadd collection is:
    # collection = ["u/descdm/preview_data_step3_2877_19_w_2024_12"]

    butler = Butler(repo, collections=collections)

    return butler


def get_table(butler, transient_id, htm_id, timespan, band="r", dataset_type="calexp"):
    """
    Get table of dataset, list of filepaths

    transient_id, instrument, visib, band, detector, filepath
    """
    dataset_refs = butler.registry.queryDatasets(
        dataset_type,
        htm20=htm_id,
        band=band,
        where="visit.timespan OVERLAPS my_timespan",
        bind={"my_timespan": timespan},
    )
    # Extract visit, band, detector
    # Get URL (On NERSC these are filepaths)
    rows = [
        (
            transient_id,
            dr.dataId["instrument"],
            dr.dataId["visit"],
            dr.dataId["band"],
            dr.dataId["detector"],
            butler.getURI(dr).geturl(),
        )
        for dr in dataset_refs
    ]
    if len(rows) > 0:
        dr_table = Table(
            rows=rows, names=("transient_id", "instrument", "visit", "band", "detector", "filepath")
        )
    else:
        dr_table = Table()

    return dr_table


def get_and_write_matching_observations(butler, transient_id, ra, dec, mjd_start, mjd_end):
    level = 10  # the resolution of the HTM grid
    pixelization = sphgeom.HtmPixelization(level)

    htm_id = pixelization.index(sphgeom.UnitVector3d(sphgeom.LonLat.fromDegrees(ra, dec)))
    start_time = astropy.time.Time(mjd_start, format="mjd")
    end_time = astropy.time.Time(mjd_end, format="mjd")

    before_timespan = Timespan(None, start_time)
    during_timespan = Timespan(start_time, end_time)
    after_timespan = Timespan(end_time, None)

    bdr = get_table(butler, transient_id, htm_id, before_timespan, band="r", dataset_type="calexp")
    ddr = get_table(butler, transient_id, htm_id, during_timespan, band="r", dataset_type="calexp")
    adr = get_table(butler, transient_id, htm_id, after_timespan, band="r", dataset_type="calexp")
    bdr.write(f"{transient_id}_image_info_before.csv", overwrite=True, delimiter=" ")
    ddr.write(f"{transient_id}_image_info_during.csv", overwrite=True, delimiter=" ")
    adr.write(f"{transient_id}_image_info_after.csv", overwrite=True, delimiter=" ")


def run():
    transients = load_transient_info()
    butler = get_butler()
    for row in transients:
        get_and_write_matching_observations(
            butler, row["transient_id"], row["ra"], row["dec"], row["mjd_start"], row["mjd_end"]
        )


if __name__ == "__main__":
    run()
