from lsst.daf.butler import Butler
from lsst import sphgeom

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

dataset_refs = butler.registry.queryDatasets("calexp", htm20=htm_id)
