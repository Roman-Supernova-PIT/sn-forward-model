import os

import numpy as np
from RomanDESCForwardModelLightcurves import (Config,
                                              get_image_and_truth_files,
                                              get_roman_psf,
                                              get_transient_info_and_host,
                                              get_truth_table,
                                              get_visit_band_detector_for_object_id)

DATASET = "RomanDESCSims"
DATADIR = os.path.join(os.path.dirname(__file__), "..", "data", DATASET)


def test_roman_psf():

    band = "H158"
    detector = 17
    x = 200
    y = 400

    # Make sure we get get each of the extensions
    # And check that they have the right relative size.
    psf = get_roman_psf(band, detector, x, y, ext_name="OVERSAMP")
    assert psf.shape == (180, 180)

    psf = get_roman_psf(band, detector, x, y, ext_name="DET_SAMP")
    assert psf.shape == (45, 45)

    psf = get_roman_psf(band, detector, x, y, ext_name="OVERDIST")
    assert psf.shape == (180, 180)

    psf = get_roman_psf(band, detector, x, y, ext_name="DET_DIST")
    assert psf.shape == (45, 45)


def test_config():
    """Do we get a Config for known DATASETs?"""

    dc2_config = Config("DC2")
    assert dc2_config.hdu_idx["image"] == 1
    assert dc2_config.hdu_idx["variance"] == 3

    roman_config = Config("RomanDESC")
    assert roman_config.hdu_idx["mask"] == 3


def test_get_visit_band_detector_for_object_id():
    """Do we get information back for a given transient?"""

    transient_id = 30328322
    visit_band_detector = get_visit_band_detector_for_object_id(transient_id, DATADIR)
    assert visit_band_detector["detector"][0] == 17


def test_get_image_and_truth_files():
    """Do we get image and truth files back for a given transient?"""

    transient_id = 50006502
    image_info, image_files, truth_files = get_image_and_truth_files(
        transient_id, DATASET, DATADIR
    )

    assert len(image_info["visit"]) == len(image_files)
    assert len(truth_files) == 6


def test_get_truth_table():
    """Do we get a truth table for a given transient?"""

    transient_id = 50006502

    image_info, image_files, truth_files = get_image_and_truth_files(
        transient_id, DATASET, DATADIR
    )

    truth_table = get_truth_table(truth_files, image_info["visit"], transient_id)

    assert len(truth_table) == 5
    assert np.max(truth_table["flux"]) > 0


def test_get_transient_info_and_host():
    """Do we get information back on a transient and its host."""

    transient_id = 50006502

    transient_info, transient_host = get_transient_info_and_host(transient_id, DATADIR)

    assert transient_info["ra"] == 8.5296134
    assert len(transient_host["ra"]) == 5
