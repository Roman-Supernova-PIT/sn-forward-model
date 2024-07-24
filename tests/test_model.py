import os

from astropy.coordinates import SkyCoord
import astropy.units as u
import astrophot as ap

from RomanDESCForwardModelLightcurves import make_target, make_window_for_target

DATASET = "RomanDESCSims"
DATADIR = os.path.join(os.path.dirname(__file__), "..", "data", DATASET)

IMAGE_BASENAMES = [
    "calexp_LSSTCam_r_r_57_5025071600940_R21_S01_u_descdm_preview_data_step1_w_2024_12_20240324T050830Z.fits",
    "calexp_LSSTCam_y_y_10_5025071600962_R21_S01_u_descdm_preview_data_step1_w_2024_12_20240326T152819Z.fits",
]

TEST_RUBIN_FILES = [
    os.path.join(os.path.dirname(__file__), "data", f) for f in IMAGE_BASENAMES
]


def test_make_target():
    """
    Test that we can make an AstroPhot target
    """
    hdu_idx = {
        "DC2": {
            "image": 1,
            "mask": 2,
            "variance": 3,
            "psfex_info": 11,
            "psfex_data": 12,
        },
        "RomanDESC": {"image": 1, "mask": 3, "variance": 2},
    }

    coord = SkyCoord(8.52941151, -43.0266337, unit=u.deg)
    target = make_target(
        TEST_RUBIN_FILES[0], coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"]
    )

    assert target is not None


def test_make_model():
    """
    Test that we can make an AstroPhot model

    Need a target to make a model for
    """
    hdu_idx = {
        "DC2": {
            "image": 1,
            "mask": 2,
            "variance": 3,
            "psfex_info": 11,
            "psfex_data": 12,
        },
        "RomanDESC": {"image": 1, "mask": 3, "variance": 2},
    }

    coord = SkyCoord(8.52941151, -43.0266337, unit=u.deg)
    target = make_target(
        TEST_RUBIN_FILES[0], coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"]
    )
    transient_ra, transient_dec = 8.52941151, -43.0266337
    host_ra, host_dec = 8.529866, -43.026571
    host_xy = target.world_to_plane(host_ra, host_dec)
    npix = 75
    window = make_window_for_target(target, transient_ra, transient_dec, npix)

    model = ap.models.AstroPhot_Model(
        name="test",
        model_type="sersic galaxy model",
        target=target,
        psf_mode="full",
        parameters={"center": host_xy},
        window=window,
    )

    assert model is not None

    model.initialize()
    assert model.parameters["PA"] is not None
    print(model.parameters)


def test_make_model_multiple_bands():
    """
    Test that we can make an AstroPhot model

    Need a target to make a model for
    """
    hdu_idx = {
        "DC2": {
            "image": 1,
            "mask": 2,
            "variance": 3,
            "psfex_info": 11,
            "psfex_data": 12,
        },
        "RomanDESC": {"image": 1, "mask": 3, "variance": 2},
    }

    coord = SkyCoord(8.52941151, -43.0266337, unit=u.deg)
    targets = ap.image.Target_Image_List(
        make_target(f, coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"])
        for f in TEST_RUBIN_FILES
    )
    transient_ra, transient_dec = 8.52941151, -43.0266337
    host_ra, host_dec = 8.529866, -43.026571
    host_xy = targets[0].world_to_plane(host_ra, host_dec)
    npix = 75
    windows = [
        make_window_for_target(t, transient_ra, transient_dec, npix) for t in targets
    ]

    models = [
        ap.models.AstroPhot_Model(
            name=f"test {i}",
            model_type="sersic galaxy model",
            target=t,
            psf_mode="full",
            parameters={"center": host_xy},
            window=w,
        )
        for i, (t, w) in enumerate(zip(targets, windows))
    ]

    model_group = ap.models.AstroPhot_Model(
        name="group",
        model_type="group model",
        models=models,
        target=targets,
    )

    print("Before initialize")
    print(model_group.parameters)
    model_group.initialize()
    print("After initialize")
    print(model_group.parameters)


#    assert model.parameters["PA"] is not None
