import os
import pytest

from astropy.coordinates import SkyCoord
import astropy.units as u
import astrophot as ap

from RomanDESCForwardModelLightcurves import (
    add_joint_center_parameter,
    lock_parameters_to_first_model,
    make_target,
    make_window_for_target,
)

DATASET = "RomanDESCSims"
DATADIR = os.path.join(os.path.dirname(__file__), "..", "data", DATASET)

IMAGE_BASENAMES = [
    "calexp_LSSTCam_r_r_57_5025071600940_R21_S01_u_descdm_preview_data_step1_w_2024_12_20240324T050830Z.fits",
    "calexp_LSSTCam_y_y_10_5025071600962_R21_S01_u_descdm_preview_data_step1_w_2024_12_20240326T152819Z.fits",
]

TEST_RUBIN_FILES = [os.path.join(os.path.dirname(__file__), "data", f) for f in IMAGE_BASENAMES]


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
    target = make_target(TEST_RUBIN_FILES[0], coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"])

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
    target = make_target(TEST_RUBIN_FILES[0], coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"])
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


@pytest.mark.parametrize("correct_sip", (False, True))
def test_make_model_multiple_bands(correct_sip):
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
        make_target(f, coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"]) for f in TEST_RUBIN_FILES
    )
    transient_ra, transient_dec = 8.52941151, -43.0266337
    host_ra, host_dec = 8.529866, -43.026571
    transient_xy = targets[0].world_to_plane(transient_ra, transient_dec)
    host_xy = targets[0].world_to_plane(host_ra, host_dec)
    npix = 75
    windows = [make_window_for_target(t, transient_ra, transient_dec, npix) for t in targets]

    model_static = [
        ap.models.AstroPhot_Model(
            name=f"galaxy model {i}",
            model_type="sersic galaxy model",
            target=t,
            psf_mode="full",
            parameters={"center": host_xy},
            window=w,
        )
        for i, (t, w) in enumerate(zip(targets, windows))
    ]
    model_sn = [
        ap.models.AstroPhot_Model(
            name=f"SN model {i}",
            model_type="point model",
            psf=t.psf,
            target=t,
            psf_mode="full",
            parameters={"center": transient_xy},
            window=w,
        )
        for i, (t, w) in enumerate(zip(targets, windows))
    ]

    bands = ["r", "y"]
    model_static_band = {}
    model_static_band["r"] = 0
    model_static_band["y"] = 1
    live_sn = [0, 0]

    if correct_sip:
        add_joint_center_parameter(model_static, model_sn, live_sn, host_xy, transient_xy, True, True)
    else:
        lock_parameters_to_first_model(model_static, model_sn, bands, model_static_band)

    all_model_list = []
    for model_host in model_static:
        if len(model_host) > 0:
            host_group_model = ap.models.AstroPhot_Model(
                name="Host",
                model_type="group model",
                models=[*model_host],
                target=targets,
            )
            all_model_list.extend(host_group_model)

    if len(model_sn) > 0:
        sn_group_model = ap.models.AstroPhot_Model(
            name="SN",
            model_type="group model",
            models=[*model_sn],
            target=targets,
        )
        all_model_list.extend(sn_group_model)

    model_host_sn = ap.models.AstroPhot_Model(
        name="Host+SN",
        model_type="group model",
        models=all_model_list,
        target=targets,
    )

    print("Initialize Host+SN model")
    print("Before init:")
    print(model_host_sn.parameters)
    model_host_sn.initialize()
    print("After initialize")
    print(model_host_sn.parameters)

    assert model_host_sn["galaxy model 0"]["PA"] is not None
    assert model_host_sn["galaxy model 1"]["PA"] is not None
