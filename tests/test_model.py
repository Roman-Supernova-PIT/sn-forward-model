import os

from astropy.coordinates import SkyCoord
import astropy.units as u
import astrophot as ap

from RomanDESCForwardModelLightcurves import make_target, make_window_for_target

DATASET = "RomanDESCSims"
DATADIR = os.path.join(os.path.dirname(__file__), "..", "data", DATASET)

TEST_RUBIN_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "calexp_LSSTCam_r_r_57_5025071600940_R21_S01_u_descdm_preview_data_step1_w_2024_12_20240324T050830Z.fits",
)


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
        TEST_RUBIN_FILE, coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"]
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
        TEST_RUBIN_FILE, coord, fwhm=0.6, pixel_scale=0.2, hdu_idx=hdu_idx["DC2"]
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
