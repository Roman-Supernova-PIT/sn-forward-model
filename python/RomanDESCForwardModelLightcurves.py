#!/usr/bin/env python
# coding: utf-8

"""
# RomanDESC SN Simulation modeling with AstroPhot

Author: Michael Wood-Vasey <wmwv@pitt.edu>
Last Verified to run: 2024-03-18

Use the [AstroPhot](https://autostronomy.github.io/AstroPhot/) package to model
the lightcurve of SN in Roman+Rubin DESC simulations

Notable Requirements:
astrophot
astropy
torch
webbpsf

Major TODO:
  * [~] Start utility support Python file as developing package
  * [ ] Write tests for package.  Decide on test data.
  * [~] Write logic into functions that can be called from Python scripts
  * [~] Implement SIP WCS in AstroPhot to handle small variations in positions
    - Instead implemented a per-image (but not per object) astrometric shift.

## Environment

This Notebook was developed and tested within a conda environment.
You can create this environment with:

```
conda create --name astrophot -c conda-forge python astropy cudatoolkit h5py \
  ipykernel jupyter matplotlib numpy pandas pyyaml pyarrow scipy requests tqdm
conda activate astrophot
pip install astrophot pyro-ppl torch webbpsf
ipython kernel install --user --name=astrophot
```
You then have to separately install the webbpsf data files
https://webbpsf.readthedocs.io/en/latest/installation.html#installing-the-required-data-files

You can declare the WEBBPSF data directory with, e.g.,
export WEBBPSF_PATH=/pscratch/sd/w/wmwv/RomanDESC/webbpsf-data
for a given session

or set the `WEBBPSF_PATH` in the conda environment,

```
conda env config vars set WEBBPSF_PATH=/pscratch/sd/w/wmwv/RomanDESC/webbpsf-data
```

This requires astrophot >= v0.15.2
"""

import argparse
import os
import re
from typing import Optional
import warnings

import astrophot as ap
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import webbpsf
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, join
from astropy.wcs import FITSFixedWarning, WCS


class Config:
    # These are 4k x 4k images
    pixel_scale = {"DC2": 0.2, "RomanDESC": 0.11}  # "/pixel
    fwhm = {"DC2": 0.6, "RomanDESC": 0.2}  # "

    # The HDU order is different between the two datasets
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
    # as are the FITS extension names
    hdu_names = {
        "DC2": {"image": "image", "mask": "mask", "variance": "variance"},
        "RomanDESC": {"image": "SCI", "mask": "DQ", "variance": "ERR"},
    }
    # so we have to use a translation regardless.

    # But the variance plane for the Roman images isn't actually right
    # So we use the Image plane for the variance.
    hdu_idx["RomanDESC"]["variance"] = hdu_idx["RomanDESC"]["image"]
    hdu_names["RomanDESC"]["variance"] = hdu_names["RomanDESC"]["image"]

    # Bad pixel mask values
    bad_pixel_bitmask = {}
    bad_pixel_bitmask["DC2"] = 0b0
    bad_pixel_bitmask["RomanDESC"] = 0b1

    all_config = {
        "fwhm": fwhm,
        "pixel_scale": pixel_scale,
        "hdu_idx": hdu_idx,
        "hdu_names": hdu_names,
        "bad_pixel_bitmask": bad_pixel_bitmask,
    }

    def __init__(self, dataset):
        self.fwhm = self.all_config["fwhm"][dataset]
        self.pixel_scale = self.all_config["pixel_scale"][dataset]
        self.hdu_idx = self.all_config["hdu_idx"][dataset]
        self.hdu_names = self.all_config["hdu_names"][dataset]
        self.bad_pixel_bitmask = self.all_config["bad_pixel_bitmask"][dataset]


def get_visit_band_detector_for_object_id(object_id, datadir):
    """
    Returns all of the image files that contain the location of the object.

    Note:
    For now this returns the results from a dict that was manually computed
    """
    image_info_file = os.path.join(datadir, "info", "visit_band_info.ecsv")
    image_info = Table.read(image_info_file)

    this_object = image_info["transient_id"] == object_id

    # Could instead Raise exception once we have an exception framework
    if sum(this_object) < 1:
        print(f"Object ID: '{object_id}' unknown.")
        return None

    return image_info[this_object]


def get_truth_table(truth_files, visits, transient_id):
    live_visits = []
    realized_flux = []
    flux = []
    mag = []

    for tf, v in zip(truth_files, visits):
        if not os.path.isfile(tf):
            print("Truth file {tf} is not a file.")
            continue
        this_truth_table = Table.read(tf, format="ascii")
        idx = this_truth_table["object_id"] == transient_id
        if sum(idx) == 0:
            continue
        transient_entry = this_truth_table[idx]
        live_visits.append(v)
        realized_flux.append(transient_entry["realized_flux"][0])
        flux.append(transient_entry["flux"][0])
        mag.append(transient_entry["mag"][0])

    truth_table = Table(
        {
            "visit": live_visits,
            "realized_flux": realized_flux,
            "flux": flux,
            "mag": mag,
        }
    )

    return truth_table


def get_transient_info_and_host(transient_id, infodir):
    # Read basic info catalog
    transient_info_file = os.path.join(infodir, "info", "transient_info_table.ecsv")
    transient_host_info_file = os.path.join(infodir, "info", "transient_host_info_table.ecsv")
    transient_info_table = Table.read(transient_info_file, format="ascii.ecsv")
    # Should eventually shift to a different way of tracking hosts
    # For now just reformatting into the previous way.
    transient_id_host_per_row = Table.read(transient_host_info_file, format="ascii.ecsv")
    transient_id_host = {}
    for r in np.unique(transient_id_host_per_row["transient_id"]):
        (idx,) = np.where(transient_id_host_per_row["transient_id"] == r)
        transient_id_host[r] = {
            "object_id": transient_id_host_per_row[idx]["object_id"],
            "ra": transient_id_host_per_row[idx]["ra"],
            "dec": transient_id_host_per_row[idx]["dec"],
        }

    transient_info_table.add_index("transient_id")

    transient_info = transient_info_table.loc[transient_id]
    transient_host = transient_id_host[transient_id]

    return transient_info, transient_host


def get_image_and_truth_files(transient_id, dataset, datadir):
    # Get list of images (visit, band, detector) that contain object position
    image_info = get_visit_band_detector_for_object_id(transient_id, datadir)

    # Define and load images and truth
    roman_image_file_format = "images/{band}/{visit}/Roman_TDS_simple_model_{band}_{visit}_{detector}.fits.gz"
    roman_truth_file_for_image_format = "truth/{band}/{visit}/Roman_TDS_index_{band}_{visit}_{detector}.txt"

    rubin_image_file_format = "calexp/*/{band}/{band}_{detector}/{visit}/calexp_LSSTCam_{band}_{band}_{detector}_{visit}_*_*_*.fits"
    rubin_truth_file_for_image_format = ""  # No truth

    image_file_basenames = []
    truth_file_basenames = []
    for instrument, visit, band, detector in zip(image_info["instrument"], image_info["visit"], image_info["band"], image_info["detector"]):
        if instrument == "WFI":
            image_file = roman_image_file_format.format(visit=visit, band=band, detector=detector)
            truth_file = roman_truth_file_for_image_format.format(visit=visit, band=band, detector=detector)
        elif instrument == "LSSTCam":
            image_file = rubin_image_file_format.format(visit=visit, band=band, detector=detector)
            truth_file = rubin_truth_file_for_image_format.format(visit=visit, band=band, detector=detector)
        else:
            print("Instrument {instrument} unknown.")

        image_file_basenames.append(image_file)
        truth_file_basenames.append(truth_file)

    image_files = [os.path.join(datadir, bn) for bn in image_file_basenames]
    truth_files = [os.path.join(datadir, bn) for bn in truth_file_basenames]

    return image_info, image_files, truth_files


def get_roman_psf(band, detector, x, y, ext_name="DET_SAMP"):
    """
    Return the Roman WFI PSF for the given band, detector at the detector position x, y

    Use `webbpsf` package for band, detector, x, y and SED.

    ext_name: ["OVERSAMP", "DET_SAMP", "OVERDIST", "DET_DIST"]

    https://roman-docs.stsci.edu/simulation-tools-handbook-home/webbpsf-for-roman/webbpsf-tutorials
    https://github.com/spacetelescope/webbpsf/blob/develop/notebooks/WebbPSF-Roman_Tutorial.ipynb
    """
    # translate from colloquial R, Z, Y, H to standard "F*" filter names
    standard_band_names = {
        "F062": "F062",
        "F087": "F087",
        "F106": "F106",
        "F129": "F129",
        "F146": "F146",
        "F158": "F158",
        "F184": "F184",
        "R062": "F062",
        "Z087": "F087",
        "Y106": "F106",
        "J129": "F129",
        "H158": "F158",
    }
    wfi = webbpsf.roman.WFI()
    wfi.filter = standard_band_names[band]
    wfi.detector = f"SCA{detector:02d}"
    wfi.detector_position = (x, y)
    wfi.options["parity"] = "odd"

    psf_hdu = wfi.calc_psf()
    psf = psf_hdu[ext_name].data

    return psf


def make_target(
    image_filepath,
    coord: SkyCoord,
    fwhm: float,
    pixel_scale: float,
    hdu_idx: dict,
    zeropoint: Optional[float] = None,
    bad_pixel_bitmask: Optional[int] = 0b0,
    psf_size: int = 51,
    do_mask=False,
):
    """Make an AstroPhot target.

    image_filepath: str, Filepath to image file.
        Image file assumed to have [image, mask, variance].
        WCS assumed to be present in image HDU header

    coord: SkyCoord object with center of window
    fwhm: float, Full-Width at Half-Maximum in arcsec
    pixel_scale: float, "/pix
       This is used along with fwhm, psf_size to set a Gaussian PSF model
       Would be better to have an actual PSF model from the image
    bad_pixel_bitmask: Int.  Packed bitmask of bad pixels on image.
    zeropoint: float, calibration of counts in image.
    psf_size: float, width of the PSF
    do_mask: Use mask.
    """
    # The RomanDESCSims Roman files have a misformatted DATE-OBS
    # (" " instead of "T" separator)
    # so we build the targets in a context manager to suppress the warning
    # messages for things that were automatically fixed when reading the file.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FITSFixedWarning)
        with fits.open(image_filepath) as hdu:
            header = hdu[0].header  # Primary header
            img = hdu[hdu_idx["image"]].data  # Image HDU
            # The WCS is stored in the image header not the primary header.
            # So grab that here.
            img_header = hdu[hdu_idx["image"]].header  # Image HDU Header
            var = hdu[hdu_idx["variance"]].data  # Variance HDU

            if do_mask:
                # We need to read the informative mask with a bad-value mask.
                # E.g., for an LSST Science Pipelines mask, mask values
                # don't mean bad, they just indicate something about the pixel
                # E.g., one flag is that that pixel is part of a footprint
                # of a valid object and we don't want to mask those!
                informative_mask = hdu[hdu_idx["mask"]].data  # Mask
                bad_pixel_mask = informative_mask & bad_pixel_bitmask

    wcs = WCS(img_header)
    band = header["FILTER"]
    detector = header["SCA_NUM"]

    zp_band = {"H158": 32.603}

    if zeropoint is None:
        zeropoint = zp_band[band]  # + 2.5 * np.log10(header["EXPTIME"])

    x, y = wcs.world_to_pixel(coord)

    psf = get_roman_psf(band, detector, x, y)

    target_kwargs = {
        "data": np.array(img, dtype=np.float64),
        "variance": var,
        "zeropoint": zeropoint,
        "psf": psf,
        "wcs": wcs,
    }

    if do_mask:
        target_kwargs["mask"] = bad_pixel_mask
    if coord is not None:
        target_kwargs["reference_radec"] = (coord.ra.degree, coord.dec.degree)

    target = ap.image.Target_Image(**target_kwargs)

    target.header.filename = image_filepath
    target.header.mjd = header["MJD-OBS"]
    target.header.band = header["FILTER"]
    # ZPTMAG is
    #     full_image.header['ZPTMAG']   = 2.5*np.log10(self.exptime*roman.collecting_area)
    # https://github.com/matroxel/roman_imsim/blob/864357c8d088164b9662007f2ebe50e23243368e/roman_imsim/sca.py#L133
    # This needs to be added to truth file "mag" to get calibrated mag
    target.header.sim_zptmag = header["ZPTMAG"]

    return target


def make_window_for_target(target, ra, dec, npix=75):
    window = target.window.copy()
    center_xy = window.world_to_pixel(ra, dec)

    xmin = center_xy[0] - npix // 2
    xmax = center_xy[0] + npix // 2
    ymin = center_xy[1] - npix // 2
    ymax = center_xy[1] + npix // 2

    window.crop_to_pixel([[xmin, xmax], [ymin, ymax]])
    return window


def plot_targets(targets, windows, plot_filename=None):
    n = len(targets.image_list)
    side = int(np.sqrt(n)) + 1
    fig, ax = plt.subplots(side, side, figsize=(3 * side, 3 * side))

    for i in range(n):
        ap.plots.target_image(fig, ax.ravel()[i], targets[i], window=windows[i], flipx=True)

    if plot_filename is not None:
        plt.savefig(plot_filename)


# We divide up because "model_image" expects a single axis object if single image
# while it wants an array of axis objects if there are multiple images in the image list
# model_image will not accept a one-element array if there is no image_list
def plot_target_model(model, **kwargs):
    if hasattr(model.target, "image_list"):
        _plot_target_model_multiple(model, **kwargs)
    else:
        _plot_target_model_single(model, **kwargs)


def _plot_target_model_multiple(
    model,
    window=None,
    titles=None,
    base_figsize=(12, 4),
    figsize=None,
    plot_filename=None,
):
    n = len(model.target.image_list)
    if figsize is None:
        figsize = (base_figsize[0], n * base_figsize[1])
    fig, ax = plt.subplots(n, 3, figsize=figsize)
    # Would like to just call this, but window isn't parsed as a list
    # https://github.com/Autostronomy/AstroPhot/issues/142
    #    ap.plots.target_image(fig, ax[:, 0], model.target, window=window, flipx=True)
    for axt, mod, win in zip(ax[:, 0], model.target.image_list, window):
        ap.plots.target_image(fig, axt, mod, win, flipx=True)

    if titles is not None:
        for i, title in enumerate(titles):
            ax[i, 0].set_title(title)
    ap.plots.model_image(fig, ax[:, 1], model, window=window, flipx=True)
    ax[0, 1].set_title("Model")
    ap.plots.residual_image(fig, ax[:, 2], model, window=window, flipx=True)
    ax[0, 2].set_title("Residual")

    if plot_filename is not None:
        plt.savefig(plot_filename)


def _plot_target_model_single(model, window=None, title=None, figsize=(16, 4)):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ap.plots.target_image(fig, ax[0], model.target, window=window, flipx=True)
    ax[0].set_title(title)
    ap.plots.model_image(fig, ax[1], model, window=window, flipx=True)
    ax[1].set_title("Model")
    ap.plots.residual_image(fig, ax[2], model, window=window, flipx=True)
    ax[2].set_title("Residual")


def plot_lightcurve(
    lightcurve, lightcurve_obs, lightcurve_truth, transient_id, dataset, snr_threshold=1, plot_filename=None
):
    color_for_band = {
        "u": "purple",
        "g": "blue",
        "r": "green",
        "i": "red",
        "z": "black",
        "y": "yellow",
        "Y": "blue",
        "J": "green",
        "H": "red",
        "F": "black",
        "Y106": "blue",
    }
    color_for_band["H158"] = color_for_band["H"]

    _, axes = plt.subplots(2, 1, height_ratios=[2, 1])
    ax = axes[0]

    for b in np.unique(lightcurve_obs["band"]):
        (idx,) = np.where((lightcurve_obs["band"] == b) & (lightcurve_obs["snr"] > snr_threshold))
        ax.errorbar(
            lightcurve_obs[idx]["mjd"],
            lightcurve_obs[idx]["mag"],
            lightcurve_obs[idx]["mag_err"],
            marker="o",
            markerfacecolor=color_for_band[b],
            markeredgecolor=color_for_band[b],
            ecolor=color_for_band[b],
            linestyle="none",
            label=f"fit {b}",
        )
    ax.set_ylabel("mag")
    # ax.set_xlabel("MJD")
    ax.set_title(f"Proof of Concept: {dataset} {transient_id}")
    plt.ylim(23.5, 19)

    if lightcurve_truth is not None:
        for b in np.unique(lightcurve["band"]):
            (idx,) = np.where(lightcurve["band"] == b)
            ax.scatter(
                lightcurve[idx]["mjd"],
                lightcurve[idx]["mag_truth"],
                edgecolor=color_for_band[b],
                facecolor="none",
                alpha=0.5,
                marker="*",
                label=f"model {b}",
            )

    ax.set_ylim(ax.get_ylim()[::-1])

    ax.legend(ncols=2)

    ###
    ax = axes[1]

    for b in np.unique(lightcurve["band"]):
        (idx,) = np.where((lightcurve["band"] == b))
        ax.errorbar(
            lightcurve[idx]["mjd"],
            lightcurve[idx]["mag_obs"] - lightcurve[idx]["mag_truth"],
            lightcurve[idx]["mag_err"],
            marker="o",
            markerfacecolor=color_for_band[b],
            markeredgecolor=color_for_band[b],
            ecolor=color_for_band[b],
            linestyle="none",
            label=f"{b}",
        )
    ax.set_ylabel("obs - truth [mag]")
    ax.set_xlabel("MJD")
    # plt.ylim(23.5, 17)
    ax.axhline(0, color="gray", ls="--")
    ax.set_ylim(1, -1)
    ax.set_xlim(axes[0].get_xlim())

    if plot_filename is not None:
        plt.savefig(plot_filename)


def plot_covariance(result, targets, plot_filename=None):
    _, axes = plt.subplots(1, 2, 1)

    plt.sca(axes[0])
    covar = result.covariance_matrix.detach().cpu().numpy()
    plt.imshow(
        covar,
        origin="lower",
        vmin=1e-8,
        vmax=1e-1,
        norm="log",
    )
    plt.colorbar()

    # Let's focus on the SN flux uncertainties:
    # This is a little clunky because I don't have a better way of looking up
    # the names of the parameters in the covariance matrix.

    plt.sca(axes[1])
    sn_flux_starts_at_parameter_idx = -len(targets.image_list)
    plt.imshow(
        covar[sn_flux_starts_at_parameter_idx:, sn_flux_starts_at_parameter_idx:],
        origin="lower",
        #    vmin=1e-6, vmax=1, norm="log",
    )
    plt.colorbar()

    if plot_filename is not None:
        plt.savefig(plot_filename)


def run_multiple_transients(
    transient_ids,
    datadir,
    infodir,
    dataset="RomanDESC",
    npix=75,
    verbose=False,
    overwrite=True,
):
    for transient_id in transient_ids:
        run_one_transient(
            transient_id,
            datadir,
            infodir,
            dataset="RomanDESC",
            npix=75,
            verbose=False,
            overwrite=True,
        )


def run_one_transient(
    transient_id,
    datadir,
    infodir,
    dataset="RomanDESC",
    npix=75,
    verbose=False,
    overwrite=True,
):
    config = Config(dataset)

    if verbose:
        print(f"Getting transient and static scene information for {transient_id}.")
    transient_info, transient_host = get_transient_info_and_host(transient_id, infodir)
    image_info, image_files, truth_files = get_image_and_truth_files(transient_id, dataset, datadir)
    lightcurve_truth = get_truth_table(truth_files, image_info["visit"], transient_id)
    print(lightcurve_truth)

    transient_coord = SkyCoord(transient_info["ra"], transient_info["dec"], unit=u.degree)
    if verbose:
        print(f"Found transient {transient_id} at ({transient_info['ra']}, {transient_info['dec']})")
        print("Building AstroPhot target image list from {len(image_info)} files.")

    targets = ap.image.Target_Image_List(
        make_target(
            f,
            coord=transient_coord,
            fwhm=config.fwhm,
            pixel_scale=config.pixel_scale,
            hdu_idx=config.hdu_idx,
            bad_pixel_bitmask=config.bad_pixel_bitmask,
        )
        for f in image_files
    )

    for i, target in enumerate(targets):
        target.header.visit = image_info["visit"][i]

    if verbose:
        print("Making windows for the scene on each image.")
    windows = [make_window_for_target(t, transient_info["ra"], transient_info["dec"], npix) for t in targets]

    if verbose:
        print("Saving postage stamps from each image.")
    plot_filename = f"transient_{dataset}_{transient_id}_stamps.png"
    plot_targets(targets, windows, plot_filename)

    # The coordinate axes are in arcseconds,
    # but in the local relative coordinate system for each image.
    # AstroPhot used the pixel scale to translate pixels -> arcsec.

    # Translate SN and host positions to projection plane positions for target.
    # By construction of our targets, this is in the same projection plane position.

    if verbose:
        print("Translating RA, Dec of transient and static objects to x, y in each image.")
    transient_xy = targets[0].world_to_plane(transient_info["ra"], transient_info["dec"])
    if len(transient_host["ra"]) > 1:
        host_xy = [
            targets[0].world_to_plane(r, d) for r, d in zip(transient_host["ra"], transient_host["dec"])
        ]
    else:
        host_xy = [targets[0].world_to_plane(transient_host["ra"], transient_host["dec"])]

    # ### Jointly fit model across images

    live_sn = [
        (target.header.mjd > transient_info["mjd_start"]) and (target.header.mjd < transient_info["mjd_end"])
        for target in targets
    ]

    model_sky = []
    model_static = []
    model_sn = []

    # The RomanDESC images are "raw" science images with sky.
    # The DC2 image are processed and have had a sky model removed.
    FIT_SKY = {"DC2": False, "RomanDESC": True}
    FIT_HOST = True
    FIT_SN = True
    CORRECT_SIP = True

    if FIT_SKY[dataset]:
        for i, (target, window) in enumerate(zip(targets, windows)):
            model_sky.append(
                ap.models.AstroPhot_Model(
                    name=f"sky model {i}",
                    model_type="flat sky model",
                    target=target,
                    window=window,
                )
            )

    # We might have multiple hosts in the scene.
    # Potentially eventually multiple stars
    if FIT_HOST:
        for i, hxy in enumerate(host_xy):
            model_static_band = {}
            this_object_model = []

            for j, (b, target, window) in enumerate(zip(image_info["band"], targets, windows)):
                this_object_model.append(
                    ap.models.AstroPhot_Model(
                        name=f"galaxy model {i, j}",
                        model_type="sersic galaxy model",
                        target=target,
                        psf_mode="full",
                        parameters={"center": hxy},
                        window=window,
                    )
                )
                # I think this assignment copies reference that points to same underlying object
                # in 'model_host' and 'model_host_band'
                # The initialization step assumes that the reference model gets initialized first.
                # So we just mark use the first model in the list of each band.
                if b not in model_static_band.keys():
                    model_static_band[b] = j

            # Define static by locking all parameters to the first in the band.
            for model in this_object_model:
                if model.name == this_object_model[model_static_band[b]].name:
                    continue
                for parameter in ["q", "PA", "n", "Re", "Ie"]:
                    model[parameter].value = this_object_model[model_static_band[b]][parameter]

            model_static.append(this_object_model)

    if FIT_SN:
        for i, (ls, target, window) in enumerate(zip(live_sn, targets, windows)):
            if not ls:
                continue
            model_sn.append(
                ap.models.AstroPhot_Model(
                    name=f"SN model {i}",
                    model_type="point model",
                    psf=target.psf,
                    target=target,
                    parameters={"center": transient_xy},
                    window=window,
                )
            )

    # AstroPhot doesn't handle SIP WCS yet.
    # We'll roughly work around this by allowing a small shift in position
    # for all (both) objects on the image.
    CORRECT_SIP = True
    if CORRECT_SIP:

        def calc_center(params):
            return params["nominal_center"].value + params["astrometric"].value

        if FIT_HOST and FIT_SN:
            host_center = [ap.param.Parameter_Node(name="nominal_center", value=hxy) for hxy in host_xy]

            sn_center = ap.param.Parameter_Node(name="nominal_center", value=transient_xy)

            live_sn_i = -1  # Accumulator to count live SN models
            for i, ls in enumerate(live_sn):
                # Require that we have the SN
                # because we need both Host and SN to do a joint astrometric offset fit
                if not ls:
                    continue
                live_sn_i += 1
                # The x, y delta is the same for both the SN and host
                # but can be different for each image.
                P_astrometric = ap.param.Parameter_Node(
                    name="astrometric",
                    value=[0, 0],
                )

                for j in range(len(host_center)):
                    model_static[j][i]["center"].value = calc_center
                    model_static[j][i]["center"].link(host_center[j], P_astrometric)

                model_sn[live_sn_i]["center"].value = calc_center
                model_sn[live_sn_i]["center"].link(sn_center, P_astrometric)
        else:
            for b, model in zip(image_info["band"], model_static):
                if model.name == model_static[model_static_band[b]].name:
                    continue
                for parameter in ["center"]:
                    model[parameter].value = model_static[model_static_band[b]][parameter]
            for b, model in zip(image_info["band"], model_sn):
                if model.name == model_sn[model_static_band[b]].name:
                    continue
                for parameter in ["center"]:
                    model[parameter].value = model_static[model_static_band[b]][parameter]

    # Constrain host model to be the same per band

    # Create a two-tier hierarchy of group models
    # following recommendation from Connor Stone.

    # Group model for each class: sky, host, sn
    all_model_list = []
    if len(model_sky) > 0:
        sky_group_model = ap.models.AstroPhot_Model(
            name="Sky",
            model_type="group model",
            models=[*model_sky],
            target=targets,
        )
        all_model_list.extend(sky_group_model)

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

    # Group model holds all the classes
    model_host_sn = ap.models.AstroPhot_Model(
        name="Host+SN",
        model_type="group model",
        models=all_model_list,
        target=targets,
    )

    # We have to initialize the model so that there is a value for `parameters["center"]`
    model_host_sn.initialize()
    print(model_host_sn.parameters)

    # FIT
    result = ap.fit.LM(model_host_sn, verbose=True).fit()
    print(result.message)

    result.update_uncertainty()

    # The uncertainties for the center positions and astrometric uncertainties
    # aren't calculated correctly right now.
    # But the flux uncertainties are reasonable.

    print(result.model.parameters)

    model_filename = f"Transient_{transient_id}_AstroPhot_model.yaml"
    result.model.save(model_filename)

    lightcurve_basename = f"lightcurve_{dataset}_{transient_id}"
    lightcurve_obs = make_lightcurve_from_fit(model_host_sn)
    lightcurve_obs_filename = lightcurve_basename + ".ecsv"
    if lightcurve_obs_filename is not None:
        lightcurve_obs.write(lightcurve_obs_filename, overwrite=overwrite)

    lightcurve = make_joint_lightcurve_from_obs_and_truth(lightcurve_obs, lightcurve_truth)

    if verbose:
        print(lightcurve)

    plot_filename = lightcurve_basename + ".png"
    plot_lightcurve(
        lightcurve,
        lightcurve_obs,
        lightcurve_truth,
        transient_id,
        dataset,
        plot_filename=plot_filename,
    )

    image_file_basenames = [os.path.basename(f) for f in image_files]
    plot_filename = f"stamps_{dataset}_{transient_id}_model.png"
    plot_target_model(
        model_host_sn,
        window=windows,
        titles=image_file_basenames,
        plot_filename=plot_filename,
    )


def make_lightcurve_from_fit(model_host_sn):
    """Takes input AstroPhot model fit and returns lightcurve."""
    sn_model_name_regex = re.compile("SN model [0-9]+")
    sn_model_names = [k for k in model_host_sn.models.keys() if sn_model_name_regex.match(k)]

    filenames = [os.path.basename(model_host_sn.models[m].target.header.filename) for m in sn_model_names]
    bands = [model_host_sn.models[m].target.header.band for m in sn_model_names]
    visits = [model_host_sn.models[m].target.header.visit for m in sn_model_names]
    mjds = [model_host_sn.models[m].target.header.mjd for m in sn_model_names]
    sim_zptmag = [model_host_sn.models[m].target.header.sim_zptmag for m in sn_model_names]

    zp = np.array([model_host_sn.models[m].target.zeropoint.detach().cpu().numpy() for m in sn_model_names])
    inst_mag = np.array(
        [
            -2.5 * model_host_sn.models[m].parameters["flux"].value.detach().cpu().numpy()
            for m in sn_model_names
        ]
    )
    mag_err = np.array(
        [
            2.5 * model_host_sn.models[m].parameters["flux"].uncertainty.detach().cpu().numpy()
            for m in sn_model_names
        ]
    )

    lightcurve_obs = Table(
        {
            "filename": filenames,
            "band": bands,
            "visit": visits,
            "mjd": mjds,
            "zp": zp,
            "sim_zptmag": sim_zptmag,
            "inst_mag": inst_mag,
            "mag_err": mag_err,
        }
    )

    lightcurve_obs["mag"] = lightcurve_obs["inst_mag"] + lightcurve_obs["zp"]
    lightcurve_obs["inst_flux"] = 10 ** (-0.4 * lightcurve_obs["inst_mag"])
    lightcurve_obs["inst_flux_err"] = (np.log(10) / 2.5) * (lightcurve_obs["inst_flux"] * mag_err)

    lightcurve_obs["snr"] = lightcurve_obs["inst_flux"] / lightcurve_obs["inst_flux_err"]

    zp_AB_to_nJy = 8.90 + 2.5 * 9

    lightcurve_obs["flux"] = 10 ** (-0.4 * (lightcurve_obs["mag"] - zp_AB_to_nJy))
    lightcurve_obs["flux_err"] = (lightcurve_obs["flux"] / lightcurve_obs["inst_flux"]) * lightcurve_obs[
        "inst_flux_err"
    ]

    lightcurve_obs["mjd"].info.format = "<10.3f"
    lightcurve_obs["zp"].info.format = ">7.4f"
    lightcurve_obs["flux"].info.format = ".3e"
    lightcurve_obs["flux_err"].info.format = ".3e"
    lightcurve_obs["snr"].info.format = "0.2f"
    lightcurve_obs["mag"].info.format = ">7.4f"
    lightcurve_obs["mag_err"].info.format = ">7.4f"

    return lightcurve_obs


def make_joint_lightcurve_from_obs_and_truth(lightcurve_obs, lightcurve_truth):

    lightcurve = join(
        lightcurve_truth,
        lightcurve_obs[
            [
                "filename",
                "visit",
                "band",
                "mjd",
                "zp",
                "sim_zptmag",
                "inst_mag",
                "mag_err",
                "mag",
                "inst_flux",
                "inst_flux_err",
                "snr",
                "flux",
                "flux_err",
            ]
        ],
        keys_left=["visit"],
        keys_right=["visit"],
        join_type="right",
        table_names=("truth", "obs"),
    )
    # Need to add the ZPTMAG stored in the FITS image header to the 'mag' value in the truth file
    # to get the calibrated AB-system magnitude
    lightcurve["mag_truth"] += lightcurve["sim_zptmag"]

    return lightcurve


def parse_and_run():
    parser = argparse.ArgumentParser(
        prog="SNForwardModel",
        description="Runs forward models of SN + static scence and generates lightcurves.",
    )
    parser.add_argument(
        "transient_id",
        type=int,
        nargs="+",
        help="""
Transient ID, or multiple IDs.
Used to look up information in 'transient_info_table.ecsv' and 'transient_host_info_table.ecsv'
        """,
    )
    parser.add_argument("--datadir", type=str, help="Location of image and truth files.")
    parser.add_argument("--infodir", type=str, help="Location of SN and host galaxy catalogs.")
    parser.add_argument("--dataset", type=str, default="RomanDESC", choices=["RomanDESC", "DC2"])
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    run_multiple_transients(
        transient_ids=args.transient_id,
        datadir=args.datadir,
        infodir=args.infodir,
        dataset=args.dataset,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    parse_and_run()
    # transient_id = 30328322
    # transient_id = 20202893
    # transient_id = 30005877
    # transient_id = 30300185
    # transient_id = 41024123441

    # This one fails to find isophote in initialization.
    # transient_id = 50006502
