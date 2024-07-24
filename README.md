# Roman Supernova Forward Model

Model a supernova plus galaxy through a sequence of images in filter and time.

## Planned Development Order
1. [X] Basic model of point source (star)
2. [X] Basic model of isolated supernova
3. [X] Model of galaxy
4. [X] Supernova plus galaxy
5. [ ] SED model of supernova from a lightcurve template (e.g., SALT-3).

## Method
Uses AstroPhot, written by Connor Stone, 
[citation]

"ASTROPHOT: fitting everything everywhere all at once in astronomical images"
Stone et al. 2023, MNRAS 525, 4
https://doi.org/10.1093/mnras/stad2477
https://github.com/Autostronomy/AstroPhot
to forward model the SN point source and host galaxy.

Based on the AstroPhot Tutorial, TimeVariableModel.ipynb, by Michael Wood-Vasey
https://github.com/autostronomy/AstroPhot-tutorials/

## Data Sources
The AstroPhot-tutorial example was with 
"A synthetic Roman Space Telescope High-Latitude Time-Domain Survey: supernovae in the deep field"
Wang et al. 2023, MNRAS, 523, 3.
https://arxiv.org/abs/2204.13553
https://doi.org/10.1093/mnras/stad1652
https://roman.ipac.caltech.edu/sims/SN_Survey_Image_sim.html

This current repository uses the 2023/2024 Roman/Rubin RomanDESCSim data set by Troxel et al.
This current script is designed to be run with the data organized following the same scheme as the RomanDESCSim Roman WFI data.
DATADIR specifices the location of the data directory.
DATASET specifies a DATASET within that DATADIR.
Specifically, the code expects to find imaging data in
${DATADIR}/${DATASET}/images
and truth catalogs in
${DATADIR}/${DATASET}/truth

It assumes the existence of a `transient_info_table.csv` and `transient_host_info_table.csv` file for each given transient ID.  See the `tests/data/RomanDESC` directory for an example of the format of these files.

For Roman WFI images
${DATADIR} should contain subdirectories `images` and `truth`

with files arranged as

${DATADIR}/images/H158/42193/Roman_TDS_simple_model_H158_42193_1.fits.gz
${DATADIR}/truth/H158/42193/Roman_TDS_index_H158_42193_1.txt

## Example
Install required packages.  See `requirements_conda.txt` and `requirements_pip.txt` for the respective requirements.  (Not all needed packages are available via Conda).


On NERSC
```
INFODIR=/pscratch/sd/w/wmwv/RomanDESC
DATADIR=/pscratch/sd/w/wmwv/RomanDESC
transient_id=41024123441
python RomanDESCForwardModelLightcurves.py ${transient_id} --infodir ${INFODIR} --datadir ${DATADIR} --dataset RomanDESC
```

A general test example script.  Update the variables in the script to match your local system locations for DATADIR.
```
sh tests/test_one_lightcurve.sh
```
