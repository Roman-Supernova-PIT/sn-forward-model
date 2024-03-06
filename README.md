# Roman Supernova Forward Model

Model a supernova plus galaxy through a sequence of images in filter and time.

## Planned Development Order
1. [ ] Basic model of point source (star)
2. [ ] Basic model of isolated supernova
3. [ ] Model of galaxy
4. [ ] Supernova plus galaxy
5. [ ] SED model of supernova from a lightcurve template (e.g., SALT-3).

## Data Source
AstroPhot-tutorial example was with 
"A synthetic Roman Space Telescope High-Latitude Time-Domain Survey: supernovae in the deep field"
Wang et al. 2023, MNRAS, 523, 3.
https://arxiv.org/abs/2204.13553
https://doi.org/10.1093/mnras/stad1652
https://roman.ipac.caltech.edu/sims/SN_Survey_Image_sim.html

Next step is to use the 2023/2024 Roman/Rubin by Troxel et al.

Uses AstroPhot, written by Connor Stone, 
[citation]
"ASTROPHOT: fitting everything everywhere all at once in astronomical images"
Stone et al. 2023, MNRAS 525, 4
https://doi.org/10.1093/mnras/stad2477
https://github.com/Autostronomy/AstroPhot
to forward model the SN point source and host galaxy.

Based on the AstroPhot Tutorial, TimeVariableModel.ipynb, by Michael Wood-Vasey
https://github.com/autostronomy/AstroPhot-tutorials/
