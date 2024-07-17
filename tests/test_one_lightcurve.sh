# Change this to what the RomanDESCSims data are.
# Below DATADIR should be the subdirectories 'images' and 'truth'.
DATADIR=${HOME}/data/RomanDESCSims
# Run from base directoroy
python python/RomanDESCForwardModelLightcurves.py 41024123441 --infodir tests/data/RomanDesc --datadir ${DATADIR} --dataset RomanDESC
