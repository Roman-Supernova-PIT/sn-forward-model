# A general test example script.
# Update the variables in the script to match your local system locations for DATADIR.
# In DATADIR should be the subdirectories 'images' and 'truth'.
DATADIR=${HOME}/data/RomanDESCSims

# In INFODIR should be transient_info_table.csv
#                      transient_host_info_table.csv
# We have these in tests/data for a few transients, so we can use those to test
INFODIR=tests/data/RomanDESC
DATASET=RomanDESC
oid=41024123441
# Run from base directoroy
python python/RomanDESCForwardModelLightcurves.py \
    ${oid} \
    --infodir ${INFODIR} \
    --datadir ${DATADIR} \
    --dataset ${DATASET}
