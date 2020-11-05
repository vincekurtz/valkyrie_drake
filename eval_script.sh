#!/usr/bin/env bash

##
#
# A bash script for batch testing the controllers over many trials. 
#
##

TOTAL_TRIALS=100;
SUCCESS_COUNT=0;

for i in $(seq 1 $TOTAL_TRIALS)
do
    echo -n "Trial $i / $TOTAL_TRIALS: "

    $(./simulate_valkyrie.py > /dev/null 2>&1)
    return_code=$?

    if (($return_code == 0))
    then
        ((SUCCESS_COUNT++))
        echo "Success!"
    else
        echo "Failed!"
    fi
    
done

echo "Experiment Finished."
echo "$SUCCESS_COUNT / $TOTAL_TRIALS successful"
