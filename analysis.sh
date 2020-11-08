#!/usr/bin/env bash

fail_file=model_qp.txt
other_file=model_ours.txt

line_nums=$(cat $fail_file | grep Fail | awk '{ print $2 }')

orig_fails=0
other_success=0

for num in $line_nums
do
    result=$(sed -n ${num}p < $other_file | awk '{ print $5 }')

    if [[ $result == "Success!" ]]
    then
        ((other_success++))
    fi

    ((orig_fails++))
done


echo "For the $orig_fails failures in $fail_file, $other_file succeeded $other_success tiems"

