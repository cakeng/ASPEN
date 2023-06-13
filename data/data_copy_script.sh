#!/bin/bash
echo -n "" > "$2"
for (( i=1; i<=$3; i++ ))
do
    cat "$1" >> "$2"
done