#!/bin/bash

RELU_FLAGS=("--use_relu" "--no-use_relu")

BASELINE_FLAGS=("--use_baseline" "--no-use_baseline")

EUCLID_FLAGS=("--euclidean" "--no-euclidean")

COUNTER=1

echo "Starting experiment suite..."

for relu in "${RELU_FLAGS[@]}"; do
    for blf in "${BASELINE_FLAGS[@]}"; do
	for euc in "${EUCLID_FLAGS[@]}"; do
        
        EXPID=$(printf "%s%s%s" "$euc" "$relu" "$blf")
	EXPID="${EXPID//-}"
        
        echo "Running expid: $EXPID | Activation: $relu | Baseline: $blf | Euclid: $euc"
        
        python fgnn.py \
	       $relu \
	       $blf \
	       $euc \
	       --expid "$EXPID"
        
        COUNTER=$((COUNTER + 1))
	
	done
    done
done

echo "done."
