#!/bin/bash

############################################################
# Auxiliary functions                                      #
############################################################

# Function to extract values from a given line (i) in the CSV file
extract_line() {
    local csv_file="$1"
    local line_number="$2"

    # Read the header row to extract the variable names
    header=$(head -n 1 "$csv_file")
    
    # Extract the line corresponding to the given line number (1-based index)
    line=$(sed -n "${line_number}p" "$csv_file")

    # Convert the header into an array of variables (CSV header can contain commas)
    IFS=',' read -r -a header_array <<< "$header"
    IFS=',' read -r -a line_array <<< "$line"

    # Loop through the header and assign the corresponding value from the line
    for x in "${!header_array[@]}"; do
        var_name="${header_array[$x]}"
        var_value="${line_array[$x]}"

        # Remove rogue line breaks
        cleaned_var_name=$(echo "$var_name" | tr -d '\r')
        cleaned_var_value=$(echo "$var_value" | tr -d '\r')
        
        # Export the value as a variable with the header name
        echo "$cleaned_var_name=$cleaned_var_value"
        export "$cleaned_var_name"="$cleaned_var_value"
    done
}

############################################################
# Main program                                             #
############################################################


# Loop over all lines in the parameter csv file
NUM_LINES=$(($(wc -l < test_vals_short_GSD120.csv) - 1))

for i in $(seq 1 $NUM_LINES); do
    echo "Submitting job for parameter values $i"
    extract_line test_vals_short_GSD120.csv $(($i + 1))
    qsub -N "lambda_ex0_est_sweep__short_$i" -V lambda_ex1_sweepscript.sh 
done