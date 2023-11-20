#!/usr/bin/awk -f

# For each line in the file
{
    # Check if the number of columns is greater than 698
    if (NF > 698) {
        print "Line " NR ": More than 698 columns (" NF " columns)";
    }
}