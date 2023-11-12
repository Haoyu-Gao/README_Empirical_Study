#!/bin/bash

# var 1 is the date and var 2 is the hour
# echo "https://data.gharchive.org/$1-$2.json.gz"
wget -q "https://data.gharchive.org/$1-$2.json.gz"
gunzip "$1-$2.json.gz"