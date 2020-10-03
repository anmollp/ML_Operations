#!/bin/bash

echo "Loading Data to Database"
python3 utility/CsvToDatabase.py
if [[ $? == 0 ]]
then
    echo "Load Successfull"
else
    echo "Load Failed"
fi

