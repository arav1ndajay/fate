#!/bin/bash
cat ./ag/test/sci.txt ./ag/test/sports.txt ./ag/test/world.txt > ./ag/test/business-outliers.txt
cat ./ag/test/business.txt ./ag/test/sports.txt ./ag/test/world.txt > ./ag/test/sci-outliers.txt
cat ./ag/test/business.txt ./ag/test/sci.txt ./ag/test/world.txt > ./ag/test/sports-outliers.txt
cat ./ag/test/business.txt ./ag/test/sci.txt ./ag/test/sports.txt > ./ag/test/world-outliers.txt