#!/bin/bash
cat ./20ng/test/misc.txt ./20ng/test/pol.txt ./20ng/test/rec.txt ./20ng/test/rel.txt ./20ng/test/sci.txt > ./20ng/test/comp-outliers.txt
cat ./20ng/test/comp.txt ./20ng/test/pol.txt ./20ng/test/rec.txt ./20ng/test/rel.txt ./20ng/test/sci.txt > ./20ng/test/misc-outliers.txt
cat ./20ng/test/comp.txt ./20ng/test/misc.txt ./20ng/test/rec.txt ./20ng/test/rel.txt ./20ng/test/sci.txt > ./20ng/test/pol-outliers.txt
cat ./20ng/test/comp.txt ./20ng/test/misc.txt ./20ng/test/pol.txt ./20ng/test/rel.txt ./20ng/test/sci.txt > ./20ng/test/rec-outliers.txt
cat ./20ng/test/comp.txt ./20ng/test/misc.txt ./20ng/test/pol.txt ./20ng/test/rec.txt ./20ng/test/sci.txt > ./20ng/test/rel-outliers.txt
cat ./20ng/test/comp.txt ./20ng/test/misc.txt ./20ng/test/pol.txt ./20ng/test/rec.txt ./20ng/test/rel.txt > ./20ng/test/sci-outliers.txt
