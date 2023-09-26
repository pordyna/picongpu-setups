#! /bin/bash

source /home/pawel/Work/LOCAL/PIConGPU-git/bash_picongpu.profile || exit
rm -r .build

cd "$PICSRC" || exit
git checkout mainline/dev || exit
#git checkout test-amplitude
#git checkout topic-nonGaussianEnvelope || exit
cd - || exit

pic-build -t 0
mkdir simOutput_example_mainline_gauss
cd simOutput_example_mainline_gauss || exit
rm -r ./*
../bin/picongpu -s 2000 -g 256 128 -d 1 1 --periodic 0 0 0 --progressPeriod 25 --openPMD.period 50 --openPMD.file simData  --openPMD.range :,: --openPMD.ext bp --openPMD.period 1 --openPMD.file focus --openPMD.range 128:129,100:101 --openPMD.ext bp



