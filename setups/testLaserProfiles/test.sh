#! /bin/bash

source /home/pawel/Work/LOCAL/PIConGPU-git/bash_picongpu.profile || exit
rm -r .build

cd "$PICSRC" || exit
#git checkout test-amplitude
git checkout mainline/dev || exit
git status
cd - || exit

pic-build -t 0
cd simOutput_ref || exit
rm -r ./*
../bin/picongpu -s 1000 -g 446 350 -d 1 1 --periodic 0 0 0 --progressPeriod 25 --openPMD.period 25 --openPMD.file simData || exit

cd ../
rm-r .build
cd "$PICSRC" || exit
pwd
git checkout topic-nonGaussianEnvelope || exit
#git checkout b0f16076178021fe2156d15dc87cea86cdbdfb5c || exit
git status
cd - || exit
pic-build -t 0
cd simOutput_new || exit
rm -r ./*
../bin/picongpu -s 1000 -g 446 350 -d 1 1 --periodic 0 0 0 --progressPeriod 25 --openPMD.period 25 --openPMD.file simData || exit

cd ../lib/python/test/testLaserProfiles/laser || exit
python verify_laser.py




