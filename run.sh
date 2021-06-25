#!/bin/sh

#Removing build folder of home directory to overcome overwriting issue
if [ -d ~/"build/" ]; then
  rm -rf ~/build/
fi

#Removing build folder of the project directory
if [ -d "build/" ]; then
  rm -rf build/
fi

mkdir build #build folder will be created and project will be build in that folder. If you want to make a folder with different name, then just change it.
cd build #Name of the folder

#Building commands
cmake ..
make -j16
#cmake -DCMAKE_BUILD_TYPE=Debug -G "CodeBlocks - Unix Makefiles" ../
#cmake --build . --target BaggageAIApi -- -j4

#Running DemoApp application
cd ..
build/baggageAPI

