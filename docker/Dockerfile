FROM ceccocats/tkdnn:latest
LABEL maintainer "Francesco Gatti"

RUN cd && git clone https://github.com/ceccocats/tkDNN.git && cd tkDNN && mkdir build && cd build \ 
    && cmake .. && make -j12


