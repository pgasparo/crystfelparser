FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
WORKDIR /usr/src/app

# install delfta
RUN git clone ADDRESS 

RUN cd delfta && make
RUN conda init
RUN echo 'conda activate crystfelparser' >> ~/.bashrc
