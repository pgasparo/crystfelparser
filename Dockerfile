FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
WORKDIR /usr/src/app

# install delfta
RUN git clone https://github.com/pgasparo/crystfelparser 

RUN cd crystfelparser && make
RUN conda init
RUN echo 'conda activate crystfelparser' >> ~/.bashrc
