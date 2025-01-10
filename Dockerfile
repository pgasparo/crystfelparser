FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
WORKDIR /usr/src/app

# install crystfelparser
RUN git clone https://github.com/pgasparo/crystfelparser 
# update conda
RUN conda update -n base conda -c anaconda

RUN cd crystfelparser && pip install .
RUN echo 'conda ini' >> ~/.bashrc
RUN echo 'conda activate crystfelparser' >> ~/.bashrc
