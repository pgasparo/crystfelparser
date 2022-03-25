FROM gpuci/miniconda-cuda:11.2-devel-ubuntu20.04
WORKDIR /usr/src/app

# install crystfelparser
RUN git clone https://github.com/pgasparo/crystfelparser 
# update conda
RUN source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate 
RUN conda update -n base -c defaults conda

RUN cd crystfelparser && make
RUN conda init
RUN echo 'conda activate crystfelparser' >> ~/.bashrc
