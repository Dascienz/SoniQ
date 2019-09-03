FROM ubuntu:latest

RUN apt-get update && \
    apt-get -y install sudo && \
    sudo apt-get -y install \
    build-essential \
    checkinstall \
    libsdl1.2-dev \
    nano

COPY ./ /dgen-sdl/

WORKDIR /dgen-sdl/

RUN sh ./configure && \
    make && \
    make install

RUN mkdir rom && \
    "export TERM=xterm" >> ~/.bashrc

CMD [ "./dgen", "rom/sonicthehedgehog.bin" ]