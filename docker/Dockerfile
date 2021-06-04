FROM mohitkhubele95/tkdnn
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository 'deb http://security.ubuntu.com/ubuntu xenial-security main'
RUN apt-get -y update
RUN apt-get -y upgrade

RUN apt-get -y install cmake g++ git sudo vim curl rapidjson-dev awscli zip unzip dpkg libcpprest-dev libboost-dev libboost-all-dev

RUN  useradd -ms /bin/bash baggageai && echo "baggageai:baggageai" | chpasswd && adduser baggageai sudo
USER baggageai
WORKDIR /home/baggageai
EXPOSE 8080

#RUN aws s3 cp s3://dim-bai-s3-dev-developer-space/smiths_29_objects/BaggageAI.zip .

RUN mkdir files
#RUN mkdir files/include
#RUN mkdir files/server

#Change path of include and server folder accordingly
#COPY --chown=baggageai:baggageai src/include/ files/include
#COPY --chown=baggageai:baggageai src/server/ files/server
COPY --chown=baggageai:baggageai . files/
#RUN aws s3 cp s3://dim-bai-s3-dev-developer-space/smiths_29_objects/libBaggageAI.so files/

WORKDIR /home/baggageai/files
#RUN chmod 777 run.sh
#ENTRYPOINT ["./run.sh"]
