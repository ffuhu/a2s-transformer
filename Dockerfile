#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim nano -y
RUN apt clean

RUN pip install --upgrade pip
RUN pip install pybind11

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
