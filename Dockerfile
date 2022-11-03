FROM nvcr.io/nvidia/pytorch:22.06-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV DOCKER_RUNNING=true

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

ARG SSL_KEYSTORE_PASSWORD
USER root

RUN mkdir /app
RUN chmod 777 /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

ENV PATH="/root/.local/bin:${PATH}"
ENV PATH="/home/user/.local/bin:${PATH}"

ADD requirements.txt .
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt
ADD . .

COPY setup.py /app/setup.py
RUN pip3 install --user --no-cache-dir -e /app

CMD [ "/bin/bash" ]