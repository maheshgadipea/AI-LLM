FROM python:3.10

RUN mkdir -p /app/model && \
    mkdir -p /app/api_server && \
    chmod -R 777 /app/api_server && \
    chmod -R 777 /app/model

ADD roberta_large.tar.gz /app/model/

COPY api_server /app/api_server

## OS configurations
RUN apt-get update && apt-get install -y build-essential 

# Install the HDF5 library
RUN apt-get update && apt-get install -y libhdf5-dev 

### Python Package installations
RUN python -m pip install pip --upgrade && \ 
              pip install "fastapi[all]" h5py

RUN pip install tensorflow transformers torch

WORKDIR  /app/api_server

# Expose port
EXPOSE 8000

#RUN Server
ENTRYPOINT ["uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000"]