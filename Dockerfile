FROM python:3.10

### Python Package installations
RUN python -m pip install pip --upgrade && \ 
            pip install nemoguardrails

RUN mkdir -p /app/config 
COPY config /app/config/

WORKDIR  /app/api_server

# Expose port
EXPOSE 8000

#RUN Server
ENTRYPOINT ["nemoguardrails"]
CMD ["server", "--config", "/app/config"]