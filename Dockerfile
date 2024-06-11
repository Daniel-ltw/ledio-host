FROM pytorch/pytorch:latest

COPY ./app /app

WORKDIR /app

RUN /bin/bash -c "pip install --upgrade pip"
RUN /bin/bash -c "pip install -r requirements.txt -U"

CMD ["tail", "-f", "/dev/null"]
