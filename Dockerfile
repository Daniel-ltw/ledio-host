FROM pytorch/pytorch:latest

COPY ./app /app

WORKDIR /app

RUN apt-get update && apt-get install --fix-broken --yes build-essential
RUN /bin/bash -c "pip install --upgrade pip"
RUN /bin/bash -c "pip install -r requirements.txt -U"

CMD ["tail", "-f", "/dev/null"]
