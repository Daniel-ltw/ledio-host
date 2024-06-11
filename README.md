# Exobrain

Let's build an exobrain assistant that can remember all the things. There are some sample html files in `/app/docs` containing some legislation around AI.

- https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:52021PC0206
- https://legistar.council.nyc.gov/LegislationDetail.aspx?ID=4344524&GUID=B051915D-A9AC-451E-81F8-6596032FA3F9

## Setup

** Fork this repo to your user account **

```
git clone git@github.com:[your-username]/exobrain.git
cd exobrain
git remote add upstream git@github.com:practical-ai-for-devs/exobrain.git
```

### Environment

The Groq api key is the only external dependency but there are also some internal environment variables for connecting the app container to do the db

```
cp sample.env .env
```

### Docker in Ubuntu

The docker container is built off the pytorch container (3.5GB download) and should take a minute or two to install once that is downloaded.

```
./build-docker
```

Initialise langfuse db

```
docker compose up db -d
docker compose exec db ./init-db.sh
```

You can use `./start-docker` and `./stop-docker` to start and stop the containers.

Note that `start-docker` will launch a shell in the app container which you can call your python code. If you need a new pip dependency then you can install it from the app container shell, it's also a good idea to update the requirements.txt so it is still there if you rebuild the container.

### Docker in Windows (powershell)

Buiding docker

```
New-Item -ItemType Directory -Force -Path .\pgdata
docker-compose build
```

Initialise langfuse database

```
docker-compose up db -d
docker-compose exec db ./init-db.sh
```

Starting docker containers

```
docker-compose up -d
```

Connecting to the docker container

```
docker-compose exec app /bin/bash
```

Stopping docker containers

```
docker-compose down
```

exit or Ctrl-d will get you out of the docker container back into powershell.

### Running without Docker

You'll need to install postgres on your machine and adjust the env vars to connect to it. Then setup a conda env as normal.

```
conda create -n exobrain
conda activate exobrain
conda install pip
pip install -r requirements.txt
```

### Setting up langfuse

1. visit http://localhost:3000/auth/sign-up
2. create an account
3. create a new project
4. create a new api key
5. paste the secret and public into your .env file
6. restart your servers

## Running

From the app container shell

```
python main.py [commands]
```

There is also a convenience `./psql` script which will connect to the postgres database if you want to inspect the data in there directly. `\d` will show you the details of the database and you can enter sql directly into the prompt.

## Tip: Chat to LangChain about it!

At any point in this process, feel free to visit [Chat LangChain](https://chat.langchain.com/). You can talk to the Docs about your process, if you're stuck or need to understand a step or a concept better, or just want to try it out!

## Step 0: review code

Run `python main.py` and ask the vanilla Groq LLM some questions about the EU AI Act or the New York Automated Employment Decision Tools bylaw. See what it knows already.

Research any code that doesn't make sense to you.

Have a look through the [demo code](./app/demo.py) - think about how you would split that into separate files and functions.

## Step 1: Slice the text into chunks

Transform the document passed into your application into chunks which will fit into the bge-small-en embedding (512 tokens)

- https://python.langchain.com/v0.2/docs/integrations/text_embedding/bge_huggingface/
- https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/
- https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer

## Step 2: Setup your pgvector store

- create connection string using the environment variables in .env
- the driver is 'psycopg' and the port 5432
- setup a record_manager to prevent chunks being readded

- https://python.langchain.com/v0.2/docs/integrations/vectorstores/pgvector/
- https://python.langchain.com/v0.2/docs/how_to/indexing/

## Step 3: save your sliced documents

- use the index function to save your sliced documents into the vector store

## Step 4: augment your chatbot

- integrate your vector store into your chatbot so it automatically augments the conversation with items from the db

https://python.langchain.com/v0.2/docs/how_to/#qa-with-rag

## Step 5: track your traces in langfuse

https://langfuse.com/docs/integrations/langchain/tracing

## Stretch

- add ability save pdf and websites directly
- experiment with different [retrieveral strategies](https://python.langchain.com/v0.2/docs/how_to/#retrievers). How could you evaluate the difference?
- try a different [embedding model](https://python.langchain.com/v0.2/docs/integrations/text_embedding/). Again, how would you know if it improves performance?
- figure out how much it would cost if you were using (openai embedding models)
