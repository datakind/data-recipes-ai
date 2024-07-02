FROM mcr.microsoft.com/azureml/promptflow/promptflow-runtime-stable:latest

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

ARG OPENAI_API_ENDPOINT
ENV OPENAI_API_ENDPOINT=$OPENAI_API_ENDPOINT

COPY ./flows /app
COPY ./utils /app/chainlit-ui-evaluation/utils
COPY ./templates /app/chainlit-ui-evaluation/templates
COPY ./management/skills.py /app/chainlit-ui-evaluation/recipes/skills.py
COPY ./ui/chat-chainlit-assistant/app.py /app/chainlit-ui-evaluation/app.py

RUN pip3 install --upgrade pip

# Needed for running chainlit code if using Mock tests
RUN pip3 install chainlit==1.1.305

# Needed for promptflow connections
RUN pip install keyrings.alt

WORKDIR /app/chainlit-ui-evaluation

# Set up Connections
RUN pf connection create --file ./openai.yaml --set api_key=$OPENAI_API_KEY  --name open_ai_connection 
RUN pf connection create --file ./azure_openai.yaml --set api_key=$OPENAI_API_KEY --set api_base=$OPENAI_API_ENDPOINT --name azure_openai 
