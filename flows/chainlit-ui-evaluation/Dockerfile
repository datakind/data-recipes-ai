# See here: https://microsoft.github.io/promptflow/cloud/azureai/runtime-change-log.html#v4
FROM mcr.microsoft.com/azureml/promptflow/promptflow-runtime-stable:20240411.v4 

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

WORKDIR /app/chainlit-ui-evaluation

# Needed for running chainlit code if using Mock tests
RUN pip3 install chainlit==1.1.305
RUN pip3 install langchain==0.2.1
RUN pip3 install langchain_openai==0.1.7
RUN pip3 install psycopg2_binary==2.9.9
RUN pip3 install keyrings.alt

# Set up Connections
RUN pf connection create --file ./openai.yaml --set api_key=$OPENAI_API_KEY  --name open_ai_connection 
RUN pf connection create --file ./azure_openai.yaml --set api_key=$OPENAI_API_KEY --set api_base=$OPENAI_API_ENDPOINT --name azure_openai 

RUN echo "DEBUG DOCKER"
RUN which python
RUN env

# # ======= SELENIUM BEGIN ========

# Update package lists for the Ubuntu system
RUN apt-get update

# Download Chrome Debian package
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb 

# Install the Chrome Debian package
RUN apt install ./google-chrome-stable_current_amd64.deb -y

# Print the version of Google Chrome installed
RUN google-chrome --version

RUN pip install selenium==4.22.0
RUN pip install chromedriver-py==126.0.6478.126

# # ======= SELENIUM END ========

ENV PATH=$PATH:/azureml-envs/prompt-flow/runtime/bin:/opt/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin