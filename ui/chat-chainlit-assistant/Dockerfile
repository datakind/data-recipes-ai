FROM python:3.11.4

COPY ui/chat-chainlit-assistant /app
COPY requirements.txt /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./utils/ ./utils
COPY ./templates ./templates
#
RUN mkdir recepes
RUN mkdir recepes/images

CMD ["chainlit", "run", "app.py", "--port", "8000", "--watch"]
