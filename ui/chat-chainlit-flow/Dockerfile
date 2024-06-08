FROM python:3.11.4

COPY ui/chat-chainlit-flow /app
COPY requirements.txt /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY ./utils/ ./utils
COPY ./templates ./templates
#
RUN mkdir recepes
RUN mkdir recepes/images

CMD ["chainlit", "run", "app.py", "--port", "8000", "-w"]
