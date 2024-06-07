FROM python:3.1.4

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ui/recipes-chat-chainlit/ /app/
WORKDIR /app

COPY ./utils/ ./utils
COPY ./templates ./templates

RUN mkdir recepes
RUN mkdir recepes/images

CMD ["chainlit", "run", "app.py", "--port", "8000"]
