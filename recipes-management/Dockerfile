FROM python:3.11.4

WORKDIR /app

COPY ./recipes-management/requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python", "./your-script.py"]

#COPY ./assistant/recipes_management .

CMD ["tail", "-f", "/dev/null"]
