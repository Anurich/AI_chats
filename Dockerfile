FROM python:3.12

WORKDIR /code

COPY ./requirements.txt ./requirements.txt 
RUN apt-get update && apt-get install -y \
   wkhtmltopdf \
   && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

RUN chmod 755 /code/start.sh

#CMD ["sh", "start.sh"]
