FROM python:3

WORKDIR /app
EXPOSE 5000

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
RUN chmod +x ./init.sh

ENTRYPOINT ["./init.sh"]
