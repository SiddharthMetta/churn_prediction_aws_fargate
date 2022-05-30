FROM python:3.8-slim
COPY ./* ./app/

# Install OpenJDK-11
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;

WORKDIR /app/
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]