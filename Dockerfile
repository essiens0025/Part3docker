# python base image in the container from Docker Hub
FROM python:3.8.12-buster

# copy files to the /app folder in the container
COPY faster.py /app/faster.py
COPY Pipfile /app/Pipfile
COPY Pipfile.lock /app/Pipfile.lock
COPY model.py /app/model.py
COPY iris_model.pkl /app/iris_model.pkl


# set the working directory in the container to be /app
WORKDIR /app


# install the packages from the Pipfile in the container
RUN pip install pipenv
RUN pipenv install --system --deploy --ignore-pipfile


# execute the command python main.py (in the WORKDIR) to start the app
CMD uvicorn faster:app --host 0.0.0.0 --port $PORT
