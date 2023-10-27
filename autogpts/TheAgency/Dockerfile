ARG BASE_IMAGE
FROM ${BASE_IMAGE} 

# Set work directory in the container
WORKDIR /app

## Install system dependencies
#RUN apt-get update \
#    && apt-get install -y build-essential curl ffmpeg \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PIP_NO_CACHE_DIR=yes \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=0 \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"



RUN pip3 install poetry

# FIXME to make this parameterized
COPY autogpts/ZEROAGPT_03/pyproject.toml  /app/
COPY autogpts/ZEROAGPT_03/poetry.lock     /app/

# Project initialization:
RUN poetry install --no-interaction --no-ansi

ENV PYTHONPATH="/app:$PYTHONPATH"

# Copy project
#COPY . /app
ADD ./autogpts/ZEROAGPT_03/ /app/

RUN mkdir /tmp/workspace/

# Make port 80 available to the world outside this container
#EXPOSE 8000

# Run the application when the container launches
#CMD ["poetry", "run", "python", "autogpt/__main__.py"]