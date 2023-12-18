# ARG PYTHON_VERSION=3.10.12
# FROM python:${PYTHON_VERSION}-slim as base
FROM python:3.10
# Prevents Python from writing pyc files.
# ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
# ENV PYTHONUNBUFFERED=1

WORKDIR /mapmatching

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Switch to the non-privileged user to run the application.
# USER root

# Copy the source code into the container.
COPY . /mapmatching

# Expose the port that the application listens on.
# EXPOSE 8899

# Run the application.
CMD python3 mapmatching.py