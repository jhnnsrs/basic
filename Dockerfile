FROM python:3.9

# Install dependencies
RUN pip install poetry rich
ENV PYTHONUNBUFFERED=1

# Copy dependencies
COPY pyproject.toml README.md /
RUN poetry config virtualenvs.create false 
RUN poetry install


# Install Application
RUN mkdir /workspace
ADD . /workspace
WORKDIR /workspace


CMD bash run.sh

