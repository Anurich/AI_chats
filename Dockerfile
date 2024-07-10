FROM python:3.12.3

# Set the working directory
WORKDIR /code

# Install MuPDF and poppler-utils dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    mupdf-tools \
    libmupdf-dev \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set environment variable for shared libraries
ENV LD_LIBRARY_PATH=/usr/local/lib/

# Copy requirements.txt file and install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the entire current directory into the Docker image
COPY . /code

# Set execute permissions for start.sh if needed
RUN chmod +x /code/start.sh

# Define the command to run your start.sh script
CMD ["sh", "start.sh"]
