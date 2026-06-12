# Use an official Ubuntu Linux base image
FROM ubuntu:22.04

# Stop interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install g++ compiler, curl, and network tools
RUN apt-get update && apt-get install -y \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy your source code into the container
COPY main.cpp httplib.h index.html ./

# Compile the C++ code (Linux uses -lpthread instead of -lws2_32)
RUN g++ -std=c++17 -O2 main.cpp -o db -lpthread

# Expose port 8080 so the outside world can access the web server
EXPOSE 8080

# Command to run when the container starts
CMD ["./db"]
