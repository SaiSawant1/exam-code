
# Use the official Golang image to build the app
FROM golang:1.23.2-bookworm

# Install make and curl to download migration tool
RUN apt-get update && apt-get install -y make curl

# Set the working directory
WORKDIR /app

# Copy everything to the container
COPY . .

# Download Go modules
RUN go mod download

# Tidy up Go modules
RUN go mod tidy


# Build the application
RUN make build

# Expose the port your Go application will listen on
EXPOSE 8080

# Run migrations and then start the application
CMD ["./bin/exam"]

