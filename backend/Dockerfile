FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the Django code into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Run Django's migrate command to create the database tables
RUN python manage.py makemigrations && python manage.py migrate

# Expose the correct port
EXPOSE 8000

# Run the Django server
CMD ["gunicorn", "paraphraser.wsgi", "-b", "0.0.0.0:8000"]