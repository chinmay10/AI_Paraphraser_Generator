FROM node:16.17.0

# Set the working directory
WORKDIR /app

# Copy the package.json and package-lock.json files
COPY package*.json ./

# Install the dependencies
RUN npm install

# Copy the rest of the source code
COPY . .

# Build the React app
RUN npm run build

# Expose the port that the React app is running on
EXPOSE 3000

# Run the React development server
CMD ["npm", "start"]
