FROM node:18-alpine

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./
RUN npm install --legacy-peer-deps

# Copy the rest of the application
COPY . .

# Install TypeScript globally
RUN npm install -g typescript

# Build the Next.js application
RUN npm run build

# Expose the port
ENV PORT=3000
EXPOSE 3000

# Start the application
CMD ["npm", "start"] 