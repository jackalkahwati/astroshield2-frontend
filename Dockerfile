FROM node:18-alpine

WORKDIR /app

# Install dependencies first for better caching
COPY package.json package-lock.json ./
RUN npm install --legacy-peer-deps

# Create necessary directories
RUN mkdir -p src/lib

# Copy the API file first
COPY src/lib/api.ts src/lib/

# Copy the rest of the application
COPY . .

# Build the Next.js application
RUN npm run build

# Expose the port
ENV PORT=3000
EXPOSE 3000

# Start the application
CMD ["npm", "start"] 