FROM node:18-alpine

WORKDIR /app

# Copy the entire project first
COPY . .

# Install dependencies
RUN npm install --legacy-peer-deps

# Build the Next.js application
RUN npm run build

# Expose the port
ENV PORT=3000
EXPOSE 3000

# Start the application
CMD ["npm", "start"] 