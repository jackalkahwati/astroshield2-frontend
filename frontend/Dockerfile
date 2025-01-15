# Build Stage
FROM node:18-alpine AS builder
WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm install

# Copy the rest of the application
COPY . .

# Ensure public directory exists
RUN mkdir -p public

# Build the application
RUN npm run build

# Production Stage
FROM node:18-alpine
WORKDIR /app

# Copy necessary files from builder stage
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

# Set environment variables
ENV NODE_ENV production
ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

EXPOSE 3000

CMD ["npm", "start"] 