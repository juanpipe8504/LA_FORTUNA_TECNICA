# syntax=docker/dockerfile:1

# ---- Build stage ----------------------------------------------------------
FROM node:22-alpine AS build
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .
# Relative by default so nginx can proxy /api to the backend container.
ARG VITE_API_URL=/api/v1
ARG VITE_MAX_UPLOAD_MB=50
ENV VITE_API_URL=${VITE_API_URL} \
    VITE_MAX_UPLOAD_MB=${VITE_MAX_UPLOAD_MB}
RUN npm run build

# ---- Runtime stage --------------------------------------------------------
FROM nginx:1.27-alpine AS runtime
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1/healthz || exit 1
