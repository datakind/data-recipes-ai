FROM ghcr.io/danny-avila/librechat
#FROM ghcr.io/danny-avila/librechat-dev:latest
#FROM ghcr.io/danny-avila/librechat:v0.7.0
RUN echo "Hi! "
COPY .env /app/.env
COPY ./ui/recipes-chat/librechat.yaml /app/librechat.yaml 
COPY ./ui/recipes-chat/tools /app/api/app/clients/tools 
COPY ./ui/recipes-chat/nginx.conf /app/client/nginx.conf

