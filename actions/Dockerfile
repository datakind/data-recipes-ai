FROM debian:bookworm-slim

RUN echo "Hi!"

ARG DATA_DB_CONN_STRING
ENV DATA_DB_CONN_STRING=$DATA_DB_CONN_STRING

# Setup Nginx and Supervisor
RUN apt-get update && apt-get install -y procps nginx supervisor && \
    rm -rf /var/lib/apt/lists/*

COPY actions/config/nginx.conf /etc/nginx/nginx.conf
COPY actions/config/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Setup user and permissions
RUN useradd -m as-user
RUN chown -R as-user:as-user /var/log /run /var/lib/nginx

# Setup workspace
RUN mkdir -p /action-server/datadir /action-server/actions
RUN chown -R as-user:as-user /action-server

WORKDIR /action-server/actions

# Setup Action Server
ADD https://downloads.robocorp.com/action-server/releases/latest/linux64/action-server /usr/local/bin/action-server
RUN chmod +x /usr/local/bin/action-server

COPY ./actions .

RUN mkdir -p /action-server/actions/actions_plugins/recipe-server/images

RUN chown -R as-user:as-user /action-server

RUN chown -R as-user:as-user /action-server/actions/actions_plugins/recipe-server/images

USER as-user

ARG DATA_DB_CONN_STRING
ENV DATA_DB_CONN_STRING=$DATA_DB_CONN_STRING

# Load individually
RUN action-server import --dir=./actions_plugins/postgres-universal --datadir=/action-server/datadir
RUN action-server import --dir=./actions_plugins/recipe-server --datadir=/action-server/datadir

RUN echo "{\"dsn\": \"$DATA_DB_CONN_STRING\"}" > ./actions_plugins/postgres-universal/postgres_data_connection.json

RUN echo "{\"dsn\": \"$RECIPE_DB_CONN_STRING\"}" > ./actions_plugins/postgres-universal/postgres_recipe_connection.json

EXPOSE 8080

CMD ["/usr/bin/supervisord"]