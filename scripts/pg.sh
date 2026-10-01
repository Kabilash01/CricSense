#!/usr/bin/env bash
# A project-local PostgreSQL for CricSense: no root, data outside the repo.
#   scripts/pg.sh start | stop | status
# Data: ${CRICSENSE_PGDATA:-$HOME/.local/share/cricsense/pg}, port 54329, database/user cricsense.
set -euo pipefail
DATA="${CRICSENSE_PGDATA:-$HOME/.local/share/cricsense/pg}"
PORT=54329
case "${1:-status}" in
  start)
    if [ ! -f "$DATA/PG_VERSION" ]; then
      mkdir -p "$DATA"
      initdb -D "$DATA" -U cricsense --auth=trust >/dev/null
      echo "initialised $DATA"
    fi
    pg_ctl -D "$DATA" -o "-p $PORT -k /tmp -c listen_addresses=localhost" -l "$DATA/server.log" -w start >/dev/null
    psql -h localhost -p $PORT -U cricsense -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='cricsense'" | grep -q 1 \
      || createdb -h localhost -p $PORT -U cricsense cricsense
    echo "postgres on localhost:$PORT, database cricsense";;
  stop) pg_ctl -D "$DATA" -m fast stop;;
  status) pg_ctl -D "$DATA" status || true;;
  *) echo "usage: $0 start|stop|status"; exit 2;;
esac
