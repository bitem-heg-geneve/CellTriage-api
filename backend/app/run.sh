#!/bin/bash

# Heroku postgres addon
export SQLALCHEMY_DATABASE_URI=${DATABASE_URL}

# If there's a prestart.sh script in the /app directory or other path specified, run it before starting
PRE_START_PATH=${PRE_START_PATH:-/app/prestart.sh}
echo "Checking for script in $PRE_START_PATH"
if [ -f $PRE_START_PATH ] ; then
    echo "Running script $PRE_START_PATH"
    . "$PRE_START_PATH"
else
    echo "There is no script $PRE_START_PATH"
fi

export APP_MODULE=${APP_MODULE-app.main:app}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8001}
export BACKEND_CORS_ORIGINS=${BACKEND_CORS_ORIGINS}
export DEBUG_PORT=${DEBUG_PORT:-5678}


# run gunicorn
# exec gunicorn --bind $HOST:$PORT "$APP_MODULE" -k uvicorn.workers.UvicornWorker

# python /tmp/debugpy --listen $HOST:$DEBUG_PORT -m gunicorn --bind $HOST:$PORT "$APP_MODULE" -k uvicorn.workers.UvicornWorker  --reload
pip install debugpy -t /tmp
python /tmp/debugpy --listen 0.0.0.0:5678 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
# python debugpy --listen 0.0.0.0:5678 -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload