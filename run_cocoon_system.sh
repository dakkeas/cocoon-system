#!/bin/bash

echo "Starting Cocoon System..."

# Go to project root
cd /home/raspberrypi/cocoon_system || exit

# Activate virtual environment
source env/bin/activate

echo "Virtual environment activated"

# Start Flask server
python app/server/app.py > flask.log 2>&1 &

# Start main controller
python main.py > main.log 2>&1 &

# Start frontend (npm)
cd app/client || exit
npm run dev > client.log 2>&1 &

# Wait for all processes
wait