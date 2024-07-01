#!bin/bash

kill -9 $(ps -aux | grep redis-server | awk '{print $2}')
kill -9 $(ps -aux | grep run.py | awk '{print $2}')
