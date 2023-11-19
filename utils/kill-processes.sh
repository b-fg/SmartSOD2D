#!bin/bash

kill -9 $(ps -aux | grep redis-server | awk '{print $2}')
kill -9 $(ps -aux | grep control | awk '{print $2}')
kill -9 $(ps -aux | grep train | awk '{print $2}')
kill -9 $(ps -aux | grep eval | awk '{print $2}')
kill -9 $(ps -aux | grep restart_step | awk '{print $2}')
kill -9 $(ps -aux | grep gmsh2sod2d | awk '{print $2}')
