#!/bin/bash -eu

PROCESS_ID=$( jps -l | grep com.mina.examples.mnist.MNistTraining | awk '{print $1}' )

OUT_DIR=/var/log/neuralnetwork/thread-dumps

if [ ! -d $OUT_DIR ]; then
  mkdir -p $OUT_DIR
fi



while true
do

  echo Delete aged files of 2 days or more ...
  find $OUT_DIR -mindepth 1 -mtime +2 -delete

  FILE_NAME="$OUT_DIR/threaddump-$PROCESS_ID--$( date +'%Y-%m-%d-%H:%M:%S' ).out"
  echo Generate thread-dump for $PROCESS_ID ...
  jstack $PROCESS_ID > ${FILE_NAME}

  # sleep 5 minutes
  echo Sleep 5 minutes
  sleep 5m

#  -mindepth 1: without this, . (the directory itself) might also match and therefore get deleted.
#  -mtime +5: process files whose data was last modified 5*24 hours ago.

done

# delete aged file of 2 days or more




