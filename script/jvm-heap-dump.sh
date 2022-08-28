#!/bin/bash -eu

PROCESS_ID=$( jps -l | grep com.mina.examples.mnist.MNistTraining | awk '{print $1}' )

OUT_DIR=/var/log/neuralnetwork/heap-dumps

if [ ! -d $OUT_DIR ]; then
  mkdir -p $OUT_DIR
fi



while true
do

  echo Delete aged files of 2 days or more ...
  find $OUT_DIR -mindepth 1 -mtime +2 -delete

  FILE_NAME="$OUT_DIR/heapdump-$PROCESS_ID--$( date +'%Y-%m-%d-%H:%M:%S' ).hprof"
  echo Generate heap-dump for $PROCESS_ID ...
  jmap -dump:live,format=b,file=${FILE_NAME} $PROCESS_ID

  # we can use jcmd as well
  # jcmd $PROCESS_ID GC.heap_dump ${FILE_NAME}

  # sleep 5 minutes
  echo Sleep 5 minutes
  sleep 5m

#  -mindepth 1: without this, . (the directory itself) might also match and therefore get deleted.
#  -mtime +5: process files whose data was last modified 5*24 hours ago.

done

# delete aged file of 2 days or more




