#!/bin/bash -eu


VM_OPTIONS="-ea -Xms4G -Xmx5G -XX:+UseG1GC -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/var/log/neuralnetwork -Xlog:gc*:file=/var/log/neuralnetwork/gc.log:time:filecount=7,filesize=8M -XX:NativeMemoryTracking=detail"

JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))


compile() {

  echo "Compiling ...."
  cd /home/mina/Desktop/NeuralNetwork && mvn clean package -DskipTests

  echo "Compilation done!"
}

if [ ! -f /home/mina/Desktop/NeuralNetwork/target/NeuralNetwork-1.0-SNAPSHOT-jar-with-dependencies.jar ]; then
  compile
fi

compile

$JAVA_HOME/bin/java $VM_OPTIONS -cp /home/mina/Desktop/NeuralNetwork/target/NeuralNetwork-1.0-SNAPSHOT-jar-with-dependencies.jar com.mina.examples.mnist.MNistTraining



