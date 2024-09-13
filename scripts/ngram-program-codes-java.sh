#!/bin/bash

exec /opt/jdk-18/bin/java -Xmx4G --module-path "/opt/javafx-sdk-17.0.12/lib" --add-modules javafx.controls,javafx.fxml -jar /mnt/jars/NGram-1.0-SNAPSHOT.jar
