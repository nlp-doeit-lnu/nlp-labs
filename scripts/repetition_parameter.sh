#!/bin/bash

#exec /opt/jdk-18/bin/java -Xmx4G --module-path "/opt/javafx-sdk-17.0.12/lib" --add-modules javafx.controls,javafx.fxml -jar /mnt/jars/projFedorova3_2023.jar
exec /opt/jdk-18/bin/java -Djava.library.path="/opt/javafx-sdk-17.0.12/lib" -Xmx4G --module-path "/opt/javafx-sdk-17.0.12/lib" --add-modules javafx.controls,javafx.fxml -jar /mnt/jars/projFedorova3_2023.jar
