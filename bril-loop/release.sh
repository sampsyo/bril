#!/bin/bash
set -e
set -x
echo "Creating briloop jar"
KOTLINC=./lib/kotlinc/bin/kotlinc

MOSHI_KOTLIN_JAR=lib/moshi-kotlin.jar
MOSHI_JAR=lib/moshi.jar
OKIO_JAR=lib/okio.jar

mkdir build-release

kotlinc \
        -cp $MOSHI_JAR:$OKIO_JAR:$MOSHI_KOTLIN_JAR \
        -include-runtime \
        -d build/temp.jar \
        *.kt
cd build-release
jar xf ../build/temp.jar
jar xf ../$MOSHI_KOTLIN_JAR
jar xf ../$MOSHI_JAR
jar xf ../$OKIO_JAR
jar cfe briloop.jar BeyondRelooperKt .
cd ..
cp build-release/briloop.jar .
echo "briloop.jar ready"

