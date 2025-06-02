clear
echo "build starting"
KOTLINC=./lib/kotlinc/bin/kotlinc

MOSHI_KOTLIN_JAR=./lib/moshi-kotlin.jar
MOSHI_JAR=./lib/moshi.jar
OKIO_JAR=./lib/okio.jar

kotlinc \
        -cp $MOSHI_JAR:$OKIO_JAR:$MOSHI_KOTLIN_JAR \
        -d build \
        *.kt

echo "build done"
