# Briloop
* Briloop is a tool to convert bril programs into briloop format, converting unstructured control flow into structured control flow, using Ramsey's BeyondRelooper Algorithm. 

## Usage
* Pass a Bril JSON program to standard input, you'll obtain a Briloop JSON program in standard output. 
```
cat core/ackermann.bril | java -jar briloop.jar | ./run.sh
```
* `./briloop.sh` is an alias for the java command so you can also do: 
```
cat core/ackermann.bril | ./briloop | ./run.sh
```

## Requirements
* Java 21

## Installation
* Run `./install.sh` (requires sudo) will create a symbolic link in /usr/local/bin

## Building
* Run `./setup.sh` to download jar dependencies. 
* Run `./build.sh` to compile the project. 
* Run `./run.sh` to run the transformation.
* Run `./release.sh` to generate the release jar.


## Dependencies
* [Moshi](https://github.com/square/moshi): For json parsing. 
* [Okio](https://github.com/square/okio): For json parsing.
