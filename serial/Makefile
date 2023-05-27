EXE_NAME = main
SRC = $(wildcard *.cpp)

CXX = g++
EXE = ${EXE_NAME}.serial
CXXFLAGS = -O3
LINK = ${CXX}

build: ${EXE}

${EXE}: ${SRC}
	${LINK} ${CXXFLAGS} ${SRC} -o ${EXE}

clean:
	rm -f ${EXE}

test: ${EXE}
	./${EXE}