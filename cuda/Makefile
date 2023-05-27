EXE_NAME = main
SRC = $(wildcard *.cu)

CXX = nvcc
EXE = ${EXE_NAME}.cuda
CXXFLAGS = -O3 --gpu-architecture=sm_60
LINK = ${CXX}

build: ${EXE}

${EXE}: ${SRC}
	${LINK} ${CXXFLAGS} ${SRC} -o ${EXE}

clean:
	rm -f ${EXE}

test: ${EXE}
	./${EXE}