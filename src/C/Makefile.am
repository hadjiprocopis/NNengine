##              ARTIFICIAL NEURAL NETWORKS SOFTWARE
##
##   An Error Back Propagation Neural Network Engine for Feed-Forward
##                      Multi-Layer Neural Networks
##
##                        by Andreas Hadjiprocopis
##                  (livantes@soi.city.ac.uk)
##              Copyright Andreas Hadjiprocopis, 1994,2007

OPTIONS_AUTOMAKE=gnu
ACLOCAL_AMFLAGS=-I ./m4

bin_PROGRAMS = NNengine ForwardPass

NNengine_SOURCES = NNengine.c BPNetworkError.h BPNetworkConstants.h BPNetworkDefinitions.h BPNetworkDeclarations.h
ForwardPass_SOURCES = ForwardPass.c BPNetworkError.h BPNetworkConstants.h BPNetworkDefinitions.h BPNetworkDeclarations.h

noinst_LIBRARIES = libnnengine.a
libnnengine_a_SOURCES = BPNetworkConstants.h\
			 BPNetworkDefinitions.h\
			 BPNetworkDeclarations.h\
			 BPNetworkStandardInclude.h\
			 Random.c\
			 CreateBPNetwork.c\
			 OperateBPNetwork.c

NNengine_LDADD = -lm libnnengine.a
ForwardPass_LDADD = -lm libnnengine.a
