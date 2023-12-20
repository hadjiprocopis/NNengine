/************************************************************************
 *									*
 *		ARTIFICIAL NEURAL NETWORKS SOFTWARE			*
 *									*
 *   An Error Back Propagation Neural Network Engine for Feed-Forward	*
 *			Multi-Layer Neural Networks			*
 *									*
 *			by Andreas Hadjiprocopis			*
 *		    (andreashad2@gmail.com)				*
 *		Copyright Andreas Hadjiprocopis, 1994			*
 *									*
 ************************************************************************/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "BPNetworkError.h"
#include "BPNetworkConstants.h"
#include "BPNetworkDefinitions.h"

int	IsEmptyString(char *str);
int	Usage(char *app_name);
int	read_line_slow_but_safe(FILE *fp, BPPrecisionType *line, int num_elements, char *dummy_buffer);
int	read_line_fast_but_risky(FILE *fp, BPPrecisionType *line, int num_elements, char *dummy_buffer);

int	main(int argc, char **argv)
{
	FILE		*FileOfInputs = NULL;
	register int	i, j, k;
	int		NumberOfLayers = 0, *Geometry = NULL, arch_start, SigmoidOutputFlag = FALSE,
			DerivativesFlag = FALSE, NumberOfInputVectors = 0, ShowInputsFlag = FALSE,
			flag = TRUE, StartVector = 1, StopVector = -1,
			ShowMemoryFlag = FALSE, ShowLineNumbersFlag = FALSE,
			stav, stov, num_discrete_output_map = 0;
	char		*buffer, InputFilename[512], WeightsFilename[512],
			**discrete_output_map = NULL;
	BPPrecisionType	SingleExemplarError = 0.0, AllExemplarsError = 0.0, dummy,
			*InputVector,
			Beta = DEFAULT_BETA_VALUE, Lamda = DEFAULT_LAMDA_VALUE;
	BPNetwork       TheBPNetwork;
	/* this pointer to function points to one or the other read_line functions
	   depending on the specific input parameter */
	int	(*read_line)(FILE *, BPPrecisionType *, int, char *) = read_line_slow_but_safe;

	InputFilename[0] = WeightsFilename[0] = '\0';

	TotalMemoryAllocated = 0;

	/* Get the commands from the command line */
	for(i=1;(i<argc)&&flag;i++){
		if( !strncmp(argv[i], "-input", 2) ){
			if( ++i < argc ) strcpy(InputFilename, argv[i]);
			else flag = Usage(argv[0]);
			continue;
		}
		if( !strncmp(argv[i], "-exemplars", 2) ){
			if( ++i < argc ) NumberOfInputVectors = atoi(argv[i]);
			else flag = Usage(argv[0]);
		}
		if( !strncmp(argv[i], "-weights", 2) ){
			if( ++i < argc ) strcpy(WeightsFilename, argv[i]);
			else flag = Usage(argv[0]);
			continue;
		}
		if( !strncmp(argv[i], "-sigmoid", 3) ){
			SigmoidOutputFlag = TRUE;
			continue;
		}
		if( !strncmp(argv[i], "-discrete_output_map", 19) ){
			if( ++i < argc ){
				char	*dop = strdup(argv[i]), *save, *token;
				int	tki;
				for(num_discrete_output_map=0;(token=strtok_r(dop, " \t", &save))!=NULL;dop=NULL,num_discrete_output_map++);
				dop = strdup(argv[i]);
				if( (discrete_output_map=(char **)malloc(num_discrete_output_map*sizeof(char *))) == NULL ){ fprintf(stderr, "%s : malloc failed for %ld bytes for char (discrete_output_map).\n", argv[0], num_discrete_output_map*sizeof(char *)); exit(1); }
				for(tki=0;(token=strtok_r(dop, " \t", &save))!=NULL;dop=NULL,discrete_output_map[tki++]=strdup(token));
			} else flag = Usage(argv[0]);
			//for(j=0;j<num_discrete_output_map;j++){ printf("xx=%s\n", discrete_output_map[j]); }
			continue;
		}
		if( !strncmp(argv[i], "-derivatives", 2) ){
			DerivativesFlag = TRUE;
			continue;
		}
		if( !strncmp(argv[i], "-show", 3) ){
			ShowInputsFlag = TRUE;
			continue;
		}
		if( !strncmp(argv[i], "-start", 4) ){
			if( ++i < argc ) StartVector = MAX(1, atoi(argv[i]));
			else flag = Usage(argv[0]);
			continue;
		}
		if( !strncmp(argv[i], "-stop", 4) ){
			if( ++i < argc ) StopVector = atoi(argv[i]);
			else flag = Usage(argv[0]);
			continue;
		}
		if( !strncmp(argv[i], "-usage", 2) || !strncmp(argv[i], "-help", 2) ){
			flag = Usage(argv[0]);
			continue;
		}
		if( !strncmp(argv[i], "-memory", 2) ){
			ShowMemoryFlag = TRUE;
			continue;
		}
		if( !strncmp(argv[i], "-fast", 2) ){
			read_line = read_line_fast_but_risky;
			continue;
		}
		if( !strncmp(argv[i], "-line_numbers", 2) ){
			ShowLineNumbersFlag = TRUE;
			continue;
		}
		if( !strncmp(argv[i], "-arch", 2) ){
			arch_start = ++i;
			while( (i<argc) && (argv[i][0] != '-') ) i++;
			if( (NumberOfLayers=i-arch_start) < 3 ){
				fprintf(stderr, "%s: Error, the total number of layers must be greater than 2.\a\n", argv[0]);
				flag = Usage(argv[0]);
				continue;
			}
			if( (Geometry=(int *)calloc(NumberOfLayers, sizeof(int))) == NULL ){
				fprintf(stderr, "%s: Error, could not allocate memory for %d int, Geometry.\n", argv[0], NumberOfLayers);
				exit(1);
			}
			TotalMemoryAllocated += NumberOfLayers * sizeof(int);
			for(j=arch_start;j<i;j++) Geometry[j-arch_start] = atoi(argv[j]);
			i--;
			continue;
		}
	}
	/* Check if the required parameters have been supplied */
	if( (InputFilename[0]=='\0') || (WeightsFilename[0]=='\0') || !Geometry || !flag){
		if( flag ) Usage(argv[0]); exit(1);
	}

	if( (StopVector>0) && (StartVector > StopVector) ){
		fprintf(stderr, "%s: Error, the starting vector was after the stop vector.\a\n", argv[0]);
		exit(1);
	}

	/* Start main */
	/* Allocate memory for the Array to hold ONE line of the Exemplars */
	if( (InputVector=(BPPrecisionType *)calloc(Geometry[0], sizeof(BPPrecisionType))) == NULL ){
		fprintf(stderr, "Could not allocate memory for %d float, InputVector.\n", Geometry[0]);
		free(Geometry);
		exit(1);
	}
	TotalMemoryAllocated += Geometry[0] * sizeof(BPPrecisionType);
	if( (buffer=(char *)calloc(MAX_CHARS_PER_LINE+2, sizeof(char))) == NULL ){
		fprintf(stderr, "Could not allocate memory for %d char, buffer.\n", MAX_CHARS_PER_LINE+2);
		free(Geometry); free(InputVector);
		exit(1);
	}
	TotalMemoryAllocated += (MAX_CHARS_PER_LINE+2) * sizeof(char);		

	/* Malloc and Initialise the BP Network */
	/* MALLOC */
	if( (TheBPNetwork=(BPNetwork )calloc(1, sizeof(struct _BACKPROP_NET_STRUCT))) == NULL ){
		fprintf(stderr, "Could not allocate memory for the BP Network.\n");
		free(Geometry); free(InputVector); free(buffer);
		exit(1);
	}
	TotalMemoryAllocated += sizeof(struct _BACKPROP_NET_STRUCT);

	/* CREATE */	
	TheBPNetwork->IndividualError = (BPPrecisionType *)NULL;
	TheBPNetwork->NetworkMap = (Neuron **)NULL;
	CreateBPNetwork( NumberOfLayers, Geometry, TheBPNetwork );

	/* INITIALISE */
	InitialiseBPNetwork( TheBPNetwork );

	if( SigmoidOutputFlag ) TheBPNetwork->OutputType = Sigmoid; else TheBPNetwork->OutputType = Linear;

	/* Fill in the Weights */
	if( LoadWeightsFromFile(TheBPNetwork, WeightsFilename, (FILE *)NULL, TRUE) != SUCCESS ){
		fprintf(stderr, "%s : call to LoadWeightsFromFile has failed for file '%s'.\n", argv[0], WeightsFilename);
		exit(1);
	}

	if( TheBPNetwork->NetworkType == Continuous ){
/*		fprintf(stderr, "Continuous Mode\n");*/
		if( num_discrete_output_map > 0 ){
			fprintf(stderr, "%s : the -discrete_output_map option is incompatible with continuous network type, i.e. first class, last class and class separation must apply and these are set during training time and are saved/read to/from the last-but-one line of the weights file.\n", argv[0]);
			DestroyBPNetwork(TheBPNetwork); free(Geometry); free(InputVector); free(buffer);
			exit(1);
		}
		TheBPNetwork->LastClassAt = 0.0;
	} else {
/*		fprintf(stderr, "Discrete Mode\n");*/
		if( (num_discrete_output_map>0) && (num_discrete_output_map != TheBPNetwork->NumberOfOutputClasses) ){
			fprintf(stderr, "%s : the -discrete_output_map option requires as many strings/chars (%d) as there are output classes (%d), exactly.\n", argv[0], num_discrete_output_map, TheBPNetwork->NumberOfOutputClasses);
			DestroyBPNetwork(TheBPNetwork); free(Geometry); free(InputVector); free(buffer);
			exit(1);
		}
		TheBPNetwork->LastClassAt = TheBPNetwork->FirstClassAt + TheBPNetwork->ClassSeparation*((BPPrecisionType )(TheBPNetwork->NumberOfOutputClasses-1));
	}
	if( (FileOfInputs=fopen(InputFilename, "r")) == NULL ){
		fprintf(stderr, "%s: Error, could not open file '%s' for reading.\n", argv[0], InputFilename);
		exit(1);
	}
	/* skip the first StartVector-1 vectors if specified on command line */
	NumberOfInputVectors = 0;
	stav = StartVector - 1; stov = StopVector - 1;
	for(i=0;i<stav;i++){
		if( read_line(FileOfInputs, &(InputVector[0]), Geometry[0], buffer) == FALSE ){
			fprintf(stderr,"%s : error, there are %d lines in the input file (%s), recheck your start and stop vector numbers (%d and %d)/1.\n", argv[0], NumberOfInputVectors, InputFilename, StartVector, StopVector);
			fclose(FileOfInputs);
			DestroyBPNetwork(TheBPNetwork);
			free(Geometry); free(InputVector); free(buffer);
			exit(1);
		}
		NumberOfInputVectors++;
	}
	if( (StopVector>0) && (i > stov) ){
		fprintf(stderr,"%s : error, there are %d lines in the input file (%s), recheck your start and stop vector numbers (%d and %d)/2.\n", argv[0], NumberOfInputVectors, InputFilename, StartVector, StopVector);
		fclose(FileOfInputs);
		DestroyBPNetwork(TheBPNetwork);
		free(Geometry); free(InputVector); free(buffer);
		exit(1);
	}
	for(i=stav;;i++){
		if( read_line(FileOfInputs, &(InputVector[0]), Geometry[0], buffer) == FALSE ){
			if( (StopVector>0) && (i<=stov) ){
				fprintf(stderr,"%s : error, there are %d lines in the input file (%s), recheck your start and stop vector numbers (%d and %d)/3.\n", argv[0], NumberOfInputVectors, InputFilename, StartVector, StopVector);
				fclose(FileOfInputs);
				DestroyBPNetwork(TheBPNetwork);
				free(Geometry); free(InputVector); free(buffer);
				exit(1);
			}
			if( !feof(FileOfInputs) ){
				fprintf(stderr,"%s : error, read %d lines of input from file '%s' and encountered an error, the end of file has not yet been reached.\n", argv[0], NumberOfInputVectors, InputFilename);
				fclose(FileOfInputs);
				DestroyBPNetwork(TheBPNetwork);
				free(Geometry); free(InputVector); free(buffer);
				exit(1);
			}
			/* it's the end of the input file */
			break;
		}
		BPNetworkFeedInputs(Geometry[0], InputVector, TheBPNetwork);
		BPNetworkForwardPropagate(TheBPNetwork);
		if( ShowLineNumbersFlag )
			printf("%d\t", i+1);
		if( ShowInputsFlag ){
			for(j=0;j<Geometry[0];j++){
				printf("%4.3lf ", InputVector[j]);
			}
			printf("\t");
		}
		if( DerivativesFlag ){
			BPNetworkCalculateDerivatives(TheBPNetwork);
			for(j=0;j<Geometry[0]-1;j++){
				printf("%lf ", TheBPNetwork->Derivatives[j]);
			}
			printf("%lf\n", TheBPNetwork->Derivatives[j]);
		} else {
			if( TheBPNetwork->NetworkType == Continuous ){
				for(j=0;j<Geometry[NumberOfLayers-1]-1;j++){
					printf("%lf ", TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output);
				}
				printf("%lf\n", TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output);
			} else {
				if( num_discrete_output_map > 0 ){
					for(j=0;j<Geometry[NumberOfLayers-1]-1;j++){
						//printf("%lf %lf %lf %lf\n", TheBPNetwork->FirstClassAt, TheBPNetwork->LastClassAt, TheBPNetwork->ClassSeparation, TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output); printf("%d\n", (int )DISCRETE_OUTPUT_CLASS_INDEX(TheBPNetwork->FirstClassAt, TheBPNetwork->LastClassAt, TheBPNetwork->ClassSeparation, TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output));
						printf("%s ", discrete_output_map[(int )(DISCRETE_OUTPUT_CLASS_INDEX(TheBPNetwork->FirstClassAt, TheBPNetwork->LastClassAt, TheBPNetwork->ClassSeparation, TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output))]);
					}
					//printf("%lf %lf %lf %lf\n", TheBPNetwork->FirstClassAt, TheBPNetwork->LastClassAt, TheBPNetwork->ClassSeparation, TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output);printf("%d\n", (int )DISCRETE_OUTPUT_CLASS_INDEX(TheBPNetwork->FirstClassAt, TheBPNetwork->LastClassAt, TheBPNetwork->ClassSeparation, TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output));
					printf("%s\n", discrete_output_map[(int )(DISCRETE_OUTPUT_CLASS_INDEX(TheBPNetwork->FirstClassAt, TheBPNetwork->LastClassAt, TheBPNetwork->ClassSeparation, TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output))]);
				} else {
					for(j=0;j<Geometry[NumberOfLayers-1]-1;j++){
						printf("%lf ", TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->DiscreteOutput);
					}
					printf("%lf\n", TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->DiscreteOutput);
				}
			}
		}
		if( (StopVector>0) && (i == stov) ) break;
		NumberOfInputVectors++;
		if( feof(FileOfInputs) ) break;
	}

	if( ShowMemoryFlag ) fprintf(stderr, "%s: %d bytes of memory used.\n", argv[0], TotalMemoryAllocated);

	DestroyBPNetwork(TheBPNetwork);

	free(Geometry); free(InputVector); free(buffer);
	exit(0);
}

int	IsEmptyString(char *str)
{
	int	i = 0;

	while( (str[i]!='\n') && (str[i]!='\0') )
		if( str[i++] != ' ' ) return(FALSE);

	return(TRUE);
}

/* it reads using single fscanf's which means we dont care if the lines in the input file are ending in \n
we just read enough numbers to form one input vector and then continue where we left it */
/* the param dummy_buffer is not required but we use it here because we need to be compatible with
the next read_line* function - do not remove it from the parameters list! */
int	read_line_slow_but_safe(FILE *fp, BPPrecisionType *line, int num_elements, char *dummy_buffer){
	int	i;
	for(i=0;i<num_elements;i++)
		if( fscanf(fp, "%lf", &(line[i])) == EOF ) return(FALSE);
	return(TRUE);
}
/* this one is faster because it reads a single line into memory and then parses from memory the values,
for large files which you are sure each input vector is on its own line */
int	read_line_fast_but_risky(FILE *fp, BPPrecisionType *line, int num_elements, char *dummy_buffer){
	int	i;
	char	*p, *v;

	if( fgets(dummy_buffer, MAX_CHARS_PER_LINE, fp) == NULL ) return FALSE;
	for(i=0,p=&(dummy_buffer[0]);i<num_elements;i++){
		line[i] = strtod(p, &v);
		p = v;
	}
	if( i != num_elements ) return(FALSE);
	return(TRUE);
}


int	Usage(char *app_name)
{
	fprintf(stderr, "Usage: %s options...\n", app_name);
	fprintf(stderr, "Options are (squarish brackets, [], denote optional parameter):\n");
	fprintf(stderr, "  -input filename       The name of the file holding the set of inputs to be fed\n");
	fprintf(stderr, "                        to the neural network. The format of this file is a sequence of\n");
	fprintf(stderr, "                        input vectors. The elements of each vector is separated by space and\n");
	fprintf(stderr, "                        the number of these elements is equal to the number of first layer units\n");
	fprintf(stderr, "                        -- e.g. inputs -- of the neural network.\n");
	fprintf(stderr, "  -weights filename     The name of the file holding the set of weights corresponding to the\n");
	fprintf(stderr, "                        trained state of the given neural network. That means that the network\n");
	fprintf(stderr, "                        has previously been trained... The format of the weights file is as follows:\n");
	fprintf(stderr, "                        'Threshold for Neuron' white space 'Weights connecting neuron and all neurons in previous layer'\n");
	fprintf(stderr, "  -arch A B ... Z       Define the architecture of the neural network as a sequence of N integers separated by\n");
	fprintf(stderr, "                        white space. N is the number of layers (including input and output layer) and\n");
	fprintf(stderr, "                        the i^th integer denotes the number of units in that layer. For example:\n");
	fprintf(stderr, "                        '-arch 5 10 23 2' creates a network of 5 inputs, 2 outputs and 2 hidden layers which\n");
	fprintf(stderr, "                        contain 10 and 23 units respectively.\n");
	fprintf(stderr, "  [-derivatives]        It will calculate the derivative values for each of the input vectors. By derivative\n");
	fprintf(stderr, "                        we mean the derivative of the function that the trained neural network represents.\n");
	fprintf(stderr, "  [-exemplars N]        The number of input exemplars, if known.\n");
	fprintf(stderr, "  [-start N]            Ignore all input vectors before the N^th (starting from 1). Default is 1.\n");
	fprintf(stderr, "  [-stop  N]            Ignore all input vectors after the N^th. Default is the last input vector.\n");
	fprintf(stderr, "  [-sigmoid]            Use this flag to request that the outputs of the network are passed through the same\n");
	fprintf(stderr, "                        non-linear (R->[0,1]) activation function used in the output of each hidden-layer\n");
	fprintf(stderr, "                        unit. If this flag is absent, the output of the neural network is a linear combination\n");
	fprintf(stderr, "                        of the signals to the last layer units -- e.g. outputs.\n");
	fprintf(stderr, "  [-discrete_output_map 'a b c ...']\n");
	fprintf(stderr, "                        Map each discrete output into this space separated list of strings (chars or numbers,\n                        whatever) e.g. first class goes to 'a', second to 'b' etc.\n                        The network type must be discrete for this to take effect, which means that first_class_at, last_class_at and class_separation were used during training,\n                        check the last-but-one line of the weights file.\n");
	fprintf(stderr, "  [-fast]               If you are sure that each input vector to the neural network\n                        is on a line on its own in the input file\n                        (i.e. each vector ends in a newline), then use this\n                        flag so as the program reads one such line in memory and\n                        then do the parsing of the numerical values. Basically the\n                        idea is that instead of doing one scanf for each vector\n                        value from file (slow) do a blog read of one line using\n                        fgets (theoretically faster) and then parse the numbers\n                        from memory. If you are not sure whether each vector is on\n                        a line on its own, then don't use this option.\n                        How much time saved: run twice the program with and\n                        without this flag on an input file which contains only\n                        a few lines of input and see how much you can save.\n                        In unix\n                            head -1000 input > cutdown\n                            date; run program with switch on 'cutdown' file; date\n                            date; run program without switch on 'cutdown' file; date\n                        to see whether your input file contains each vector on its own line, compare the\n                        two outputs above using unix's diff command.\n");
	fprintf(stderr, "  [-show]               Use this flag to request that the inputs to the neural network (e.g. those in the\n");
	fprintf(stderr, "                        input file) are printed at the output in addition to the outputs.\n");
	fprintf(stderr, "  [-line_numbers]       Precede each line of output with its line number. Useful for finding which is which after a sort\n");
	fprintf(stderr, "  [-memory]             Use this flag to find out how much memory has been (c)allocated by the process.\n");
	fprintf(stderr, "  [-usage|-help]        Print this informative piece of junk.\n");	
	fprintf(stderr, "ForwardPass v8.0, program by A.Hadjiprocopis, (C) Noodle Woman Software.\n");
	fprintf(stderr, "Bugs and suggestions to the author, andreashad2@gmail.com\n");
	fprintf(stderr, "Free to modify, plagiarise, delete and use this program for non-commercial\n");
	fprintf(stderr, "institutions and individuals.\n");

	return(FALSE);
}
