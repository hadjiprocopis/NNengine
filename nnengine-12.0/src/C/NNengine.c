/************************************************************************
 *									*
 *		ARTIFICIAL NEURAL NETWORKS SOFTWARE			*
 *									*
 *   An Error Back Propagation Neural Network Engine for Feed-Forward	*
 *			Multi-Layer Neural Networks			*
 *									*
 *			by Andreas Hadjiprocopis			*
 *			(andreashad2@gmail.com)			*
 *		   Copyright Andreas Hadjiprocopis, 1994,2007		*
 *									*
 ************************************************************************/

/* This file always starts training with random weights */

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#include "BPNetworkError.h"
#include "BPNetworkConstants.h"
#include "BPNetworkDefinitions.h"

int	IsEmptyString(char *str);
int	Usage(char *app_name, const char *extra_message);

#define	DEFAULT_PROGRESS_ITERATIONS	25

int	main(int argc, char **argv)
{
	FILE			*FileOfInputs, *pid_file, *error_surface_file, *ew_file, *progress_file;
	register int		i, j, k;
	int	x = 34;
	int			Iterations = 0, NumberOfLayers = 0, *Geometry = NULL,
				NumberOfExemplars = 0, ShakeCounter,
				progress_iterations = DEFAULT_PROGRESS_ITERATIONS,
				arch_start, my_pid, NumberOfNumbers = 0,
		                UpdateWeightsMethod = EXEMPLAR_WEIGHT_UPDATE_METHOD,
				FinalNumberOfIterations, DumpErrorSurfaceFlag = FALSE,
				NumberOfOutputClasses = 0,
				StartVector = 1, StopVector = -1,
				shouldSaveWeightsEvery = -1,
				shouldSaveWeightsEvery_UniqueFilename = FALSE,
				NumInputs, NumOutputs;
	char			buffer[MAX_CHARS_PER_LINE+2], pid_filename[256],
				InputFilename[256], WeightsFilename[256], ErrorSurfaceFilename[256],
				EWFilename[256], progressFilename[256], ResumeTrainingWeightsFilename[256],
				tmpWeightsFilename[256+30],
				trainingFlag, saveWeightsNowFlag = FALSE, ProduceHalfNetwork = FALSE,
				flag = TRUE, MonitorEWFlag = FALSE, SigmoidOutputFlag = FALSE,
				VerboseFlag = TRUE, LoadParametersFromWeightsFileFlag = TRUE,
				pid_basename[240];
	BPPrecisionType		SingleExemplarError = 0.0, AllExemplarsError = 0.0,
				**InputExemplars, **OutputExemplars, OldError, ErrorRate, ErrorRateRate,
				Beta = DEFAULT_BETA_VALUE, Lamda = DEFAULT_LAMDA_VALUE, OldRate,
				FirstClassAt = 0.0, LastClassAt = 0.0, new_beta, dummy,
				*pIndErr, **pInpExe, **pOutExe;
	BPNetwork		TheBPNetwork;
	BPNetworkEnumType	TrainingType = Continuous;
	long			CurrentSeed = time(0), myTime_num_seconds;
	char			myDateString[100]; /* man ctime specifies the length of this as 26 */
	time_t			myTime_struct; /* saving time now */

	TotalMemoryAllocated = 0;

	pid_filename[0]='\0'; strcpy(pid_basename, "NNengine."); /* this is the file containing the pid of this process, user can define the basename (default is NNengine) to make it NNengine.1234 and to hold a single number 1234 (which you can use to kill or communicate with this process) */

	/* This defines the signals to catch */
	my_pid = SetSignalHandler();
	/* This says that no signal is currently awaiting processing */
	ResetSignalHandler();

	InputFilename[0] = WeightsFilename[0] = ResumeTrainingWeightsFilename[0] = EWFilename[0] = progressFilename[0] = '\0';

	/* Get the commands from the command line */
	for(i=1;(i<argc)&&flag;i++){
		if( !strcmp(argv[i], "-input") ){
			if( ++i < argc ) strcpy(InputFilename, argv[i]);
			else flag = Usage(argv[0], "-input");
			continue;
		}
		if( !strcmp(argv[i], "-weights") ){
			if( ++i < argc ) strcpy(WeightsFilename, argv[i]);
			else flag = Usage(argv[0], "-weights");
			continue;
		}
		if( !strcmp(argv[i], "-pid_basename") ){
			if( ++i < argc ) strcpy(pid_basename, argv[i]);
			else flag = Usage(argv[0], "-pid_basename");
			continue;
		}
		if( !strcmp(argv[i], "-pid_filename") ){
			if( ++i < argc ) strcpy(pid_filename, argv[i]);
			else flag = Usage(argv[0], "-pid_filename");
			continue;
		}
		if( !strcmp(argv[i], "-resume_training") ){
			if( ++i < argc ) strcpy(ResumeTrainingWeightsFilename, argv[i]);
			else flag = Usage(argv[0], "-resume_training");
			continue;
		}
		if( !strcmp(argv[i], "-dont_load_parameters_from_weight_file") ){
			LoadParametersFromWeightsFileFlag = FALSE;
			continue;
		}
		if( !strcmp(argv[i], "-halfnetwork") ){
			ProduceHalfNetwork = TRUE;
			continue;
		}		
		if( !strcmp(argv[i], "-save") ){
			if( ++i < argc ) shouldSaveWeightsEvery = atoi(argv[i]);
			else flag = Usage(argv[0], "-save");
			continue;
		}
		if( !strcmp(argv[i], "-save_weights_every_file_unique") ){
			shouldSaveWeightsEvery_UniqueFilename = TRUE;
			continue;
		}
		if( !strcmp(argv[i], "-progress_filename") ){
			if( ++i < argc ) strcpy(progressFilename, argv[i]);
			else flag = Usage(argv[0], "progress_filename");
			continue;
		}
		if( !strcmp(argv[i], "-show_progress_iterations") ){
			if( ++i < argc ) progress_iterations = atoi(argv[i]);
			else flag = Usage(argv[0], "-show_progress_iterations");
			continue;
		}
		if( !strcmp(argv[i], "-sigmoid") ){
			SigmoidOutputFlag = TRUE;
			continue;
		}
		if( !strcmp(argv[i], "-error_surface") ){
			MonitorEWFlag = TRUE;
			if( ++i < argc ) strcpy(EWFilename, argv[i]);
			else flag = Usage(argv[0], "-error_surface");
			continue;
		}
		if( !strcmp(argv[i], "-start") ){
			if( ++i < argc ) StartVector = MAX(1, atoi(argv[i]));
			else flag = Usage(argv[0], "-start");
			continue;
		}
		if( !strcmp(argv[i], "-seed") ){
			if( ++i < argc ) CurrentSeed = MAX(1L, (long )atoi(argv[i]));
			else flag = Usage(argv[0], "-seed");
			continue;
		}
		if( !strcmp(argv[i], "-stop") ){   
			if( ++i < argc ) StopVector = atoi(argv[i]);
			else flag = Usage(argv[0], "-stop");
			continue;
		}
		if( !strcmp(argv[i], "-h") || !strcmp(argv[i], "-usage") || !strcmp(argv[i], "-help") ){
			flag = Usage(argv[0], NULL);
			continue;
		}
		if( !strcmp(argv[i], "-arch") ){
			arch_start = ++i;
			while( (i<argc) && (argv[i][0] != '-') ) i++;
			if( (NumberOfLayers=i-arch_start) < 3 ){
				fprintf(stderr, "%s: Error, the total number of layers must be greater than 2.\n", argv[0]);
				flag = Usage(argv[0], "-arch, number of layers must be >= 3");
				continue;
			}
			if( (Geometry=(int *)calloc(NumberOfLayers, sizeof(int))) == NULL ){
				fprintf(stderr, "%s: Error, could not allocate memory for %d int, Geometry.\n", argv[0], NumberOfLayers);
				flag = Usage(argv[0], "-arch, memory allocation problems...");
				continue;
			}
			TotalMemoryAllocated += NumberOfLayers * sizeof(int);
			for(j=arch_start;j<i;j++) Geometry[j-arch_start] = atoi(argv[j]);
			i--;
			continue;
		}
		if( !strcmp(argv[i], "-iters") ){   
			if( ++i < argc )  Iterations = atoi(argv[i]);
			else flag = Usage(argv[0], "-iters");
			if( Iterations <= 0 ){
				fprintf(stderr, "%s: Error, the number of iterations must be a positive integer.\n", argv[0]);
				flag = Usage(argv[0], "-iters, iterations must be positive integer");
			}
			continue;
		}
		if( !strcmp(argv[i], "-beta") ){   
			if( ++i < argc )  Beta = (BPPrecisionType )atof(argv[i]);
			else flag = Usage(argv[0], "-beta");
			if( Beta <= 0.0 ){
				fprintf(stderr, "%s: Error, the learning rate parameter (beta) must be a positive real number.\n", argv[0]);
				flag = Usage(argv[0], "-beta, must be positive real number");
			}
			continue;
		}
		if( !strcmp(argv[i], "-lamda") ){   
			if( ++i < argc )  Lamda = (BPPrecisionType )atof(argv[i]);
			else flag = Usage(argv[0], "-lamda");
			continue;
		}
		if( !strcmp(argv[i], "-epoch") ){
	                UpdateWeightsMethod = EPOCH_WEIGHT_UPDATE_METHOD;
			continue;
		}
		if( !strcmp(argv[i], "-num_output_classes") ){   
			if( ++i < argc )  NumberOfOutputClasses = atoi(argv[i]);
			else flag = Usage(argv[0], "-num_output_classes");
			if( NumberOfOutputClasses <= 1 ){
				fprintf(stderr, "%s: Error, the number of output classes must be greater than 1.\n", argv[0]);
				flag = Usage(argv[0], "-num_output_classes");
			}
			continue;
		}
		if( !strcmp(argv[i], "-first_class_at") ){
			if( ++i < argc )  FirstClassAt = (BPPrecisionType )atof(argv[i]);
			else flag = Usage(argv[0], "-first_class_at");
			continue;
		}
		if( !strcmp(argv[i], "-last_class_at") ){
			if( ++i < argc )  LastClassAt = (BPPrecisionType )atof(argv[i]);
			else flag = Usage(argv[0], "-last_class_at");
			continue;
		}
		if( !strcmp(argv[i], "-discrete_t") ){
			TrainingType = Discrete;
			continue;
		}
		if( !strcmp(argv[i], "-silent") ){
			VerboseFlag = FALSE;
			continue;
		}
	}
	VerboseFlag = FALSE;
	/* Check if necessary parameters are given */
	if( !InputFilename[0] || !WeightsFilename[0] || !Geometry || (Iterations==0) || !flag || (MonitorEWFlag&&(!EWFilename[0])) ){
		if( flag ) Usage(argv[0], "one of input filename, weights filename, architecture, iterations or error surface (if specified) filename was not given"); exit(1);
	}
	/* Seed the random number generator */
	Seed(CurrentSeed);

	/* Open the files for r/w */
	if( (FileOfInputs=fopen(InputFilename, "r")) == NULL ){
		fprintf(stderr, "%s: Error, could not open file '%s' for reading.\n", argv[0], InputFilename);
		exit(1);
	}
	if( MonitorEWFlag )
		if( (ew_file=fopen(EWFilename, "w")) == NULL ){
			fprintf(stderr, "%s: Error, could not open file '%s' for writing.\n", argv[0], EWFilename);
			exit(1);
		}
	if( progressFilename[0] != '\0' ){
		if( (progress_file=fopen(progressFilename, "w")) == NULL ){
			fprintf(stderr, "%s: Could not open file '%s' for writing progress.\n", argv[0], progressFilename);
			exit(1);
		}
		myTime_num_seconds = time(&myTime_struct);
		ctime_r(&myTime_struct, myDateString); myDateString[strlen(myDateString)-1] = '\0'; /* extra newline argghhhhhh */
		fprintf(progress_file, "# STARTED AT %s (%ld)\n", myDateString, myTime_num_seconds);
	}
			
	/* Find out our pid in case some one wants to kill us -- make it easy for them by creating a file - user can specify this file, or a basename to this file - the former is dangerous because it might be overwritten by multiple instances */
	if( pid_filename[0] == '\0' ){ sprintf(pid_filename, "%s.%d", pid_basename, my_pid); }
	if( (pid_file=fopen(pid_filename, "w")) == NULL ){
		fprintf(stderr, "%s: Could not open file '%s' for writing the pid.\n", argv[0], pid_filename);
		exit(1);
	}
	fprintf(pid_file, "%d\n", my_pid);
	fprintf(stderr, "%s: pid is %d\n", argv[0], my_pid);
	fclose(pid_file);

	/* Find out how many numbers (%f) in this file */
	while(1){
		fgets(buffer, MAX_CHARS_PER_LINE, FileOfInputs);
		if( feof(FileOfInputs) ) break;
		if( IsEmptyString(buffer) ) continue;
		NumberOfExemplars++;
	}
	rewind( FileOfInputs );
	/* Check if the start and stop vectors supplied (if any) are correct */
	if( StopVector < 0 ) StopVector = NumberOfExemplars;
	if( StopVector > NumberOfExemplars ) StopVector = NumberOfExemplars;
	if( StartVector > StopVector ){
		fclose(FileOfInputs);
		fprintf(stderr, "%s: Error, the starting vector was after the stop vector.\n", argv[0]);
		exit(1);
	}
	NumberOfExemplars = MAX(1, StopVector - StartVector+1);

	/* Allocate memory for the Input and Output Arrays to hold the Exemplars */
	if( (InputExemplars=(BPPrecisionType **)calloc(NumberOfExemplars, sizeof(BPPrecisionType *))) == NULL ){
		fprintf(stderr, "%s: Error, could not allocate memory for %d *PrecisionType, InputExemplars.\n", argv[0], NumberOfExemplars);
		free(Geometry);fclose(FileOfInputs); if( progressFilename[0] != '\0' ) fclose(progress_file);unlink(pid_filename);
		exit(1);
	}
	TotalMemoryAllocated += NumberOfExemplars * sizeof(BPPrecisionType *);

	NumInputs = Geometry[0];
	NumOutputs = Geometry[NumberOfLayers-1];
	for(i=0;i<NumberOfExemplars;i++)
		if( (InputExemplars[i]=(BPPrecisionType *)calloc(NumInputs, sizeof(BPPrecisionType))) == NULL ){
			fprintf(stderr, "%s: Error, could not allocate memory for %d PrecisionType, InputExemplars.\n", argv[0], NumInputs);
			for(j=0;j<i;j++) free(InputExemplars[j]);
			free(InputExemplars);free(Geometry);fclose(FileOfInputs); if( progressFilename[0] != '\0' ) fclose(progress_file);unlink(pid_filename);
			exit(1);
		}
	TotalMemoryAllocated += NumberOfExemplars * NumInputs * sizeof(BPPrecisionType);

	if( (OutputExemplars=(BPPrecisionType **)calloc(NumberOfExemplars, sizeof(BPPrecisionType *))) == NULL ){
		fprintf(stderr, "%s: Error, could not allocate memory for %d *PrecisionType, OutputExemplars.\n", argv[0], NumberOfExemplars);
		for(j=0;j<NumberOfExemplars;j++) free(InputExemplars[j]);
		free(InputExemplars);free(Geometry);fclose(FileOfInputs); if( progressFilename[0] != '\0' ) fclose(progress_file);unlink(pid_filename);
		exit(1);
	}
	TotalMemoryAllocated += NumberOfExemplars * sizeof(BPPrecisionType *);

	for(i=0;i<NumberOfExemplars;i++)
		if( (OutputExemplars[i]=(BPPrecisionType *)calloc(NumOutputs, sizeof(BPPrecisionType))) == NULL ){
			fprintf(stderr, "%s: Error, could not allocate memory for %d PrecisionType, OutputExemplars.\n", argv[0], NumOutputs);
			for(j=0;j<NumberOfExemplars;j++) free(InputExemplars[j]); free(InputExemplars);
			for(j=0;j<i;j++) free(OutputExemplars[j]); free(OutputExemplars);
			free(Geometry);fclose(FileOfInputs); if( progressFilename[0] != '\0' ) fclose(progress_file);unlink(pid_filename);
			exit(1);
		}
	TotalMemoryAllocated += NumberOfExemplars * NumOutputs * sizeof(BPPrecisionType);

	/* Malloc and Initialise the BP Network */
	/* MALLOC */
	if( (TheBPNetwork=(BPNetwork )calloc(1, sizeof(struct _BACKPROP_NET_STRUCT))) == NULL ){
		fprintf(stderr, "%s: Error, could not allocate memory for the BP Network.\n", argv[0]);
		free(Geometry);fclose(FileOfInputs); if( progressFilename[0] != '\0' ) fclose(progress_file);unlink(pid_filename);
		exit(1);
	}
	TotalMemoryAllocated +=  sizeof(struct _BACKPROP_NET_STRUCT);

	/* CREATE */
	TheBPNetwork->IndividualError = (BPPrecisionType *)NULL;
	TheBPNetwork->NetworkMap = (Neuron **)NULL;
	CreateBPNetwork( NumberOfLayers, Geometry, TheBPNetwork );
	TheBPNetwork->Beta	= Beta;
	TheBPNetwork->Lamda	= Lamda;
	if( SigmoidOutputFlag )
		TheBPNetwork->OutputType = Sigmoid;
	else
		TheBPNetwork->OutputType = Linear;
	/* WARNING : loading a weights file and setting the flag to false will ignore command line ie these options below. */
	if( NumberOfOutputClasses > 0 ){
		TheBPNetwork->NumberOfOutputClasses	= NumberOfOutputClasses;
		TheBPNetwork->ClassSeparation		= (LastClassAt - FirstClassAt) / (NumberOfOutputClasses-1);
		TheBPNetwork->FirstClassAt		= FirstClassAt;
		TheBPNetwork->LastClassAt		= LastClassAt;
		TheBPNetwork->NetworkType		= Discrete;
		TheBPNetwork->TrainingType		= TrainingType;
	} else {
		TheBPNetwork->NumberOfOutputClasses	= 0;
		TheBPNetwork->ClassSeparation		= 1.0;
		TheBPNetwork->FirstClassAt		= 0.0;
		TheBPNetwork->LastClassAt		= 0.0;
		TheBPNetwork->NetworkType		= Continuous;
		TheBPNetwork->TrainingType		= Continuous;
	}
	/* INITIALISE */
	InitialiseBPNetwork( TheBPNetwork );
	printf("TOTAL MEMORY ALLOCATED = %d bytes\n", TotalMemoryAllocated);

	/* Fill in the Input/Output Exemplar Arrays from file */
	for(i=1;i<StartVector;i++)
		for(j=0;j<(NumInputs+NumOutputs);j++) fscanf(FileOfInputs, "%lf", &dummy);
	for(i=StartVector;i<=StopVector;i++){
		for(j=0;j<NumInputs;j++)
			fscanf(FileOfInputs, "%lf", &(InputExemplars[i-StartVector][j]) );
		for(j=0;j<NumOutputs;j++)
			fscanf(FileOfInputs, "%lf", &(OutputExemplars[i-StartVector][j]) );
	}
	/* We do not need this anymore */
	fclose(FileOfInputs);

	if( (ResumeTrainingWeightsFilename[0] != '\0') && (access(ResumeTrainingWeightsFilename, R_OK) == 0) ){
		/* loading weights from existing weights file which has the same arch as this one! */
		FILE	*resume_training_weights_file;
		if( (resume_training_weights_file=fopen(ResumeTrainingWeightsFilename, "r")) == NULL ){
			fprintf(stderr, "%s : Could not open file '%s' for reading resume training weights.\n", argv[0], ResumeTrainingWeightsFilename);
			/* Free memory allocated */
			free(Geometry);
			for(i=0;i<NumberOfExemplars;i++){
				free(InputExemplars[i]);
				free(OutputExemplars[i]);
			}
			free(InputExemplars);
			free(OutputExemplars);
			free(TheBPNetwork);
			exit(1);
		}
		if( LoadWeightsFromFile(TheBPNetwork, (char *)NULL, resume_training_weights_file, LoadParametersFromWeightsFileFlag) != SUCCESS ){
			fprintf(stderr, "%s : call to LoadWeightsFromFile has failed for file '%s'.\n", argv[0], ResumeTrainingWeightsFilename);
			fclose(resume_training_weights_file); exit(1);
		}
		fclose(resume_training_weights_file);
		printf("LOADED WEIGHTS FROM '%s'\n", ResumeTrainingWeightsFilename);
	}

	if( MonitorEWFlag ){
		/* Put some information at the header of the error/weights dump file */
		/* Do the labelling first: W(A,B,C) means the weight connecting the B node of Layer A and the C node of Layer (A-1) */
		/*			 T(A,B) means the threshold of Node B in Layer A */
		/* Save Weights in the Weight File */
		/* Comment mark is `#' */
		fprintf(ew_file, "# W(A,B,C) means the weight connecting the B node of Layer A and the C node of Layer (A-1)\n");
		fprintf(ew_file, "# T(A,B) means the threshold of Node B in Layer A\n# Produced with the following command line\n# ");
		for(i=0;i<argc;i++) fprintf(ew_file, "%s ", argv[i]);
		fprintf(ew_file, "\n\n#iter ");
		for(i=1;i<TheBPNetwork->LayersNum;i++)
			for(j=0;j<TheBPNetwork->Layer[i];j++){
				fprintf(ew_file, "T(%d,%d) ", i, j);
				for(k=0;k<TheBPNetwork->NetworkMap[i][j]->Inpnum;k++)
					fprintf(ew_file, "W(%d,%d,%d) ", i, j, k);
			}
		fprintf(ew_file, "Error\n");
	}

	/* save weights before training - this is either a random weight initialisation matrix at iter 0 or the weights as read from an input file */
	if( shouldSaveWeightsEvery > 0 ){
		/* SaveWeightsInFile(TheBPNetwork, (char *)NULL, "w", stdout); */
		if( shouldSaveWeightsEvery_UniqueFilename == TRUE ) sprintf(tmpWeightsFilename, "%s_0", WeightsFilename); else strcpy(tmpWeightsFilename, WeightsFilename);
		if( VerboseFlag ) fprintf(stderr, "SAVING WEIGHTS BEFORE TRAINING, weights file=%s\n", tmpWeightsFilename);
		if( ProduceHalfNetwork == TRUE ) SaveHalfWeightsInFile(TheBPNetwork, tmpWeightsFilename, "w", (FILE *)NULL, (FILE *)NULL);
		SaveWeightsInFile(TheBPNetwork, tmpWeightsFilename, "w", (FILE *)NULL);
	}

	ShakeCounter = 0;
	OldError = 1000000.0;
	OldRate  = 1000000.0;

	/* Seed the random number generator again so as to start fresh in case some random numbers wasted in preparing the network */
	Seed(CurrentSeed);

	/* Training Stage */
	if( VerboseFlag ) fprintf(stderr, "TRAINING STAGE\n");
	i = 0;
	trainingFlag = TRUE; /* keep training */
	saveWeightsNowFlag = FALSE; /* save the weights now and cont training (when a signal is caught) */
	while( trainingFlag ){
		AllExemplarsError = 0.0;
		for(j=StartVector,pInpExe=&(InputExemplars[j-StartVector]),pOutExe=&(OutputExemplars[j-StartVector]);j<=StopVector;j++,pInpExe++,pOutExe++){
			//BPNetworkFeedInputs(NumInputs, InputExemplars[j-StartVector], TheBPNetwork);
			BPNetworkFeedInputs(NumInputs, *pInpExe, TheBPNetwork);
			BPNetworkForwardPropagate(TheBPNetwork);
			//BPNetworkBackPropagate(NumOutputs, OutputExemplars[j-StartVector], TheBPNetwork);
			BPNetworkBackPropagate(NumOutputs, *pOutExe, TheBPNetwork);
			if( UpdateWeightsMethod == EXEMPLAR_WEIGHT_UPDATE_METHOD ){
				BPNetworkAdjustWeights(TheBPNetwork, 1);
				/*SaveWeightsInFile(TheBPNetwork, (char *)NULL, "w", stdout);*/
			}
			SingleExemplarError = 0.0;
			for(k=0,pIndErr=&(TheBPNetwork->IndividualError[k]);k<NumOutputs;k++,pIndErr++){
				//SingleExemplarError += ABS(TheBPNetwork->IndividualError[k]);
				SingleExemplarError += ABS(*pIndErr);
			}
			AllExemplarsError += (SingleExemplarError/NumOutputs);
		}
		if( MonitorEWFlag ) MonitorEW(i, TheBPNetwork, ew_file);

		AllExemplarsError /= NumberOfExemplars; 
		TheBPNetwork->TotalError = AllExemplarsError;
		ErrorRate = ABS(OldError-AllExemplarsError);
		ErrorRateRate = ABS(OldRate - ErrorRate);

		if( !((i+1) % progress_iterations) ){
			fprintf(stderr, "ERROR(%d) = %.7f, dE = %.7f, ddE = %.7f\n", i+1, AllExemplarsError, ErrorRate, ErrorRateRate);
			if( progressFilename[0] != '\0' ){ fprintf(progress_file, "ERROR(%d) = %.7f, dE = %.7f, ddE = %.7f\n", i+1, AllExemplarsError, ErrorRate, ErrorRateRate); fflush(progress_file); }
		}

		/* save weights every so many iterations if that was specified */
		if( saveWeightsNowFlag || ((shouldSaveWeightsEvery>0) && !((i+1) % shouldSaveWeightsEvery)) ){
			/* SaveWeightsInFile(TheBPNetwork, (char *)NULL, "w", stdout); */
			if( shouldSaveWeightsEvery_UniqueFilename == TRUE ) sprintf(tmpWeightsFilename, "%s_%d", WeightsFilename, (i+1)); else strcpy(tmpWeightsFilename, WeightsFilename);
			if( VerboseFlag ) fprintf(stderr, "SAVING WEIGHTS DURING TRAINING at iter=%d, weights file=%s\n", i+1, tmpWeightsFilename);
			if( ProduceHalfNetwork == TRUE ) SaveHalfWeightsInFile(TheBPNetwork, tmpWeightsFilename, "w", (FILE *)NULL, (FILE *)NULL);
			SaveWeightsInFile(TheBPNetwork, tmpWeightsFilename, "w", (FILE *)NULL);
			saveWeightsNowFlag = FALSE;
			if( ((i+1) % progress_iterations) ){
				/* write the progress as well - if not already written */
				fprintf(stderr, "ERROR(%d) = %.7f, dE = %.7f, ddE = %.7f\n", i+1, AllExemplarsError, ErrorRate, ErrorRateRate);
				if( progressFilename[0] != '\0' ){ fprintf(progress_file, "ERROR(%d) = %.7f, dE = %.7f, ddE = %.7f\n", i+1, AllExemplarsError, ErrorRate, ErrorRateRate); fflush(progress_file); }
			}
		}

		if( UpdateWeightsMethod == EPOCH_WEIGHT_UPDATE_METHOD ){
			BPNetworkAdjustWeights(TheBPNetwork, NumberOfExemplars);
			/*SaveWeightsInFile(TheBPNetwork, (char *)NULL, "w", stdout);*/
		}

		if( i >= Iterations ) break;
		i++;

		OldError = AllExemplarsError;
		OldRate = ErrorRate;

		if( SignalCaught != SIGNAL_QUEUE_IS_EMPTY ){
			switch(SignalCaught){
				/* long live unix : these are user defined signals, if you want to add more actions, then use SIGRTMIN+1 etc up to SIGRTMAX */
				case SIGUSR1:
					new_beta = TheBPNetwork->Beta * (1.0-(BPPrecisionType )(CHANGE_BETA_STEP));
					fprintf(stderr, "%s: Caught signal %d. Decreasing Beta: %f -> %f\n", argv[0], SignalCaught, TheBPNetwork->Beta, new_beta);
					psignal(SignalCaught, "in this OS, the signal is described as");
					TheBPNetwork->Beta = new_beta;
					CatchSignal(SIGUSR1);
					break;
				case SIGUSR2:
					new_beta = TheBPNetwork->Beta * (1.0+(BPPrecisionType )(CHANGE_BETA_STEP));
					fprintf(stderr, "%s: Caught signal %d. Increasing Beta: %f -> %f\n", argv[0], SignalCaught, TheBPNetwork->Beta, new_beta);
					psignal(SignalCaught, "in this OS, the signal is described as");
					TheBPNetwork->Beta = new_beta;
					CatchSignal(SIGUSR2);
					break;
/*	can't do that because SIGRTMIN is not a constant but a function ...*/
/*				case SIGRTMIN:
					// save weights but don't exit - continue
					RemoveSignalHandler();
					saveWeightsNowFlag = TRUE;
					fprintf(stderr, "%s: Caught signal %d. Saving weights (it may take a while if this is a heavy process, be patient)\n", argv[0], SignalCaught);
					psignal(SignalCaught, "in this OS, the signal is described as");
					break;
*/
				/* someone is telling us to quit, so save weights and exit ... */
				case SIGINT:
				case SIGTERM:
				case SIGQUIT:
					/* stop training, save weights and exit as if numIters was less, no harm done */
					RemoveSignalHandler();
					trainingFlag = FALSE;
					fprintf(stderr, "%s: Caught signal %d. Saving weights and exit (it may take a while if this is a heavy process, be patient)...\n", argv[0], SignalCaught);
					psignal(SignalCaught, "in this OS, the signal is described as");
					break;
				default:
#ifdef SIGRTMIN
					if( SignalCaught == SIGRTMIN ){
						/* this should have been in a separate case above but SIGRTMIN is not a constant but a function (__libc_current_sigrtmin) in linux */
						/* save weights but don't exit - continue */
						RemoveSignalHandler();
						saveWeightsNowFlag = TRUE;
						fprintf(stderr, "%s: Caught signal %d. Saving weights (it may take a while if this is a heavy process, be patient)\n", argv[0], SignalCaught);
						psignal(SignalCaught, "in this OS, the signal is described as");
					} else {
#endif
						fprintf(stderr, "%s: Caught signal %d. Don't know what to do with it, ignoring it...\n", argv[0], SignalCaught);
						psignal(SignalCaught, "in this OS, the signal is described as");
#ifdef SIGRTMIN
					}
#endif
					break;
			} /* switch */
			ResetSignalHandler();
		}
	} /* for(i , the iterations */
	fprintf(stderr, "ERROR(%d) = %lf, dE = %.7f, ddE = %.7f\n", i, AllExemplarsError, ErrorRate, ErrorRateRate);
	FinalNumberOfIterations = i;

	/* Save Weights in the Weight File */
	fprintf(stderr, "SAVING WEIGHTS in '%s'\n", WeightsFilename);
	if( ProduceHalfNetwork == TRUE ) SaveHalfWeightsInFile(TheBPNetwork, WeightsFilename, "w", (FILE *)NULL, (FILE *)NULL);
	SaveWeightsInFile(TheBPNetwork, WeightsFilename, "w", (FILE *)NULL);

	/* Test all the Inputs through a forward pass */
	if( VerboseFlag ){
		fprintf(stderr, "TESTING STAGE\n");
		/* print a header line saying inputs/outputs */
		for(i=0;i<NumInputs*3-3;i++) fprintf(stderr, " ");
		fprintf(stderr, "INPUTS"); /* 6 characters length */
		for(i=0;i<NumInputs*3-3;i++) fprintf(stderr, " ");
		fprintf(stderr, "\t");
		for(i=0;i<NumOutputs*3-3;i++) fprintf(stderr, " ");
		fprintf(stderr, "EXPECT"); /* 6 characters length */
		for(i=0;i<NumOutputs*3-3;i++) fprintf(stderr, " ");
		fprintf(stderr, "\t");
		if( TheBPNetwork->NetworkType == Discrete ){
			for(i=0;i<NumOutputs*3-7;i++) fprintf(stderr, " ");
			fprintf(stderr, "ACTUAL(CONT.)"); /* 14 characters length */
		} else {
			for(i=0;i<NumOutputs*3-3;i++) fprintf(stderr, " ");
			fprintf(stderr, "ACTUAL");
		}
		fprintf(stderr, "\n");
	}
	for(i=StartVector;i<=StopVector;i++){
		BPNetworkFeedInputs(NumInputs, InputExemplars[i-StartVector], TheBPNetwork);
		BPNetworkForwardPropagate(TheBPNetwork);
		if( VerboseFlag ){
			for(j=0;j<NumInputs;j++){
				fprintf(stderr, "%4.3lf ", InputExemplars[i-StartVector][j]);
			}
			fprintf(stderr, "\t");
		}
		SingleExemplarError = 0.0;
		if( VerboseFlag ){
			for(j=0;j<NumOutputs;j++){
				fprintf(stderr, "%4.3lf ", OutputExemplars[i-StartVector][j]);
				SingleExemplarError += ABS(TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output - OutputExemplars[i-StartVector][j]);
			}
			fprintf(stderr, "\t");
			if( TheBPNetwork->NetworkType == Continuous )
				for(j=0;j<NumOutputs;j++)
					fprintf(stderr, "%4.3lf ", TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output);
			else for(j=0;j<NumOutputs;j++)
				fprintf(stderr, "%4.3lf (%4.3lf) ", TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->DiscreteOutput, TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output);
			fprintf(stderr, "\n");
		} else {
			for(j=0;j<NumOutputs;j++){
				SingleExemplarError += ABS(TheBPNetwork->NetworkMap[NumberOfLayers-1][j]->Output - OutputExemplars[i-StartVector][j]);
			}
		}
	} /* forloop */

	printf("\nITERATIONS = (%d), %d, EXEMPLARS = %d\nBETA = %lf, LAMDA = %lf\nFINAL ERROR = %lf\nFFNN ARCHITECTURE =", Iterations, FinalNumberOfIterations, NumberOfExemplars, TheBPNetwork->Beta, TheBPNetwork->Lamda, AllExemplarsError);
	for(i=0;i<NumberOfLayers;i++)
		printf(" %d", Geometry[i]);
	printf("\nTOTAL MEMORY ALLOCATED = %d bytes", TotalMemoryAllocated);

	if( MonitorEWFlag ){
		MonitorEW(i, TheBPNetwork, ew_file); /* for one last time */
		fclose(ew_file);
	}

	if( TheBPNetwork->NetworkType == Continuous )
		printf("\nOutput (Network) Type is Continuous, Training was Continuous\n");
	else{
		if( TheBPNetwork->TrainingType == Continuous )
			printf("\nOutput (Network) Type is Discrete, Training was Continuous\n");
		else
			printf("\nOutput (Network) Type is Discrete, Training was Discrete\n");
	}

	DestroyBPNetwork(TheBPNetwork);
	unlink(pid_filename);
	if( progressFilename[0] != '\0' ){
		myTime_num_seconds = time(&myTime_struct);
		ctime_r(&myTime_struct, myDateString); myDateString[strlen(myDateString)-1] = '\0'; /* extra newline argghhhhhh */
		fprintf(progress_file, "# FINISHED AT %s (%ld)\n", myDateString, myTime_num_seconds);
		fclose(progress_file);
	}

	/* Free memory allocated */
	free(Geometry);
	for(i=0;i<NumberOfExemplars;i++){
		free(InputExemplars[i]);
		free(OutputExemplars[i]);
	}
	free(InputExemplars);
	free(OutputExemplars);
	free(TheBPNetwork);

	exit(0);
}

int	IsEmptyString(char *str)
{
	int	i = 0;

	while( (str[i]!='\n') && (str[i]!='\0') )
		if( str[i++] != ' ' ) return(FALSE);

	return(TRUE);
}

int	Usage(char *app_name, const char *extra_message)
{
	fprintf(stderr, "Usage: %s options...\n", app_name);
	fprintf(stderr, "Options are (squarish brackets, [], denote optional parameter):\n");
	fprintf(stderr, "  -input filename       The name of the file holding the set of inputs to\n");
	fprintf(stderr, "                        be fed to the neural network. The format of this\n");
	fprintf(stderr, "                        file is a sequence of vectors. Each vector consists\n");
	fprintf(stderr, "                        of two sub-vectors: the input and the output vector.\n");
	fprintf(stderr, "                        The input vector is fed to the network while the\n");
	fprintf(stderr, "                        output vector is used in guiding the learning process.\n");
	fprintf(stderr, "                        The number of elements in each of the input sub-vectors\n");
	fprintf(stderr, "                        is equal to the number of first layer units\n");
	fprintf(stderr, "                        -- e.g. inputs --of the neural network. The number\n");
	fprintf(stderr, "                        of elements of the output sub-vectors is equal to\n");
	fprintf(stderr, "                        the number of last layer units -- e.g. outputs.\n");

	fprintf(stderr, "  -weights filename     The name of the file to save the set of weights\n");
	fprintf(stderr, "                        corresponding to the final trained state of the\n");
	fprintf(stderr, "                        neural network. The format of the weights file\n");
	fprintf(stderr, "                        is as follows: 'Threshold for Neuron' white\n");
	fprintf(stderr, "                        space 'Weights connecting neuron and all neurons\n");
	fprintf(stderr, "                        in previous layer'.\n");
	
	fprintf(stderr, "  -arch A B ... Z       Define the architecture of the neural network\n");
	fprintf(stderr, "                        as a sequence N integers separated by white\n");
	fprintf(stderr, "                        space. N is the number of layers (including input\n");
	fprintf(stderr, "                        and output layer) and the i^th integer denotes the\n");
	fprintf(stderr, "                        number of units in that layer. For example:\n");
	fprintf(stderr, "                        example: '-arch 5 10 23 2' creates a nework of 5 inputs\n");
	fprintf(stderr, "                        2 outputs and 2 hidden layers which contain\n");
	fprintf(stderr, "                        10 and 23 units respectively.\n");

	fprintf(stderr, "  -iters N              The number of epochs (iterations) the basic training process\n");
	fprintf(stderr, "                        must be repeated.\n");

	fprintf(stderr, "  [-start N]            Ignore all input vectors before the N^th (starting from 1).\n");
	fprintf(stderr, "                        Default is 1.\n");

	fprintf(stderr, "  [-stop  N]            Ignore all input vectors after the N^th. Default is the last\n");
	fprintf(stderr, "                        input vector.\n");

	fprintf(stderr, "  [-halfnetwork]        Use this flag to request that the weights be saved in three different\n");
	fprintf(stderr, "                        files as follows: the first file has the specified weights file name\n");
	fprintf(stderr, "                        and contains all the weights of the network - nothing strange with this.\n");
	fprintf(stderr, "                        the second file has the postfix '_first_half' and contains the weights\n");
	fprintf(stderr, "                        for the first half of the neural network only. This is useful when you\n");
	fprintf(stderr, "                        want to operate the network as a compressor i.e. train a network as: 200 10 2 2 10 200\n");
	fprintf(stderr, "                        use the inputs as outputs and then split the network in half. You will\n");
	fprintf(stderr, "                        have a mapping from 200 to 2 dimensions, something like PCA.\n");
	fprintf(stderr, "                        Finally, the third file will contain the other half of the network\n");
	fprintf(stderr, "                        and it will be postfixed by '_second_half'. If you want to decompress your\n");
	fprintf(stderr, "                        compressed data use this network.\n");

	fprintf(stderr, "  [-sigmoid]            Use this flag to request that the outputs of the network are\n");
	fprintf(stderr, "                        passed through the same non-linear (R->[0,1]) activation function\n");
	fprintf(stderr, "                        used in the output of each hidden-layer unit. If this flag is\n");
	fprintf(stderr, "                        unit. If this flag is absent, the output of the neural network is\n");
	fprintf(stderr, "                        bsent, the output of the neural network is a linear combination\n");
	fprintf(stderr, "                        of the signals to the last layer units -- e.g. outputs.\n");

	fprintf(stderr, "  [-beta B]             The learning rate parameter. How much should the weights change\n");
	fprintf(stderr, "                        from cycle to cycle? Use a number between 0.005 to 1.5 (rule of thumb).\n");
	fprintf(stderr, "                        The smaller this parameter is the more stable the gradient descent is\n");
	fprintf(stderr, "                        but much slower too. A high learning rate might lead to a minimum much,\n");
	fprintf(stderr, "                        faster but as the minimum point is reached, the network becomes unstable.\n");

	fprintf(stderr, "  [-lamda L]            Lamda (momentum) determines how much the previous weight state affects\n");
	fprintf(stderr, "                        the current weight change. Use a very small value (0.05 and less) or\n");
	fprintf(stderr, "                        do not use it at all -- e.g. default is zero. The training might\n");
	fprintf(stderr, "                        become unstable when using this parameter.\n");

	fprintf(stderr, "  [-epoch]              Weight updates may take place either every time a new input vector\n");
	fprintf(stderr, "                        (exemplar) is fed to the network or when all the input vectors are\n");
	fprintf(stderr, "                        fed -- e.g. an epoch.\n");

	fprintf(stderr, "  [-show_progress_iterations number]\n");
	fprintf(stderr, "                        Show progress every 'number' iterations.\n");

	fprintf(stderr, "  [-progress_filename name]\n");
	fprintf(stderr, "                        Progress is shown in stderr every %d iterations, by default or whatever\n", DEFAULT_PROGRESS_ITERATIONS);
	fprintf(stderr, "                        number you specify by the '-show_progress_iterations number' option.\n");
	fprintf(stderr, "                        Optionally, progress can also be saved in a file specified here every\n");
	fprintf(stderr, "                        %d iterations, by default or whatever you specify.\n", DEFAULT_PROGRESS_ITERATIONS);
	fprintf(stderr, "                        The format of this file is :\n");
	fprintf(stderr, "                        ERROR(25) = 0.3646753, dE = 0.0001904, ddE = 0.0000289 (time)\n");
	
	fprintf(stderr, "  [-pid_basename string]\n");
	fprintf(stderr, "                        this is the file containing the pid of this process, user can define the basename (default is NNengine)\n");
	fprintf(stderr, "                        to make it NNengine.1234 and to hold a single number 1234 (which you can use to kill or communicate\n");
	fprintf(stderr, "                        with this process).\n");

	fprintf(stderr, "  [-pid_filename filename]\n");
	fprintf(stderr, "                        this is the file containing the pid of this process EXACTLY, not a basename\n");
	fprintf(stderr, "                        the pid of this process will be written into this file as a single integer with a newline\n");
	fprintf(stderr, "                        at the end. The benefit of using a basename is that each instance of this NNengine\n");
	fprintf(stderr, "                        will create a unique PID file automatically, whereas using a filename, two NNengine processes\n");
	fprintf(stderr, "                        can overwrite each other's pid files. The advantage is that you can open it to read it because\n");
	fprintf(stderr, "                        you know the filename exactly. NOTE that when a NNengine finishes this file is deleted\n");
	fprintf(stderr, "                        HOWEVER, if the process crashes, this file stays in the local directory, so you may get confused\n");
	
	fprintf(stderr, "  [-resume_training a_weights_file]\n");
	fprintf(stderr, "                        The starting state of the network (weights) will not be set randomly but\n");
	fprintf(stderr, "                        will be loaded from the weights of the specified file, therefore being\n");
	fprintf(stderr, "                        able to resume training done at an earlier time from the point in space\n");
	fprintf(stderr, "                        it was left - WARNING, there might be some discrepancies because of accuracy\n");
	fprintf(stderr, "                        lost in files.\n");
	fprintf(stderr, "                        If the specified file does not exist, the program will just ignore this option\n");
	fprintf(stderr, "                        and continue as normal with training starting with random weights.\n");
	fprintf(stderr, "                        There is no problem if the specified file is the same as the weights output (-weights)\n");
	fprintf(stderr, "                        filename. In this case, the new weights will overwrite the old weights after they\n");
	fprintf(stderr, "                        have been loaded to the network - so resume training will take effect but initial\n");
	fprintf(stderr, "                        weights file will be overwritten with the final weights of the training.\n");

	fprintf(stderr, "  -dont_load_parameters_from_weight_file\n");
	fprintf(stderr, "                        by default, if loading weights from file (-resume_training)\n");
	fprintf(stderr, "                        training type, network type, class separation, first class at, etc.\n");
	fprintf(stderr, "                        will all be loaded from the weights file, unless you specify this flag\n");
	fprintf(stderr, "                        then it expects these parameters to be provided at the command line.\n");

	fprintf(stderr, "  [-num_output_classes N] N specifies the number of output classes in the training data set,\n");
	fprintf(stderr, "                        in the case when discrete data is used. For example for classification\n");
	fprintf(stderr, "                        of a set of input patterns into 5 categories use N = 5 (0.0, 0.25, 0.5, 0.75, 1.0). Note that this\n");
	fprintf(stderr, "                        switch is used in conjuction with `-first_class_at' and `-last_class_at' -- see below.\n");
	fprintf(stderr, "                        When this parameter is > 1, then the output is considered to be discrete and it will be\n");
	fprintf(stderr, "                        quantised to fit these output classes, e.g. an output of 0.23 will become 0.25 at the output\n");
	fprintf(stderr, "                        This quantisation will apply only to when the outputs are printed last, during training\n");
	fprintf(stderr, "                        this has no effect - the error is calculated on the 0.23 value and not on the 0.25\n");
	fprintf(stderr, "                        (see -discrete_t which does that).\n");

	fprintf(stderr, "  [-first_class_at F]   F (a float) specifies the numerical value of the first class in the\n");
	fprintf(stderr, "                        discrete data set.\n");

	fprintf(stderr, "  [-last_class_at L]    L (a float) specifies the numerical value of the last class in the\n");
	fprintf(stderr, "                        discrete data set. If the number of classes, N, is 4 and F = 0.0\n");
	fprintf(stderr, "                        while L = 1.0 that means the classes are at 0.0, 0.33, 0.66 and 1.0.\n");

	fprintf(stderr, "  [-discrete_t]         Discrete training means that when the output of the\n");
	fprintf(stderr, "                        neural network is quantised (e.g. the switches\n");
	fprintf(stderr, "                        `-classes', `-firstclass' and `-lastclass' were used)\n");
	fprintf(stderr, "                        the error propagated back to the various neurons will\n");
	fprintf(stderr, "                        be the discrepancy between the discretised output and\n");
	fprintf(stderr, "                        expected output. For example if there were 4 output\n");
	fprintf(stderr, "                        classes at 0.0, 0.33, 0.66 and 1.0, the expected\n");
	fprintf(stderr, "                        output of the node was 0.33 and the actual output of\n");
	fprintf(stderr, "                        the node was 0.25 (which after quantisation becomes\n");
	fprintf(stderr, "                        0.33). If the `-discrete_t' switch was specified then\n");
	fprintf(stderr, "                        the error is 0.33 (actual,discrete) - 0.33 (expected,\n");
	fprintf(stderr, "                        discrete) = 0.0. On the absense of the `-discrete_t'\n");
	fprintf(stderr, "                        switch the error is 0.25 (actual,continuous) - 0.33\n");
	fprintf(stderr, "                        (expected,discrete) -- which makes a lot of a difference.\n");

	fprintf(stderr, "  [-seed S]             Supply a seed to the random number generator so that\n");
	fprintf(stderr, "                        results can be reproduced (same seed means same weights).\n");
	fprintf(stderr, "                        The absence of this option indicates that a random seed\n");
	fprintf(stderr, "                        (using C function time(0)) will be selected.\n");

	fprintf(stderr, "  [-error_surface File]    Supply a filename if you want to have all the weights and the\n");
	fprintf(stderr, "                        total error monitored every time there is a weight update.\n");
	fprintf(stderr, "                        The output file will contain columns for each free parameter of the\n");
	fprintf(stderr, "                        system (weights and thresholds) as well as the error. Each row in the\n");
	fprintf(stderr, "                        file will represent a snapshot of free parameter state with the\n");
	fprintf(stderr, "                        respective error at that point in training. The first column contains\n");
	fprintf(stderr, "                        the current iteration and the last column contains the error at that\n");
	fprintf(stderr, "                        iteration. The header contains more explanation as to which column\n");
	fprintf(stderr, "                        corresponds to which free parameter.\n");
	fprintf(stderr, "                        Once you have this file, it is interesting to make some graphs in order\n");
	fprintf(stderr, "                        to see how the free parameters vary with the error and iterations\n");
	fprintf(stderr, "                        If you are using gnuplot, then you can do:\n");
	fprintf(stderr, "                                 plot 'err_surf' us 1:2 with lines\n");
	fprintf(stderr, "                        this will plot the variation of the weight corresponding to column #2\n");
	fprintf(stderr, "                        as the number of iterations increases.\n");
	fprintf(stderr, "                                plot 'err_surf' us 2:100 with lines\n");
	fprintf(stderr, "                        will plot the variation of the said weight with the error (assuming\n");
	fprintf(stderr, "                        error is in column 100\n");
	fprintf(stderr, "                        set contour; set surface; set dgrid3d 50; splot 'err_surf' us 1:2:3 with lines\n");
	fprintf(stderr, "                        will plot (yes use splot) a 3D surface of these columns, etc.\n");

	fprintf(stderr, "  [-save S]             Save weights into the file specified with the '-weights' parameter\n");
	fprintf(stderr, "                        every 'S' iterations. The previous weights into this file will\n");
	fprintf(stderr, "                        be overwritten. This parameter is useful if training takes a\n");
	fprintf(stderr, "                        very long time and you want to see whether is what you want.\n");
	fprintf(stderr, "                        In conjuction with '-save_weights_every_file_unique' option, a new weights\n");
	fprintf(stderr, "                        file can be created each time (e.g weights_1, weights_2).\n");

	fprintf(stderr, "  [-save_weights_every_file_unique]\n");
	fprintf(stderr, "                        In saving weights file every so many iterations (see the -save option)\n");
	fprintf(stderr, "                         by default, the old weights file will be overwritten by the new one.\n");
	fprintf(stderr, "                         If you want to keep all the weights file to monitor the progress of training\n");
	fprintf(stderr, "                         especially in conjuction with testing in order to find the optimum number\n");
	fprintf(stderr, "                         of training iterations then specify this flag which will create each time\n");
	fprintf(stderr, "                         the weights are saved a unique weights file by postfixing the current iteration\n");
	fprintf(stderr, "                         number to the specified weights file name (with -weights option).\n");
	fprintf(stderr, "                         Care should be taken with disk space because weight files can be really huge\n");
	fprintf(stderr, "                         depending on network architecture.\n");

	fprintf(stderr, "  [-silent]             Supply less information at the standard output, nameley the output\n");
	fprintf(stderr, "                        during the last forward pass (which could be kilometric).\n");
	fprintf(stderr, "                        total error monitored every time there is a weight update.\n");
	fprintf(stderr, "  [-usage|-help]        Print this informative piece of junk.\n\n");

	fprintf(stderr, "Interaction at runtime, signals accepted:\n");
	fprintf(stderr, "                        A user can interact at runtime with the program by sending a signal\n");
	fprintf(stderr, "                        to it using the 'kill -NAME PID' command where NAME is the signal name\n");
	fprintf(stderr, "                        (see 'man kill' and 'kill -l' for a list of signals in unix) and\n");
	fprintf(stderr, "                        PID is the process id of our neural network program. The pid of our program\n");
	fprintf(stderr, "                        can be determined via a number of ways: 1) When you run the program the PID is\n");
	fprintf(stderr, "                        displayed on the screen - the first thing that is printed in fact, 2) a file in\n");
	fprintf(stderr, "                        the current directory is created with the name NNengine.XXXX where XXXX is the PID\n");
	fprintf(stderr, "                        of our program, this file also contains the PID, 3) finally, ps -axu | grep NNengine\n");
	fprintf(stderr, "                        will list all the processes named NNengine with their respective PID, if you have more\n");
	fprintf(stderr, "                        than one running, then you can not be sure which one you need. Use methods 1) or 2).\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "*** Signals: Important1 ***\n");
	fprintf(stderr, "  The program may take some time to respond to\n");
	fprintf(stderr, "  a signal so be patient. DO NOT SEND A SIGNAL TWICE unless\n");
	fprintf(stderr, "  you need/want to (this is NOT Windows)\n");
	fprintf(stderr, "  because there is a signal queue and it will be stored\n");
	fprintf(stderr, "  there. If there is a large number of weights or there is\n");
	fprintf(stderr, "  a large number of training examples, then signals may take\n");
	fprintf(stderr, "  some time to be dealt with. Be patient!\n");
	fprintf(stderr, "  When a signal is dealt, a message will appear on screen.\n");
	fprintf(stderr, "*** Signals: Important2 ***\n");
	fprintf(stderr, "  when you want to send a signal use kill -INT instead of kill -2\n");
	fprintf(stderr, "  because in different OS, the numbers might change.\n");
	fprintf(stderr, "  Whatever you do kill -INT is not the same as kill -KILL which is\n");
	fprintf(stderr, "  SIGKILL (the famous number 9) will kill the application instantly\n");
	fprintf(stderr, "  with no weights saved!!!\n");
	fprintf(stderr, "\n");

	fprintf(stderr, "USR1                    This signal can be send by executing 'kill -USR1 XXX' (where XXX is the PID of our program)\n");
	fprintf(stderr, "                        Upon receipt of the signal, the program will DECREASE BETA (learning rate) by a %.0f%%\n", CHANGE_BETA_STEP*100.0);
	fprintf(stderr, "                        (this percentage is set at compile time in BPNetworkConstants.h (look for CHANGE_BETA_STEP)\n");
	fprintf(stderr, "USR2                    This signal can be send by executing 'kill -USR2 XXX' (where XXX is the PID of our program)\n");
	fprintf(stderr, "                        Upon receipt of the signal, the program will INCREASE BETA (learning rate) by a %.0f%%\n", CHANGE_BETA_STEP*100.0);
	fprintf(stderr, "                        (this percentage is set at compile time in BPNetworkConstants.h (look for CHANGE_BETA_STEP)\n");
	fprintf(stderr, "RTMIN                   Upon receipt of this signal, the program will save the weights in the file specified\n");
	fprintf(stderr, "                        during execution (with -weights ...)\n");
	fprintf(stderr, "                        If the program takes long and you need to see the weights, send this signal.\n");
	fprintf(stderr, "                        you will get a perfectly usable weights file which can be passed to the ForwardPass\n");
	fprintf(stderr, "                        command.\n");
	fprintf(stderr, "INT                     This signal will cause our program to save the weights and exit as if the number\n");
	fprintf(stderr, "                        of iterations reached the total number of iterations. There will be no effect to the\n");
	fprintf(stderr, "                        output of the program except\n");
	fprintf(stderr, "                        from the fact that the total number of iterations was not reached.\n");
	fprintf(stderr, "                        This signal can also be sent using a Control-C from the prompt.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "NNengine v8.0, program by A.Hadjiprocopis, (C) Noodle Woman Software.\n");
	fprintf(stderr, "Bugs and suggestions to the author, andreashad2@gmail.com\n");
	fprintf(stderr, "Free to modify, plagiarise, delete and use this program for non-commercial\n");
	fprintf(stderr, "institutions and individuals.\n");

	if( extra_message != NULL ) fprintf(stderr, "\n%s\n", extra_message);
	return(FALSE);
}

