/************************************************************************
 *									*
 *		ARTIFICIAL NEURAL NETWORKS SOFTWARE			*
 *									*
 *   An Error Back Propagation Neural Network Engine for Feed-Forward	*
 *			Multi-Layer Neural Networks			*
 *									*
 *			by Andreas Hadjiprocopis			*
 *		    (andreashad2@gmail.com, ex livantes@soi.city.ac.uk)				*
 *		Copyright Andreas Hadjiprocopis, 1994,2007		*
 *									*
 ************************************************************************/

/************************************************************************
 *									*
 *			FILE: OperateBPNetwork.c			*
 *	It contains all the routines for feeding the inputs, training	*
 *	updating the weight matrix etc. etc.				*
 *									*
 ************************************************************************/
 
#include "BPNetworkStandardInclude.h"

/* The prototypes of the Sigmoid and its derivative functions, not used
   by routines in other files */
BPPrecisionType	SigmoidFunction(
	BPPrecisionType	x );
BPPrecisionType	DerivativeSigmoidFunction(
	BPPrecisionType	x );

/************************************************************************
 *									*
 *			FUNCTION: BPNetworkFeedInputs			*
 *	It will accept as inputs an One-Dimensional vector, the Input	*
 *	vector (no difference if it is for training or testing), and the*
 *	number of its components (ie how many inoput nodes?).		*
 *	Also a pointer referencing the network we are talking about must*
 *	be supplied. It will return a normal Value from those specified *
 *	in the error.h file						*
 *									*
 ************************************************************************/
ReturnCode      BPNetworkFeedInputs(
	int		inpnum,
	BPPrecisionType	*inputs,
	BPNetwork	network )
{
	int             node, L0=network->Layer[0];
	Neuron		*pNMap;
	BPPrecisionType	*pI;

	if( inpnum != network->Layer[0] ){
		fprintf(stderr, "Error xxxx\n");
		return( ILLEGAL_PARAMETERS );
	}

	for(node=0,pNMap=&(network->NetworkMap[0][0]),pI=inputs;node<L0;node++,pNMap++,pI++){
		(*pNMap)->NetActivation	=
		(*pNMap)->Output	= *pI;
	}
	return( SUCCESS );
}

/************************************************************************
 *									*
 *		FUNCTION: BPNetworkForwardPropagate			*
 *	It will forward the line of inputs through the network, refernced*
 *	by *network, calculating output and activations for every Neuron*
 *	of that network.						*
 *	This is the function version for the Discrete mode.		*
 *									*
 ************************************************************************/
ReturnCode      BPNetworkForwardPropagate(
	BPNetwork       network )
{
	int             layer,
			node,
			previous_layer_node,
			*pL, *pLminus1;
	BPPrecisionType	*pW, out;
	Neuron		*pNMap1, **pNMap2, *pNMapPre1, **pNMapPre2;

	for(layer=1,pL=&(network->Layer[layer]),pLminus1=&(network->Layer[layer-1]),pNMapPre2=&(network->NetworkMap[layer-1]),pNMap2=&(network->NetworkMap[layer]);layer<network->LayersNum;layer++,pL++,pLminus1++,pNMap2++,pNMapPre2++){
		for(node=0,pNMap1=*pNMap2;node<*pL;node++,pNMap1++){
			(*pNMap1)->NetActivation = 0.0;
			for(previous_layer_node=0,pW=&(((*pNMap1)->Weight[previous_layer_node])),pNMapPre1=*pNMapPre2;previous_layer_node<*pLminus1;previous_layer_node++,pW++,pNMapPre1++){
				(*pNMap1)->NetActivation +=
					(*pW) *
					(*pNMapPre1)->Output;
			}
			(*pNMap1)->NetActivation += (*pNMap1)->Threshold;
			if( layer == (network->LayersNum-1) ){
				if( network->OutputType == Sigmoid ){
					out = (*pNMap1)->Output = SigmoidFunction( (*pNMap1)->NetActivation );
					(*pNMap1)->DiscreteOutput =
						DISCRETE_OUTPUT(network->FirstClassAt, network->LastClassAt, network->ClassSeparation, out);
				} else {
					out = (*pNMap1)->Output = (*pNMap1)->NetActivation;
					(*pNMap1)->DiscreteOutput =
						DISCRETE_OUTPUT(network->FirstClassAt, network->LastClassAt, network->ClassSeparation, out);
				}
			} else {
				(*pNMap1)->Output =
					SigmoidFunction( (*pNMap1)->NetActivation );
/*				network->NetworkMap[layer][node]->DiscreteOutput = 0.0;*/
			}
		}
	}

	return( SUCCESS );
}



/************************************************************************
 *									*
 *			FUNCTION: BPNetworkBackPropagate		*
 *	It will calculate the error at the output nodes and then by	*
 * 	starting from the output nodes it will propagate back the delta	*
 *	values of each Neuron in the Network refernced by *network.	*
 *	A pointer to an One-Dimensional array holding the desired output*
 *	values and their number has to be suplied.			*
 *									*
 ************************************************************************/
ReturnCode      BPNetworkBackPropagate(
	int		outnum,
	BPPrecisionType	*outputs,
	BPNetwork	network )
{
	int             lastlayer = network->LayersNum-1,
			node, layer, next_layer_node, previous_layer_node,
			*pL, *pLL;
	BPPrecisionType	sum = 0.0, discrepancy, *pOut, *pE, *pWC, *pW, *pPW, delta;
	Neuron		*pNMapL1, **pNMapL2, *pNMapN1, **pNMapN2, *pNMap1, **pNMap2,
			*pNMapP1, **pNMapP2;

	if( outnum != network->Layer[lastlayer] ){
		fprintf(stderr, "Error xxxx\n");
		return( ILLEGAL_PARAMETERS );
	}

	/* First let's calculate the discrepancy between actual and
	   desired output values, assuming that a forward propagation
	   has already taken place. */

	pNMapL2 = &(network->NetworkMap[lastlayer]);
	for(node=0,pE=&(network->IndividualError[node]),pOut=&(outputs[node]),pNMapL1=&((*pNMapL2)[node]);node<outnum;node++,pE++,pOut++,pNMapL1++)
		*pE =  (*pOut - (*pNMapL1)->Output);

	/* Now calculate the Delta values at the Output Layer ... */
	/* If the output is not sigmoid but just linear, the derivative is just 1 */
	pNMapL2 = &(network->NetworkMap[lastlayer]);
	if( network->OutputType == Sigmoid ){
		if( network->TrainingType == Continuous ){
			for(node=0,pOut=&(outputs[node]),pNMapL1=&((*pNMapL2)[node]);node<outnum;node++,pOut++,pNMapL1++)
				(*pNMapL1)->Delta =
					((*pOut) - (*pNMapL1)->Output) *
					DerivativeSigmoidFunction( (*pNMapL1)->NetActivation );
		} else {
			for(node=0,pOut=&(outputs[node]),pNMapL1=&((*pNMapL2)[node]);node<outnum;node++,pOut++,pNMapL1++){
				discrepancy = (*pOut) - (*pNMapL1)->Output;
				if( ABS(discrepancy) <= network->ClassSeparation )
					(*pNMapL1)->Delta = 0.0;
				else
					(*pNMapL1)->Delta =
						discrepancy *
						DerivativeSigmoidFunction( (*pNMapL1)->NetActivation );
			}
		}
	} else {
		if( network->TrainingType == Continuous ){
			for(node=0,pOut=&(outputs[node]),pNMapL1=&((*pNMapL2)[node]);node<outnum;node++,pOut++,pNMapL1++)
				(*pNMapL1)->Delta = ((*pOut) - (*pNMapL1)->Output);
		} else {
			for(node=0,pOut=&(outputs[node]),pNMapL1=&((*pNMapL2)[node]);node<outnum;node++,pOut++,pNMapL1++){
				discrepancy = (*pOut) - (*pNMapL1)->Output;
				if( ABS(discrepancy) <= network->ClassSeparation )
					(*pNMapL1)->Delta = 0.0;
				else
					(*pNMapL1)->Delta = discrepancy;
			}
		}
	}

	/* ... And the Delta values for all the hidden layers */
	for(layer=lastlayer-1,pNMap2=&(network->NetworkMap[layer]),pNMapN2=&(network->NetworkMap[layer+1]);layer>0;layer--,pNMap2--,pNMapN2--){
		for(node=0,pNMap1=&((*pNMap2)[node]);node<network->Layer[layer];node++,pNMap1++){
			sum = 0.0;
			for(next_layer_node=0,pNMapN1=&((*pNMapN2)[next_layer_node]);next_layer_node<network->Layer[layer+1];next_layer_node++,pNMapN1++){
				sum += ( (*pNMapN1)->Delta *
					 (*pNMapN1)->Weight[node] );
			}
			(*pNMap1)->Delta = sum * DerivativeSigmoidFunction( (*pNMap1)->NetActivation );
		}
	}

	/* Now calculate the weight changes, but do not apply them yet */
	/* We will cumulate the changes and when the next function is called
	   (BPNetworkAdjustWeights) then we will apply them. */

	for(layer=1,pL=&(network->Layer[layer]),pLL=&(network->Layer[layer-1]),pNMap2=&(network->NetworkMap[layer]),pNMapP2=&(network->NetworkMap[layer-1]);layer<network->LayersNum;layer++,pL++,pLL++,pNMap2++,pNMapP2++){
		for(node=0,pNMap1=&((*pNMap2)[node]);node<*pL;node++,pNMap1++){
			delta = (*pNMap1)->Delta;
			for(previous_layer_node=0,pNMapP1=&((*pNMapP2)[previous_layer_node]),pWC=&((*pNMap1)->WeightChange[previous_layer_node]),pW=&((*pNMap1)->Weight[previous_layer_node]),pPW=&((*pNMap1)->PreviousWeight[previous_layer_node]);previous_layer_node<*pLL;previous_layer_node++,pWC++,pNMapP1++){
				*pWC +=
				      (((network->Beta) *
					delta *
					((*pNMapP1)->Output )) +
				       ((network->Lamda) *
					ABS( (*pW) - (*pPW) )));

			}
			(*pNMap1)->ThresholdChange += network->Beta * delta;
		}
 	}
	

	return( SUCCESS );
}

/************************************************************************
 *									*
 *			FUNCTION: BPNetworkAdjustWeights		*
 *	It will calculate the new weight matrix of the Network refernced*
 *	by *network, after a back propagation has taken place		*
 *	It will take also the weight_change_average_factor parameter	*
 *	This parameter will average the weight change. If weights are	*
 *	updated every time a new exemplar is fed in then this parameter *
 *	is 1. If we are updating when all exemplars have fed in then	*
 *	This parameter is equal to the exemplars number			*
 *									*
 ************************************************************************/
ReturnCode      BPNetworkAdjustWeights(
	BPNetwork       network,
	int		weight_change_average_factor)
{
	int             layer, node,
			previous_layer_node,
			*pL, *pLL;
	BPPrecisionType	*pW, *pWC, *pPW, temp_weight;
	Neuron		*pNMap1, **pNMap2, *pNMapP1, **pNMapP2;

	for(layer=1,pL=&(network->Layer[layer]),pLL=&(network->Layer[layer-1]),pNMap2=&(network->NetworkMap[layer]);layer<network->LayersNum;layer++,pL++,pLL++,pNMap2++,pNMapP2++){
		for(node=0,pNMap1=&((*pNMap2)[node]);node<*pL;node++,pNMap1++){
			for(previous_layer_node=0,pNMapP1=&((*pNMapP2)[previous_layer_node]),pW=&((*pNMap1)->Weight[previous_layer_node]),pPW=&((*pNMap1)->PreviousWeight[previous_layer_node]),pWC=&((*pNMap1)->WeightChange[previous_layer_node]);previous_layer_node<*pLL;previous_layer_node++,pNMapP1++,pPW++,pW++,pWC++){
				*pPW = *pW;
				network->NetworkMap[layer][node]->PreviousWeight[previous_layer_node] = *pW;
				*pW += ((*pWC) / weight_change_average_factor );
				/* When weights have been updated, then zero weight change */
				*pWC = 0.0;
			}
			(*pNMap1)->Threshold += (*pNMap1)->ThresholdChange;
			(*pNMap1)->ThresholdChange = 0.0;
		}
	}
	
	return( SUCCESS );
}

/************************************************************************
 *									*
 *			FUNCTION: ShakeWeights				*
 * It will add/substract a random value from the weights and thresholds *
 * Useful when stack in a local minima					*
 * It will take a pointer to the network.
 * It will take the max noise to add.					*
 * returns nothing.							*
 *									*
 ************************************************************************/
void	ShakeWeights(
	BPNetwork	network,
	BPPrecisionType max_noise )
{
	int		layer, node,
			previous_layer_node,
			*pL, *pLL;
	Neuron		*pNMap1, **pNMap2;
	BPPrecisionType	*pW;

	for(layer=1,pL=&(network->Layer[layer]),pLL=&(network->Layer[layer-1]),pNMap2=&(network->NetworkMap[layer]);layer<network->LayersNum;layer++,pL++,pLL++,pNMap2++){
		for(node=0,pNMap1=&((*pNMap2)[node]);node<*pL;node++,pNMap1++){
			for(previous_layer_node=0,pW=&((*pNMap1)->Weight[previous_layer_node]);previous_layer_node<*pLL;previous_layer_node++,pW++){
				*pW += RandomNumber(max_noise);
			}
		}
	}

}


/************************************************************************
 *									*
 *			FUNCTION: RestoreWeightsToPreviousValues	*
 * It will restore the weight values with their previous value		*
 * (before the last AdjustWeights occurence).				*
 * It takes the pointer to the network					*
 * returns nothing.							*
 *									*
 ************************************************************************/
void	RestoreWeightsToPreviousValues(
	BPNetwork	network )
{
	int		layer, node,
			previous_layer_node,
			*pL, *pLL;

	Neuron		*pNMap1, **pNMap2;
	BPPrecisionType	*pW, *pPW;

	for(layer=1,pL=&(network->Layer[layer]),pLL=&(network->Layer[layer-1]),pNMap2=&(network->NetworkMap[layer]);layer<network->LayersNum;layer++,pL++,pLL++,pNMap2++){
		for(node=0,pNMap1=&((*pNMap2)[node]);node<*pL;node++,pNMap1++){
			for(previous_layer_node=0,pW=&((*pNMap1)->Weight[previous_layer_node]),pPW=&((*pNMap1)->PreviousWeight[previous_layer_node]);previous_layer_node<*pLL;previous_layer_node++,pW++,pPW++){
				*pW = *pPW;
			}
		}
	}
}


/************************************************************************
 *									*
 *			FUNCTION: SigmoidFunction			*
 *	It will return the sigmoid function of value x			*
 *	This function is defined here. Note that if this function change*
 *	then its derivative, below, has to change			*
 *									*
 ************************************************************************/
BPPrecisionType	SigmoidFunction(
	BPPrecisionType	x )
{
	if( ABS(x) > 5.0 ){
/*		fprintf(stderr, "Saturation: value should be -5.0 < %3.3lf < 5.0\n", x);*/
		if( x > 0.0 )
			return((BPPrecisionType )0.9999);
		else
			return((BPPrecisionType )0.0001);
	}

	return( 1.0 / (1.0 + exp((double )(-x))) );
}


/************************************************************************
 *									*
 *		FUNCTION: DerivativeSigmoidFunction			*
 *	It will return the derivative sigmoid function of value x	*
 *									*
 ************************************************************************/
BPPrecisionType	DerivativeSigmoidFunction(
	BPPrecisionType	x )
{
	return( (SigmoidFunction(x) * (1.0 - SigmoidFunction(x))) );
}

/************************************************************************
 *									*
 *		FUNCTION: SaveWeightsInFile				*
 * It will save the current weight state of the network in a file	*
 * The first argunment is the network, the second argument is the	*
 * filename to save them into, the third argument is the mode ie "w"	*
 * for overwriting an existing weight file or "a" for appending		*
 * The fourth argument must be NULL if the second argument is not NULL	*
 * This means that either you supply a filename and the function fopens *
 * it, writes to it and then fcloses it ** OR ** you fopen the file 	*
 * before calling this function and you supply us with a valid stream	*
 * in this case NULL the second and third arguments			*
 *									*
 ************************************************************************/
ReturnCode	SaveWeightsInFile(
	BPNetwork	network,
	char		*Filename,
	char		*Mode,
	FILE		*FileOfWeights)
{ 
	int	i, j, k;
	char	file_mode[5];
	FILE	*filestream;

	if( Filename != (char *)NULL ){
		if( Mode == (char *)NULL )
			strcpy(file_mode, "w"); /* default file open mode */
		else strcpy(file_mode, Mode);
		if( (filestream=fopen(Filename, file_mode)) == NULL ){
			fprintf(stderr, "Could not open file '%s' for operation '%s' in function 'SaveWeightsInFile'.\n", Filename, file_mode);
			return( FILE_OPEN_ERROR );
		}
	} else {
		if( FileOfWeights == (FILE *)NULL ){
			return( ILLEGAL_PARAMETERS );
		}
		filestream = FileOfWeights;
	}

       /* Save Weights in the Weight File */
	for(i=1;i<network->LayersNum;i++)
		for(j=0;j<network->Layer[i];j++){
			fprintf(filestream, "%.16lf\n", (double )(network->NetworkMap[i][j]->Threshold));
			for(k=0;k<network->NetworkMap[i][j]->Inpnum;k++)
				fprintf(filestream, "%.16lf ", (double )(network->NetworkMap[i][j]->Weight[k]));
			fprintf(filestream, "\n");
		}
	/* The last line should contain information about the network and also discrete mode info */
	if( network->TrainingType == Discrete )
		fprintf(filestream, "%d %lf %lf DiscreteTraining\n", network->NumberOfOutputClasses, network->FirstClassAt, network->ClassSeparation);
	else
		fprintf(filestream, "%d %lf %lf ContinuousTraining\n", network->NumberOfOutputClasses, network->FirstClassAt, network->ClassSeparation);

	for(i=0;i<network->LayersNum;i++)
		fprintf(filestream, "%d ", network->Layer[i]);
	fprintf(filestream, "\n");

	if( Filename != (char *)NULL )
		fclose(filestream);

	return( SUCCESS );
}
/************************************************************************
 *									*
 *		FUNCTION: SaveHalfWeightsInFile				*
 * It will save the current weight state of the network in a file	*
 * The first argunment is the network, the second argument is the	*
 * filename to save them into, the third argument is the mode ie "w"	*
 * for overwriting an existing weight file or "a" for appending		*
 * The fourth argument must be NULL if the second argument is not NULL	*
 * This means that either you supply a filename and the function fopens *
 * it, writes to it and then fcloses it ** OR ** you fopen the file 	*
 * before calling this function and you supply us with a valid stream	*
 * in this case NULL the second and third arguments			*
 *									*
 ************************************************************************/
ReturnCode	SaveHalfWeightsInFile(
	BPNetwork	network,
	char		*Filename,
	char		*Mode,
	FILE		*FileOfWeights1,
	FILE		*FileOfWeights2)
{ 
	int	i, j, k;
	char	file_mode[5];
	char	fn1[1000], fn2[1000];
	FILE	*filestream1, *filestream2;

	if( Filename != (char *)NULL ){
		sprintf(fn1, "%s_first_half", Filename);
		sprintf(fn2, "%s_second_half", Filename);
		if( Mode == (char *)NULL )
			strcpy(file_mode, "w"); /* default file open mode */
		else strcpy(file_mode, Mode);
		if( (filestream1=fopen(fn1, file_mode)) == NULL ){
			fprintf(stderr, "Could not open file '%s' for operation '%s' in function 'SaveWeightsInFile'.\n", Filename, file_mode);
			return( FILE_OPEN_ERROR );
		}
		if( (filestream2=fopen(fn2, file_mode)) == NULL ){
			fprintf(stderr, "Could not open file '%s' for operation '%s' in function 'SaveWeightsInFile'.\n", Filename, file_mode);
			return( FILE_OPEN_ERROR );
		}
	} else {
		if( FileOfWeights1 == (FILE *)NULL ){
			return( ILLEGAL_PARAMETERS );
		}
		if( FileOfWeights2 == (FILE *)NULL ){
			return( ILLEGAL_PARAMETERS );
		}
		filestream1 = FileOfWeights1;
		filestream2 = FileOfWeights2;
	}

       /* Save Weights in the Weight File */
	for(i=1;i<network->LayersNum/2;i++)
		for(j=0;j<network->Layer[i];j++){
			fprintf(filestream1, "%.9lf\n", (double )(network->NetworkMap[i][j]->Threshold));
			for(k=0;k<network->NetworkMap[i][j]->Inpnum;k++)
				fprintf(filestream1, "%.9lf ", (double )(network->NetworkMap[i][j]->Weight[k]));
			fprintf(filestream1, "\n");
		}
	/* The last line should contain information about the network and also discrete mode info */
	if( network->TrainingType == Discrete )
		fprintf(filestream1, "%d %lf %lf DiscreteTraining\n", network->NumberOfOutputClasses, network->FirstClassAt, network->ClassSeparation);
	else
		fprintf(filestream1, "%d %lf %lf ContinuousTraining\n", network->NumberOfOutputClasses, network->FirstClassAt, network->ClassSeparation);

	for(i=0;i<network->LayersNum/2;i++)
		fprintf(filestream1, "%d ", network->Layer[i]);
	fprintf(filestream1, "\n");

	if( Filename != (char *)NULL )
		fclose(filestream1);

       /* Save Weights in the Weight File (second half) */
	for(i=network->LayersNum/2;i<network->LayersNum;i++)
		for(j=0;j<network->Layer[i];j++){
			fprintf(filestream2, "%.9lf\n", (double )(network->NetworkMap[i][j]->Threshold));
			for(k=0;k<network->NetworkMap[i][j]->Inpnum;k++)
				fprintf(filestream2, "%.9lf ", (double )(network->NetworkMap[i][j]->Weight[k]));
			fprintf(filestream2, "\n");
		}
	/* The last line should contain information about the network and also discrete mode info */
	if( network->TrainingType == Discrete )
		fprintf(filestream2, "%d %lf %lf DiscreteTraining\n", network->NumberOfOutputClasses, network->FirstClassAt, network->ClassSeparation);
	else
		fprintf(filestream2, "%d %lf %lf ContinuousTraining\n", network->NumberOfOutputClasses, network->FirstClassAt, network->ClassSeparation);

	for(i=network->LayersNum/2;i<network->LayersNum;i++)
		fprintf(filestream2, "%d ", network->Layer[i]);
	fprintf(filestream2, "\n");

	if( Filename != (char *)NULL )
		fclose(filestream2);

	return( SUCCESS );
}
/************************************************************************
 *									*
 *		FUNCTION: LoadWeightsFromFile				*
 * It will load the weight state of a network from a file		*
 * The first argunment is the network, the second argument is the	*
 * filename to load them from.						*
 * The third argument must be NULL if the second argument is not NULL	*
 * This means that either you supply a filename and the function fopens *
 * it, writes to it and then fcloses it ** OR ** you fopen the file 	*
 * before calling this function and you supply us with a valid stream	*
 * in this case NULL the second argument				*
 *									*
 ************************************************************************/
ReturnCode	LoadWeightsFromFile(
	BPNetwork	network,
	char		*Filename,
	FILE		*FileOfWeights,
	char		LoadParametersFromWeightsFileFlag) /* the last-but-one line of the weights file contains first_class,
 						class_separation etc. TRUE: ignore command line options about these and load the values from the file,
						FALSE: ignore values in file, use only user */
{ 
	int	i, j, k;
	FILE	*filestream;
	char	buffer[MAX_CHARS_PER_LINE];

	if( Filename != (char *)NULL ){
		if( (filestream=fopen(Filename, "r")) == NULL ){
			fprintf(stderr, "Could not open file '%s' for reading in function 'LoadWeightsFromFile'.\n", Filename);
			return( FILE_OPEN_ERROR );
		}
	} else {
		if( FileOfWeights == (FILE *)NULL ){
			return( ILLEGAL_PARAMETERS );
		}
		filestream = FileOfWeights;
	}

	
	/* The last line contains the network architecture, let's verify */
	do fgets(buffer, MAX_CHARS_PER_LINE, filestream); while( !feof(filestream) ); fseek(filestream, -strlen(buffer), SEEK_END);
	int	n;
	for(i=0;i<network->LayersNum;i++){
		if( fscanf(filestream, "%d", &n) != 1 ){
			fprintf(stderr, "LoadWeightsFromFile : error in reading network architecture from file, premature end of file, number of layers specified in weights file (%d) is less than the number of layers specified in command line (%d). #1\n", i+1, network->LayersNum);
			return( ILLEGAL_PARAMETERS );
		}
		if( feof(filestream) ){
			fprintf(stderr, "LoadWeightsFromFile : error in reading network architecture from file, premature end of file, number of layers specified in weights file (%d) is less than the number of layers specified in command line (%d). #2\n", i+1, network->LayersNum);
			return( ILLEGAL_PARAMETERS );
		}
		if( n != network->Layer[i] ){
			fprintf(stderr, "LoadWeightsFromFile : mismatch in network architecture specified in weights file and existing network, layer %d must have %d neurons but weights file specifies %d neurons, check last line of weights file for architecture.\n", i+1, network->Layer[i], n);
			return( ILLEGAL_PARAMETERS );
		}
	}
	rewind(filestream);

	/* Load the Weights */
	/* First is a threshold value for the unit and then all the weights connecting it to the next layer units*/
	for(i=1;i<network->LayersNum;i++){
		for(j=0;j<network->Layer[i];j++){
			if( fscanf(filestream, "%lf", &(network->NetworkMap[i][j]->Threshold)) != 1 ){
				fprintf(stderr, "LoadWeightsFromFile : error in reading network architecture from file, fscanf could not read one threshold value.\n");
				return( ILLEGAL_PARAMETERS );
			}
			if( feof(filestream) ){
				fprintf(stderr, "LoadWeightsFromFile : error in reading network architecture from file, premature end of file, are you sure you have the correct architecture - check last line of weights file (1).\n");
				return( ILLEGAL_PARAMETERS );
			}
			for(k=0;k<network->NetworkMap[i][j]->Inpnum;k++){
				if( fscanf(filestream, "%lf", &(network->NetworkMap[i][j]->Weight[k])) != 1 ){
					fprintf(stderr, "LoadWeightsFromFile : error in reading network architecture from file, fscanf could not read one weight value.\n");
					return( ILLEGAL_PARAMETERS );
				}
				if( feof(filestream) ){
					fprintf(stderr, "LoadWeightsFromFile : error in reading network architecture from file, premature end of file, are you sure you have the correct architecture - check last line of weights file (2).\n");
					return( ILLEGAL_PARAMETERS );
				}
			}
		}
	}

/* last lines of weight file:
	0 0.000000 1.000000 ContinuousTraining
	47 175 235 75 3
*/
	/* The last but one line should contain information about the network and also discrete mode info */
	int	_noc;
	BPPrecisionType	_fc, _cs;
	fscanf(filestream, "%d%lf%lf%*[ \n]%[^ \n]%*[ \n]", &_noc, &_fc, &_cs, buffer);
	if( LoadParametersFromWeightsFileFlag == TRUE ){
		network->NumberOfOutputClasses = _noc;
		network->FirstClassAt = _fc;
		network->ClassSeparation = _cs;
		if( !strcmp(buffer, "DiscreteTraining") )
			network->TrainingType = Discrete;
		else
			network->TrainingType = Continuous;
		if( network->NumberOfOutputClasses > 0 )
			network->NetworkType = Discrete;
		else	network->NetworkType = Continuous;
	}
	if( Filename != (char *)NULL ) fclose(filestream);

	return( SUCCESS );
}

/************************************************************************
 *									*
 *		FUNCTION: MonitorEW					*
 *  It will dump the state of all the weights, thresholds, in columns,  *
 * and the output error							*
 *									*
 ************************************************************************/
ReturnCode	MonitorEW(
	int		iteration,
	BPNetwork	network,
	FILE		*filestream)
{ 
	int	i, j, k;
	char	file_mode[5];

	if( filestream == (FILE *)NULL ) return( ILLEGAL_PARAMETERS );

	fprintf(filestream, "%d ", iteration);
       /* Save Weights, thresholds and error */
	for(i=1;i<network->LayersNum;i++)
		for(j=0;j<network->Layer[i];j++){
			fprintf(filestream, "%.9lf ", (double )(network->NetworkMap[i][j]->Threshold));
			for(k=0;k<network->NetworkMap[i][j]->Inpnum;k++)
				fprintf(filestream, "%.9lf ", (double )(network->NetworkMap[i][j]->Weight[k]));
		}
	fprintf(filestream, "%.9lf\n", (double )(network->TotalError));

//	fprintf(filestream, "# END\n");

	return( SUCCESS );
}
BPPrecisionType DiscreteOutput(
	BPNetwork       network,
	BPPrecisionType	x )
{
	return( (network->ClassSeparation)*((BPPrecisionType )ROUND((x-network->FirstClassAt)/(network->ClassSeparation))) );
}

void	SignalHandler(
	int		signal_number )
{
	if( SignalCaught == SIGNAL_QUEUE_IS_EMPTY ){
		SignalCaught = signal_number;
	} else {
		if( signal_number == SIGINT ){
			SignalCaught = signal_number;
		} else {
			fprintf(stderr, "SignalHandler: Previous signal (%d) awaiting processing. New signal arrived is %d\n", SignalCaught, signal_number);
			psignal(signal_number, "previous signal description");
			psignal(SignalCaught, "current signal description");
		}
	}
}


int	SetSignalHandler(void)
{
	int	my_pid = getpid(), i;

	/* termination signals */
	if( signal(SIGINT, SignalHandler) == SIG_ERR )
		fprintf(stderr, "SetSignalHandler: Can't catch signals 'SIGINT'.\n");
	if( signal(SIGQUIT, SignalHandler) == SIG_ERR )
		fprintf(stderr, "SetSignalHandler: Can't catch signals 'SIGQUIT'.\n");
	if( signal(SIGTERM, SignalHandler) == SIG_ERR )
		fprintf(stderr, "SetSignalHandler: Can't catch signals 'SIGTERM'.\n");

	/* user defined actions */
	if( signal(SIGUSR1, SignalHandler) == SIG_ERR )
		fprintf(stderr, "SetSignalHandler: Can't catch signals 'SIGUSR1'.\n");
	if( signal(SIGUSR2, SignalHandler) == SIG_ERR )
		fprintf(stderr, "SetSignalHandler: Can't catch signals 'SIGUSR2'.\n");
#ifdef SIGRTMIN
	if( signal(SIGRTMIN, SignalHandler) == SIG_ERR )
		fprintf(stderr, "SetSignalHandler: Can't catch signals 'SIGRTMIN'.\n");
#endif
	return(my_pid);
}

void	RemoveSignalHandler(
	void )
{
/*	if( sigignore(SIGINT) == -1 )
		fprintf(stderr, "RemoveSignalHandler: Can't remove signal 'SIGINT'.\n");
	if( sigignore(SIGUSR1) == -1 )
		fprintf(stderr, "RemoveSignalHandler: Can't remove signal 'SIGUSR1'.\n");
	if( sigignore(SIGUSR2) == -1 )
		fprintf(stderr, "RemoveSignalHandler: Can't remove signal 'SIGUSR2'.\n");
	if( sigignore(SIGQUIT) == -1 )
		fprintf(stderr, "RemoveSignalHandler: Can't remove signal 'SIGQUIT'.\n");
	if( sigignore(SIGTERM) == -1 )
		fprintf(stderr, "RemoveSignalHandler: Can't remove signal 'SIGTERM'.\n");
	if( sigignore(SIGRTMIN) == -1 )
		fprintf(stderr, "RemoveSignalHandler: Can't remove signal 'SIGRTMIN'.\n");
*/
}
	
void	CatchSignal(
	int		signal_number )
{
	if( signal(signal_number, SignalHandler) == SIG_ERR ){
		fprintf(stderr, "RemoveSignalHandler: Can't catch signal %d.\n", signal_number);
		psignal(signal_number, "in this OS the signal is");
	}
}

void	ResetSignalHandler(
	void )
{
	SignalCaught = SIGNAL_QUEUE_IS_EMPTY;
}

/************************************************************************
 *									*
 *		FUNCTION: BPNetworkCalculateDerivatives			*
 *	It will calculate the value of the derivative of the neural	*
 *	network (referenced by *network), given a set of inputs, weights*
 *	and thresholds.							*
 *									*
 *	It is necessary that there is a call to ForwardPropagate	*
 *	prior to using this function. This is not done here.		*
 *									*
 ************************************************************************/
ReturnCode	BPNetworkCalculateDerivatives(
	BPNetwork       network )
{
	int		layer,
			node,
			previous_layer_node, input,
			ScratchSize = network->Layer[0];
	BPPrecisionType	*Scratch, *SecondScratch;

	for(layer=1;layer<network->LayersNum;layer++)
		ScratchSize = MAX(ScratchSize, network->Layer[layer]);

	if( (Scratch=(BPPrecisionType *)calloc(ScratchSize, sizeof(BPPrecisionType))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += ScratchSize * sizeof(BPPrecisionType);

	if( (SecondScratch=(BPPrecisionType *)calloc(ScratchSize, sizeof(BPPrecisionType))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += ScratchSize * sizeof(BPPrecisionType);

	/* We will calculate derivatives for each of the inputs x1, x2, ... xLayer[0] */
	for(input=0;input<network->Layer[0];input++){
		/* The first layer factor is the i^th column of the weight matrix connecting the input
		   layer to the next one. */
		for(node=0;node<network->Layer[1];node++)
			Scratch[node] = network->NetworkMap[1][node]->Weight[input] *
					DerivativeSigmoidFunction(network->NetworkMap[1][node]->NetActivation);
		for(layer=2;layer<network->LayersNum;layer++){
			for(node=0;node<network->Layer[layer];node++){
				SecondScratch[node] = 0;
				for(previous_layer_node=0;previous_layer_node<network->Layer[layer-1];previous_layer_node++)
					SecondScratch[node] += (network->NetworkMap[layer][node]->Weight[previous_layer_node] *
								Scratch[previous_layer_node] );
			}
			for(node=0;node<network->Layer[layer];node++)
				Scratch[node] = DerivativeSigmoidFunction(network->NetworkMap[layer][node]->NetActivation) *
						SecondScratch[node];
		}
		for(node=0;node<network->Layer[layer];node++){
			SecondScratch[node] = 0;
			for(previous_layer_node=0;previous_layer_node<network->Layer[layer-1];previous_layer_node++)
				SecondScratch[node] += (network->NetworkMap[layer][node]->Weight[previous_layer_node] *
							Scratch[previous_layer_node] );
		}
		if( network->OutputType == Sigmoid ){
			for(node=0;node<network->Layer[layer];node++)
				Scratch[node] = DerivativeSigmoidFunction(network->NetworkMap[layer][node]->NetActivation) *
						SecondScratch[node];
		} else {
			for(node=0;node<network->Layer[layer];node++)
				Scratch[node] = network->NetworkMap[layer][node]->NetActivation *
						SecondScratch[node];
		}
		network->Derivatives[input] = Scratch[0];
	}

	free(Scratch); free(SecondScratch);
	return( SUCCESS );
}

