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
#include "BPNetworkStandardInclude.h"

/************************************************************************
 *									*
 *			FILE: CreateBPNetwork.c				*
 *	This file contatins the routines for creating and initialising	*
 *	a neural network including the CreateNeuron routine for creating*
 *	and initialising a neuron structure. It also contains the 	*
 *	routine to destroy the network.					*
 *									*
 ************************************************************************/

/********************************************************************************
 * The CreateBPNetwork Function will take an already allocated pointer		*
 * to the structure BPNetwork and copy the network dimensions into it		*
 * It will free all the dimensions-dependent structures from memory		*
 * such as the Neural Network and the error pointers in order to be able	*
 * to calloc new ones using the new dimensions. It will accept three		*
 * parameters as follows:							*
 *      ... network, is the network						*
 *      ... layersnum, is the total number of layers including input/output	*
 *      ... *nodesnum,is a One Dimensional array which contains the number of	*
 *          nodes in each of the network layers.				*
 ********************************************************************************/
ReturnCode      CreateBPNetwork(
	int             layersnum,
	int             *nodesnum,
	BPNetwork       network )
{
	int		i, *pL, *pN;
	
	/* First of all destroy any previous network structure */
	/* Note that the pointer to the structure, network, is not destroyed */
	/* The pointer Networkmap is NULLed though */
	DestroyBPNetwork(network);

	/* Initialise the total number of layers */
	network->LayersNum = layersnum;

	if( (network->Layer=(int *)malloc(layersnum*sizeof(int))) == NULL ){
		fprintf(stderr, "CreateBPNetwork() : could not allocate %lu bytes for %d integers for network->Layer.\n", layersnum*sizeof(int), layersnum);
		return MEMORY_ALLOCATION_FAILED;
	}

	/* load the number of nodes in each Layer[] */
	for(i=0,pL=network->Layer,pN=nodesnum;i<layersnum;i++,pL++,pN++) *pL = *pN;

	network->Beta	= DEFAULT_BETA_VALUE;
	network->Lamda	= DEFAULT_LAMDA_VALUE;

	return( SUCCESS );
}                        	
/* End of the CreateBPNetwork Function */

/*****************************************************************************
 *      The InitialiseBPNetwork Function will only one parameter	     *
 *      ... network, is the network					     *
 * It will allocate in memory the dimensions dependent structures	     *
 *****************************************************************************/
ReturnCode      InitialiseBPNetwork(
	BPNetwork       network )
{
	int             i, j, ret,
			layer, node,
			inpnum = 1, outnum,
			*pL;
	Neuron		**pNMap2, *pNMap1;
	/* Allocate memory for a Two-Dimensional array of type Neuron to store
	   the memory addresses of the Neurons in the Two-Dimensional Network */
	if( (network->NetworkMap=(Neuron **)calloc(network->LayersNum, sizeof(Neuron *))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += network->LayersNum * sizeof(Neuron *);

	for(i=0,pL=network->Layer,pNMap2=network->NetworkMap;i<network->LayersNum;i++,pL++,pNMap2++){
		if( (*pNMap2=(Neuron *)calloc(*pL, sizeof(Neuron))) == NULL )
			return( MEMORY_ALLOCATION_FAILED );
		TotalMemoryAllocated += (*pL) * sizeof(Neuron);
		for(j=0,pNMap1=*pNMap2;j<*pL;j++,pNMap1++){
			if( (*pNMap1=(Neuron )calloc(1, sizeof(struct _NEURON_STRUCT))) == NULL )
				return( MEMORY_ALLOCATION_FAILED );
			TotalMemoryAllocated += sizeof(struct _NEURON_STRUCT);
		}
	}

	/* Allocate memory for a One-Dimensional Array of BPPrecisionTypes to store
	   the error at each output node */
	if( (network->IndividualError=(BPPrecisionType *)calloc(network->Layer[network->LayersNum-1], sizeof(BPPrecisionType))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += network->Layer[network->LayersNum-1] * sizeof(BPPrecisionType);

	/* Allocate memory for a One-Dimensional array of BPPrecision type, to hold the derivatives */
	if( (network->Derivatives=(BPPrecisionType *)calloc(network->Layer[0], sizeof(BPPrecisionType))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += network->Layer[0] * sizeof(BPPrecisionType);

        /* Create the Neurons */
	inpnum=1;
	for(layer=0,pNMap2=network->NetworkMap;layer<network->LayersNum;layer++,pNMap2++){
		if( layer>0 )
			inpnum = network->Layer[layer-1];
		outnum = 1;
		if( layer<(network->LayersNum-1) )
			outnum = network->Layer[layer+1];
		for(node=0,pNMap1=*pNMap2;node<network->Layer[layer];node++,pNMap1++)
			if( (ret=CreateNeuron(layer, node, inpnum, outnum, *pNMap1)) != SUCCESS ){
				fprintf(stderr, "InitialiseBPNetwork : call to CreateNeuron has failed for %d inputs and %d outputs, layer %d, return code was %d.\n", inpnum, outnum, layer, ret);
				return ret;
			}
	}
	return( SUCCESS );
}
/* End of the InitialiseBPNetwork Function */


/************************************************************************
 *			FUNCTION: CreateNeuron				*
 *	It will create a neuron structure, initialise it according to	*
 *  ... layer, node: this neuron is the node node on the layer layer	*
 *  ... inpnum: Is the number of input nodes of that neuron		*
 *  ... outnum: Is the number of output nodes				*
 * 	Finally it will return a pointer to that structure: neuron	*
 ************************************************************************/
ReturnCode      CreateNeuron(
	int     layer,
	int     node,
	int     inpnum,
	int     outnum,
	Neuron  neuron )
{
	int     i;
	BPPrecisionType	*pW, *pPW, *pWC;

	/* The Integer part */
	neuron->Layer = layer;
	neuron->Node  = node;
	neuron->Inpnum= inpnum;
	neuron->Outnum= outnum;

	/* The Initialisation part */
	neuron->NetActivation	= 0.0;
	neuron->Output		= 0.0;
	neuron->DiscreteOutput	= 0.0;
	neuron->Delta		= 0.0;
	
	/* The Float/Double part */
	/* The One-Dimensional Weight, WeightChange, and Previousweight array */
	if( (neuron->Weight=(BPPrecisionType *)calloc(inpnum, sizeof(BPPrecisionType))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += inpnum * sizeof(BPPrecisionType);

	if( (neuron->WeightChange=(BPPrecisionType *)calloc(inpnum, sizeof(BPPrecisionType))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += inpnum * sizeof(BPPrecisionType);

	if( (neuron->PreviousWeight=(BPPrecisionType *)calloc(inpnum, sizeof(BPPrecisionType))) == NULL )
		return( MEMORY_ALLOCATION_FAILED );
	TotalMemoryAllocated += inpnum * sizeof(BPPrecisionType);

	/* Random numbers will fill the weight array and the threshold field */
	/* But first check if the layer is the first, It has unit weights */
	pW = neuron->Weight; pPW = neuron->PreviousWeight; pWC = neuron->WeightChange;
	if( layer == 0 ){
		for(i=0;i<inpnum;i++,pW++,pPW++,pWC++){
			*pW = *pPW = 1.0;
			*pWC = 0.0;
		}
	} else
		for(i=0;i<inpnum;i++,pW++,pPW++,pWC++){
			*pW = *pPW = RandomNumber(MAX_WEIGHT_VALUE);
			*pWC = 0.0;
		}
	/* The threshold field is random everywhere */
	neuron->Threshold = RandomNumber(MAX_THRESHOLD_VALUE);
	neuron->ThresholdChange = 0.0;

	/* That is the end of the initialisation/creation */
	/* Now the Neuron's memory address is at BPNetwork[networknum].NetworkMap[layer][node]
	   and any field of the Neuron structure can be accessed from there */

	return( SUCCESS );

}
/* End of the CreateNeuron Function */



/************************************************************************************************
 *      The DestroyBPNetwork Function will free all the dynamically allocated fields in the     *
 *      BPNetwork and Neuron structure.							 *
 *	network is the pointer to the network we want to destroy				*
 ************************************************************************************************/
ReturnCode      DestroyBPNetwork(
	BPNetwork       network )
{
	int	layer,
		node;

	
	/* We have to free all the Memory we allocated dynamically using calloc, first */
	/* Go through each Neuron in the Network and free the *Weight array */
	
	if( network->Derivatives != NULL ) free(network->Derivatives);
	if( network->IndividualError != NULL ) free(network->IndividualError);
	if( network->NetworkMap == NULL ) return( SUCCESS );
	for(layer=0;layer<network->LayersNum;layer++){
		for(node=0;node<network->Layer[layer];node++){
			free(network->NetworkMap[layer][node]->Weight);
			free(network->NetworkMap[layer][node]->PreviousWeight);
			free(network->NetworkMap[layer][node]->WeightChange);
			free(network->NetworkMap[layer][node]);
		}
		free(network->NetworkMap[layer]);
	}
	if( network->Layer != NULL ) free(network->Layer);
	free(network->NetworkMap);
	/* All the Neurons are gone now... */

	return( SUCCESS );     
}
/* End of DestroyBPNetwork Function */


/* End of CreateBPNetwork.c file */
