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

/************************************************************************
 *									*
 *		FILE: BPNetworkConstants.h				*
 *	It contains all the constants of the Engine functions.		*
 *									*
 ************************************************************************/
 
#define	MAX_WEIGHT_VALUE		0.96
#define	MAX_THRESHOLD_VALUE		0.96
#define	DEFAULT_BETA_VALUE		0.09
#define	DEFAULT_LAMDA_VALUE		0.0

#define	TRUE				1
#define	FALSE				0
#define	EXEMPLAR_WEIGHT_UPDATE_METHOD	0x10
#define EPOCH_WEIGHT_UPDATE_METHOD	0x11

#define	SIGNAL_QUEUE_IS_EMPTY		-1

/*By 5%*/
#define	CHANGE_BETA_STEP		0.05

#define	MAX_CHARS_PER_LINE		25000

#define	ABS(a)				( ((a)<0) ? (-(a)):(a) )
#define	ROUND(a)			( (((int )(a+0.5))!=((int )(a))) ? ((int )(a+1)):((int )(a)) )
#define	LIMIT(first, last, x)		( ((x)<(first)) ? (first) : (((x)>(last))?(last):(x)) )
/* this one maps the continuous valued output, x, into discrete values given first, last class and step */
#define DISCRETE_OUTPUT(first, last, step, x)	( (first) + ((step)*((BPPrecisionType )ROUND( ABS((first)-(LIMIT(first,last,x)))/(step)))) )
/* this one gives the class index of hte continous valued output, x, e.g. 1, 2, 3, etc. this can help to map to arbitrary
   classes, e.g. 'A','B' etc. for presenting the output better, this index starts from zero and can be used to index an array of chars
   or strings as output */
#define DISCRETE_OUTPUT_CLASS_INDEX(first, last, step, x)	( ROUND( ABS((first)-(LIMIT(first,last,x)))/(step)) )
#define	MAX(a, b)			( ((a)<(b)) ? (b):(a) )
#define	MIN(a, b)			( ((a)<(b)) ? (a):(b) )
