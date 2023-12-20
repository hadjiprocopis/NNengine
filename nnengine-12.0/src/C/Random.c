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

/* This is the number of random numbers that Iam allowed to extract from a
   single seed. If that number is exceeded (ie if I called the RandomNumber() )
   function more times than that constant then a new seed is used :
   	seed(time(0));
   the value of this constant is 2**25, 25 is arbitrary but should be less than
   the period of my random number generator, random()*/
#define	MAX_RANDOM_ARRAY_DEPTH	33554432

/* This is the period (or something less 2**31 or so) of my unix system random
   number generator. Note that this number is less than the actual period
   because an int would never accept that silly value od 2**32-1, i dont know why.. */
#define	MAX_POSSIBLE_RANDOM	1073741824

/* a random number generator for the PC */
#ifdef	NNENGINE_MSDOS_VERSION
BPPrecisionType   RandomNumber(
	BPPrecisionType	max )
{
	int             a_rand;
	BPPrecisionType	the_rand;

	a_rand = random(10000);
	the_rand = (a_rand%2 == 0 ? -1.0:1.0) * ((BPPrecisionType )a_rand) * max / 10000;
	return(the_rand);

}
#endif





/********************************************************************************
 *	RandomNumber: it will accept a BPPreT number as a maximum random	*
 *	number (note: +/- max), and it will return a BPPreT random number	*
 *	in the range +max/-max.							*
 *	The technique to adjust the range of the number is to map the actual 	*
 *	number in the +map / -map region, using the equation below.		*
 ********************************************************************************/
BPPrecisionType	RandomNumber(
	BPPrecisionType	max )
{
	
	static	int	HowManyTimes = 0;
	double		a_rand = 0.0;
	

	if( HowManyTimes == 0 )	srand48(Seed(0L));
	if( HowManyTimes++ >= MAX_RANDOM_ARRAY_DEPTH ){
		HowManyTimes = 0;
		srand48(Seed(0L));
	}

	/* Note max cannot be zero or negative */
	/*fprintf(stderr, "random = %ld\n", mrand48());*/
	a_rand = ((2*max)/(MAX_POSSIBLE_RANDOM-1))*(lrand48()/2) - max;


	return((BPPrecisionType )(a_rand));
}	

long	Seed(
	long	a_seed )
{
	static	long	current_seed = 1974L; /* This initialiser does not apply */

	if( a_seed == 0L ){
		return current_seed;
	}
	current_seed = a_seed;
}
