dnl	      ARTIFICIAL NEURAL NETWORKS SOFTWARE		      
dnl								       
dnl   An Error Back Propagation Neural Network Engine for Feed-Forward    
dnl		      Multi-Layer Neural Networks
dnl		      
dnl			by Andreas Hadjiprocopis
dnl		  (andreashad2@gmail.com)
dnl	      Copyright Andreas Hadjiprocopis, 1994,2007
dnl AX_CC_MAXOPT macros (and all contents of the m4 directory)
dnl were 'borrowed' from fftw-3.1 by Matteo Frigo and Steven G. Johnson (thanks!)
dnl look at :  http://www.fftw.org/mailman/listinfo/fftw-announce

AC_INIT([NNengine],[12.0],[andreashad2@gmail.com])
m4_pattern_allow([AC_CONFIG_MACRO])
AC_CONFIG_MACRO_DIR([m4])
AC_CANONICAL_HOST
AC_CANONICAL_TARGET
case "${target_os}" in
	*solaris* | *linux*)		
		echo "solaris or linux system"
		CPPFLAGS="-DNNENGINE_LINUXSOLARIS_VERSION"
		;;
	djgpp | *mingw32* | *emx*)
		echo dos
		CPPFLAGS="-DNNENGINE_WINDOWS_VERSION"
		;;
	*bsd*)
		echo BSDS
		CPPFLAGS="-DNNENGINE_BSD_VERSION"
		;;
	*darwin*)
		echo "mac"
		CPPFLAGS="-DNP_LINUXSOLARIS_VERSION"
		;;
esac

echo target os: "${target_os}"
echo target cpu: "${target_cpu}"
echo build cpu : "${build_cpu}"

dnl determine CFLAGS first
AX_CC_MAXOPT

CFLAGS="$CFLAGS -Werror"

cpuid=""
cpuspeed=""
type -P x86info >& /dev/null
if test $? -eq 0 ; then
	x86info -f > cpuid
	cpuid=`grep "CPU Model" cpuid | cut -d':' -f 2`
	echo "$cpuid" | grep -i unknown >& /dev/null
	if test $? -eq 0 ; then cpuid=`grep "Processor name string" cpuid | cut -d':' -f 2 | cut -d'@' -f 1`; fi
	cpuspeed=`grep "Processor name string" cpuid | cut -d':' -f 2 | cut -d'@' -f 2`
else
	# try mac
	type -P /usr/sbin/system_profiler >& /dev/null
	if test $? -eq 0 ; then
		cpuid=`/usr/sbin/system_profiler | grep "Processor Name" | cut -d':' -f 2`
		cpuspeed=`/usr/sbin/system_profiler | grep "Processor Speed" | cut -d':' -f 2`
	else
		echo "can't figure out cpu architecture using -march=native which might not work in some cases (see mac)"
		CFLAGS="$CFLAGS -march=native"
	fi
fi

echo CPU ID: "$cpuid"
echo CPU SPEED: "$cpuspeed"
case "$cpuid" in
	*thlon*64* )
		CFLAGS="$CFLAGS -m64 -march=athlon64-sse3"
		;;
	*Intel*Core*i* )
		CFLAGS="$CFLAGS -march=nocona -O2 -pipe -m64"
		# when Apple updates to > 4.3 then -march=core2 -mtune=generic -O2 -pipe
		;;
	*Intel*Xeon* )
		CFLAGS="$CFLAGS -march=nocona -m64"
		;;
esac
echo CFLAGS USED: "$CFLAGS"


dnl Use the name and version of your package (which doesn't have to match
dnl the names of your executables):
AM_INIT_AUTOMAKE()

dnl Checks for programs.
dnl find out if our favourite compilers exist - the most favourable is
dnl placed last!
AC_PROG_CC([cc, gcc, gcc3, gcc-4.1])
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET

LT_INIT

AC_CONFIG_FILES([Makefile
	src/Makefile
	src/C/Makefile
])
AC_OUTPUT
