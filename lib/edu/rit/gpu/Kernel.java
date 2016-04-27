//******************************************************************************
//
// File:    Kernel.java
// Package: edu.rit.gpu
// Unit:    Interface edu.rit.gpu.Kernel
//
// This Java source file is copyright (C) 2014 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.gpu;

/**
 * Interface Kernel specifies the base interface for a GPU kernel function.
 * To use a GPU kernel function:
 * <OL TYPE=1>
 * <P><LI>
 * Write the C or C++ code for a CUDA module containing one or more kernel
 * functions to be executed on the GPU. The kernel function(s) must be declared
 * to use "C" call syntax; for example:
 * <BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;extern "C" __global__ void foo (...)</TT>
 * <BR>Compile the CUDA module using the <TT>nvcc</TT> compiler.
 * <P><LI>
 * For each kernel function in the CUDA module, write a <I>separate</I> Java
 * interface, as follows.
 * <UL>
 * <LI>
 * The interface name can be whatever you want.
 * <LI>
 * The interface must extend interface Kernel.
 * <LI>
 * The interface must declare <I>one and only one</I> method.
 * <LI>
 * The method's name must be identical to that of the kernel function. Suppose
 * the method's name is <TT>foo</TT>.
 * <LI>
 * The <TT>foo</TT> method must return void.
 * <LI>
 * The <TT>foo</TT> method's arguments must appear in the same order as those
 * of the kernel function.
 * <LI>
 * Declare each <TT>foo</TT> method argument's data type to correspond with the
 * kernel function argument's data type as listed in the table below. (Kernel
 * function arguments of other types are not supported at this time.)
 * <P>
 * <TABLE BORDER=0 CELLPADDING=0 CELLSPACING=0>
 * <TR><TD><I>Java data type</I></TD><TD WIDTH=20> </TD><TD><I>C data type</I></TD></TR>
 * <TR><TD><TT>boolean</TT></TD><TD WIDTH=20> </TD><TD><TT>int</TT></TD></TR>
 * <TR><TD><TT>byte</TT></TD><TD WIDTH=20> </TD><TD><TT>char</TT>, <TT>unsigned char</TT></TD></TR>
 * <TR><TD><TT>char</TT></TD><TD WIDTH=20> </TD><TD><TT>unsigned short int</TT></TD></TR>
 * <TR><TD><TT>short</TT></TD><TD WIDTH=20> </TD><TD><TT>short int</TT>, <TT>unsigned short int</TT></TD></TR>
 * <TR><TD><TT>int</TT></TD><TD WIDTH=20> </TD><TD><TT>int</TT>, <TT>unsigned int</TT></TD></TR>
 * <TR><TD><TT>long</TT></TD><TD WIDTH=20> </TD><TD><TT>long long int</TT>, <TT>unsigned long long int</TT></TD></TR>
 * <TR><TD><TT>float</TT></TD><TD WIDTH=20> </TD><TD><TT>float</TT></TD></TR>
 * <TR><TD><TT>double</TT></TD><TD WIDTH=20> </TD><TD><TT>double</TT></TD></TR>
 * <TR><TD>{@linkplain GpuByteArray}</TD><TD WIDTH=20> </TD><TD><TT>char*</TT>, <TT>unsigned char*</TT></TD></TR>
 * <TR><TD>{@linkplain GpuShortArray}</TD><TD WIDTH=20> </TD><TD><TT>short int*</TT>, <TT>unsigned short int*</TT></TD></TR>
 * <TR><TD>{@linkplain GpuIntArray}</TD><TD WIDTH=20> </TD><TD><TT>int*</TT>, <TT>unsigned int*</TT></TD></TR>
 * <TR><TD>{@linkplain GpuLongArray}</TD><TD WIDTH=20> </TD><TD><TT>long long int*</TT>, <TT>unsigned long long int*</TT></TD></TR>
 * <TR><TD>{@linkplain GpuFloatArray}</TD><TD WIDTH=20> </TD><TD><TT>float*</TT></TD></TR>
 * <TR><TD>{@linkplain GpuDoubleArray}</TD><TD WIDTH=20> </TD><TD><TT>double*</TT></TD></TR>
 * <TR><TD>{@linkplain GpuByteMatrix}</TD><TD WIDTH=20> </TD><TD><TT>char**</TT>, <TT>unsigned char**</TT></TD></TR>
 * <TR><TD>{@linkplain GpuShortMatrix}</TD><TD WIDTH=20> </TD><TD><TT>short int**</TT>, <TT>unsigned short int**</TT></TD></TR>
 * <TR><TD>{@linkplain GpuIntMatrix}</TD><TD WIDTH=20> </TD><TD><TT>int**</TT>, <TT>unsigned int**</TT></TD></TR>
 * <TR><TD>{@linkplain GpuLongMatrix}</TD><TD WIDTH=20> </TD><TD><TT>long long int**</TT>, <TT>unsigned long long int**</TT></TD></TR>
 * <TR><TD>{@linkplain GpuFloatMatrix}</TD><TD WIDTH=20> </TD><TD><TT>float**</TT></TD></TR>
 * <TR><TD>{@linkplain GpuDoubleMatrix}</TD><TD WIDTH=20> </TD><TD><TT>double**</TT></TD></TR>
 * <TR><TD>{@linkplain GpuStructVbl}</TD><TD WIDTH=20> </TD><TD><TT>struct T*</TT></TD></TR>
 * <TR><TD>{@linkplain GpuStructArray}</TD><TD WIDTH=20> </TD><TD><TT>struct T*</TT></TD></TR>
 * </TABLE>
 * </UL>
 * <P><LI>
 * Call the {@link Gpu#getModule(String) getModule()} method on a {@linkplain
 * Gpu Gpu} object, specifying the name of the compiled CUDA module file
 * relative to the Java class path. This returns a {@linkplain Module Module}
 * object.
 * <P><LI>
 * Suppose the kernel function interface is named Bar. Call the {@link
 * Module#getKernel(Class) getKernel()} method on the {@linkplain Module
 * Module} object, as follows:
 * <BR><TT>&nbsp;&nbsp;&nbsp;&nbsp;Bar proxy = module.getKernel (Bar.class);</TT>
 * <BR>This returns a reference to a proxy object that implements interface Bar.
 * <P><LI>
 * Configure the kernel by calling the {@link #setGridDim(int) setGridDim()},
 * {@link #setBlockDim(int) setBlockDim()}, {@link #setCacheConfig(CacheConfig)
 * setCacheConfig()}, and {@link #setDynamicSharedMemorySize(int)
 * setDynamicSharedMemorySize()} methods on the Bar proxy object as necessary.
 * <P><LI>
 * Set the contents of any input variables (instances of {@linkplain GpuVbl
 * GpuVbl} subclasses), and transfer these from host to device.
 * <P><LI>
 * Launch the kernel by calling the <TT>foo</TT> method on the Bar proxy object,
 * passing in the desired argument values. The <TT>foo</TT> method returns when
 * the kernel finishes.
 * <P><LI>
 * Transfer any output variables (instances of {@linkplain GpuVbl GpuVbl}
 * subclasses) from device to host, and examine their contents.
 * </OL>
 * <P>
 * Study the source code of the classes in package {@linkplain
 * edu.rit.gpu.example edu.rit.pj2.example} to see examples of Parallel Java 2
 * GPU programs.
 * <P>
 * <I>Note:</I> A class that implements interface Kernel is not multiple
 * thread safe. It is assumed that only one thread at a time will call methods
 * on a Kernel object.
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
public interface Kernel
	{

// Exported operations.

	/**
	 * Specify that this kernel function will be invoked with a one-dimensional
	 * grid. If not specified, the default is a one-dimensional grid with
	 * <TT>gridDimX</TT> = 1.
	 *
	 * @param  gridDimX  Grid X dimension (&ge; 1).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gridDimX</TT> &lt; 1.
	 *
	 * @see  #setGridDim(int,int)
	 * @see  #setGridDim(int,int,int)
	 */
	public void setGridDim
		(int gridDimX);

	/**
	 * Specify that this kernel function will be invoked with a two-dimensional
	 * grid. If not specified, the default is a one-dimensional grid with
	 * <TT>gridDimX</TT> = 1.
	 *
	 * @param  gridDimX  Grid X dimension (&ge; 1).
	 * @param  gridDimY  Grid Y dimension (&ge; 1).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gridDimX</TT> &lt; 1 or
	 *     <TT>gridDimY</TT> &lt; 1.
	 *
	 * @see  #setGridDim(int)
	 * @see  #setGridDim(int,int,int)
	 */
	public void setGridDim
		(int gridDimX,
		 int gridDimY);

	/**
	 * Specify that this kernel function will be invoked with a
	 * three-dimensional grid. If not specified, the default is a
	 * one-dimensional grid with <TT>gridDimX</TT> = 1.
	 *
	 * @param  gridDimX  Grid X dimension (&ge; 1).
	 * @param  gridDimY  Grid Y dimension (&ge; 1).
	 * @param  gridDimZ  Grid Z dimension (&ge; 1).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>gridDimX</TT> &lt; 1 or
	 *     <TT>gridDimY</TT> &lt; 1 or <TT>gridDimZ</TT> &lt; 1.
	 *
	 * @see  #setGridDim(int)
	 * @see  #setGridDim(int,int)
	 */
	public void setGridDim
		(int gridDimX,
		 int gridDimY,
		 int gridDimZ);

	/**
	 * Specify that this kernel function will be invoked with a one-dimensional
	 * block. If not specified, the default is a one-dimensional block with
	 * <TT>blockDimX</TT> = 1.
	 *
	 * @param  blockDimX  Block X dimension (&ge; 1).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>blockDimX</TT> &lt; 1.
	 *
	 * @see  #setBlockDim(int,int)
	 * @see  #setBlockDim(int,int,int)
	 */
	public void setBlockDim
		(int blockDimX);

	/**
	 * Specify that this kernel function will be invoked with a two-dimensional
	 * block. If not specified, the default is a one-dimensional block with
	 * <TT>blockDimX</TT> = 1.
	 *
	 * @param  blockDimX  Block X dimension (&ge; 1).
	 * @param  blockDimY  Block Y dimension (&ge; 1).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>blockDimX</TT> &lt; 1 or
	 *     <TT>blockDimY</TT> &lt; 1.
	 *
	 * @see  #setBlockDim(int)
	 * @see  #setBlockDim(int,int,int)
	 */
	public void setBlockDim
		(int blockDimX,
		 int blockDimY);

	/**
	 * Specify that this kernel function will be invoked with a
	 * three-dimensional block. If not specified, the default is a
	 * one-dimensional block with <TT>blockDimX</TT> = 1.
	 *
	 * @param  blockDimX  Block X dimension (&ge; 1).
	 * @param  blockDimY  Block Y dimension (&ge; 1).
	 * @param  blockDimZ  Block Z dimension (&ge; 1).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>blockDimX</TT> &lt; 1 or
	 *     <TT>blockDimY</TT> &lt; 1 or <TT>blockDimZ</TT> &lt; 1.
	 *
	 * @see  #setBlockDim(int)
	 * @see  #setBlockDim(int,int)
	 */
	public void setBlockDim
		(int blockDimX,
		 int blockDimY,
		 int blockDimZ);

	/**
	 * Specify that this kernel function will be invoked with the given cache
	 * configuration. If not specified, the default is to use the GPU's cache
	 * configuration setting.
	 *
	 * @param  config  Cache configuration.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>config</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  Gpu#setCacheConfig(CacheConfig)
	 */
	public void setCacheConfig
		(CacheConfig config);

	/**
	 * Specify that this kernel function will be invoked with the given dynamic
	 * shared memory size. If not specified, the default is 0.
	 *
	 * @param  bytesize  Number of dynamic shared memory bytes (&ge; 0).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>bytesize</TT> &lt; 0.
	 */
	public void setDynamicSharedMemorySize
		(int bytesize);

	}
