//******************************************************************************
//
// File:    GpuDoubleArray.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.GpuDoubleArray
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
 * Class GpuDoubleArray provides a double array GPU variable. This is an array
 * of type <TT>double</TT> stored in the GPU's memory and mirrored in the CPU's
 * memory.
 * <P>
 * Class GpuDoubleArray supports mirroring all of the GPU's data array in the
 * CPU's memory, mirroring only a portion of the GPU's data array in the CPU's
 * memory, or mirroring none of the GPU's data array. Class GpuDoubleArray
 * provides operations for copying all or portions of the data array from the
 * CPU to the GPU or from the GPU to the CPU.
 * <P>
 * To use a double array GPU variable:
 * <OL TYPE=1>
 * <P><LI>
 * Construct an instance of class GpuDoubleArray by calling either the {@link
 * Gpu#getDoubleArray(int) getDoubleArray()} method on a {@linkplain Gpu Gpu}
 * object or the {@link Module#getDoubleArray(String,int) getDoubleArray()}
 * method on a {@linkplain Module Module} object, specifying the length of the
 * array and (optionally) the length of the array portion mirrored in the CPU.
 * <P><LI>
 * Set the {@link #item item} field's elements to the desired values. Call the
 * {@link #hostToDev() hostToDev()} method to copy the mirrored portion of the
 * array from CPU memory to GPU memory.
 * <P><LI>
 * Pass the GpuDoubleArray object as an argument of a GPU kernel function call.
 * In the GPU code, this becomes a pointer (type <TT>double*</TT>) to an array
 * of <I>N</I> data elements, where <I>N</I> is the length of the array. The GPU
 * code sets the array's elements to the desired values.
 * <P><LI>
 * Call the {@link #devToHost() devToHost()} method to copy the mirrored portion
 * of the array from GPU memory to CPU memory. Examine the values of the {@link
 * #item item} field's elements.
 * </OL>
 *
 * @author  Alan Kaminsky
 * @version 03-Apr-2014
 */
public class GpuDoubleArray
	extends GpuArray
	{

// Exported data members.

	/**
	 * The mirrored portion of the double array in CPU memory.
	 */
	public final double[] item;

// Hidden constructors.

	/**
	 * Construct a new dynamically allocated GPU double array.
	 *
	 * @param  gpu     Gpu object.
	 * @param  len     Number of elements in GPU memory.
	 * @param  cpulen  Number of elements mirrored in CPU memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuDoubleArray
		(Gpu gpu,
		 int len,
		 int cpulen)
		{
		super (gpu, len, 8L*len, cpulen);
		item = new double [cpulen];
		}

	/**
	 * Construct a new statically allocated GPU double array.
	 *
	 * @param  gpu     Gpu object.
	 * @param  len     Number of elements in GPU memory.
	 * @param  cpulen  Number of elements mirrored in CPU memory.
	 * @param  dptr    Pointer to array in GPU memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuDoubleArray
		(Gpu gpu,
		 int len,
		 int cpulen,
		 long dptr)
		{
		super (gpu, len, 8L*len, cpulen, dptr);
		item = new double [cpulen];
		}

// Exported operations.

	/**
	 * Copy the given portion of this GPU array from the host CPU's memory to
	 * the GPU device's memory. <TT>len</TT> elements starting at index
	 * <TT>srcindex</TT> in the CPU array are copied to the GPU array starting
	 * at index <TT>dstindex</TT>.
	 *
	 * @param  dstindex  GPU array starting index.
	 * @param  srcindex  CPU array starting index.
	 * @param  len       Number of elements to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>dstindex</TT> &lt; 0,
	 *     <TT>srcindex</TT> &lt; 0, <TT>len</TT> &lt; 0, <TT>dstindex+len</TT>
	 *     &gt; <TT>length()</TT>, or <TT>srcindex+len</TT> &gt;
	 *     <TT>cpuLength()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev
		(int dstindex,
		 int srcindex,
		 int len)
		{
		if (dstindex < 0 || srcindex < 0 || len < 0 ||
			dstindex + len > this.len || srcindex + len > cpulen)
				throw new IndexOutOfBoundsException();
		if (len > 0)
			Cuda.cuMemcpyHtoD (gpu.ctx, dptr, dstindex, item, srcindex, len);
		}

	/**
	 * Copy the given portion of this GPU array from the GPU device's memory to
	 * the host CPU's memory. <TT>len</TT> elements copied starting at index
	 * <TT>srcindex</TT> in the GPU array are copied to the CPU array starting
	 * at index <TT>dstindex</TT>.
	 *
	 * @param  dstindex  CPU array starting index.
	 * @param  srcindex  GPU array starting index.
	 * @param  len       Number of elements to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>dstindex</TT> &lt; 0,
	 *     <TT>srcindex</TT> &lt; 0, <TT>len</TT> &lt; 0, <TT>dstindex+len</TT>
	 *     &gt; <TT>cpuLength()</TT>, or <TT>srcindex+len</TT> &gt;
	 *     <TT>length()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost
		(int dstindex,
		 int srcindex,
		 int len)
		{
		if (dstindex < 0 || srcindex < 0 || len < 0 ||
			dstindex + len > cpulen || srcindex + len > this.len)
				throw new IndexOutOfBoundsException();
		if (len > 0)
			Cuda.cuMemcpyDtoH (gpu.ctx, item, dstindex, dptr, srcindex, len);
		}

	}
