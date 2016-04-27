//******************************************************************************
//
// File:    GpuArray.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.GpuArray
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
 * Class GpuArray is the abstract base class for a GPU array variable. This is
 * an array of data stored in the GPU's memory and mirrored in the CPU's memory.
 * Construct an instance of a subclass of class GpuArray by calling a factory
 * method on a {@linkplain Gpu Gpu} object or a {@linkplain Module Module}
 * object.
 * <P>
 * Class GpuArray supports mirroring all of the GPU's data array in the CPU's
 * memory, mirroring only a portion of the GPU's data array in the CPU's memory,
 * or mirroring none of the GPU's data array. Class GpuArray provides operations
 * for copying all or portions of the data array from the CPU to the GPU or from
 * the GPU to the CPU.
 * <P>
 * In the GPU memory, the array is implemented as a pointer to an array of
 * <I>N</I> data elements, where <I>N</I> is the length of the array.
 *
 * @author  Alan Kaminsky
 * @version 03-Apr-2014
 */
public abstract class GpuArray
	extends GpuVbl
	{

// Hidden data members.

	int len;     // Number of elements in GPU memory
	int cpulen;  // Number of elements mirrored in CPU memory

// Hidden constructors.

	/**
	 * Construct a new dynamically allocated GPU array.
	 *
	 * @param  gpu          Gpu object.
	 * @param  len          Number of elements in GPU memory.
	 * @param  bytesize     Number of bytes allocated in GPU memory.
	 * @param  cpulen       Number of elements mirrored in CPU memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuArray
		(Gpu gpu,
		 int len,
		 long bytesize,
		 int cpulen)
		{
		super (gpu, bytesize);
		this.len = len;
		this.cpulen = cpulen;
		}

	/**
	 * Construct a new statically allocated GPU array.
	 *
	 * @param  gpu          Gpu object.
	 * @param  len          Number of elements in GPU memory.
	 * @param  bytesize     Number of bytes allocated in GPU memory.
	 * @param  cpulen       Number of elements mirrored in CPU memory.
	 * @param  dptr         Pointer to array in GPU memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuArray
		(Gpu gpu,
		 int len,
		 long bytesize,
		 int cpulen,
		 long dptr)
		{
		super (gpu, bytesize, dptr);
		this.len = len;
		this.cpulen = cpulen;
		}

// Exported operations.

	/**
	 * Returns the length of the GPU array in the GPU device's memory.
	 *
	 * @return  Number of elements in the GPU array.
	 */
	public int length()
		{
		return len;
		}

	/**
	 * Returns the length of the GPU array mirrored in the host CPU's memory.
	 *
	 * @return  Number of elements in the CPU array.
	 */
	public int cpuLength()
		{
		return cpulen;
		}

	/**
	 * Copy this GPU array from the host CPU's memory to the GPU device's
	 * memory. This is equivalent to the call
	 * <TT>hostToDev(0,0,cpuLength())</TT>.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev()
		{
		hostToDev (0, 0, cpulen);
		}

	/**
	 * Copy the initial portion of this GPU array from the host CPU's memory to
	 * the GPU device's memory. This is equivalent to the call
	 * <TT>hostToDev(0,0,len)</TT>.
	 *
	 * @param  len  Number of elements to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 0 or <TT>len</TT>
	 *     &gt; <TT>cpuLength()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev
		(int len)
		{
		hostToDev (0, 0, len);
		}

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
	public abstract void hostToDev
		(int dstindex,
		 int srcindex,
		 int len);

	/**
	 * Copy this GPU array from the GPU device's memory to the host CPU's
	 * memory. This is equivalent to the call
	 * <TT>devToHost(0,0,cpuLength())</TT>.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost()
		{
		devToHost (0, 0, cpulen);
		}

	/**
	 * Copy the initial portion of this GPU array from the GPU device's memory
	 * to the host CPU's memory. This is equivalent to the call
	 * <TT>devToHost(0,0,len)</TT>.
	 *
	 * @param  len  Number of elements to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 0 or <TT>len</TT>
	 *     &gt; <TT>cpuLength()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost
		(int len)
		{
		devToHost (0, 0, len);
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
	public abstract void devToHost
		(int dstindex,
		 int srcindex,
		 int len);

	}
