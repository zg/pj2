//******************************************************************************
//
// File:    GpuStructArray.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.GpuStructArray
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

import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Class GpuStructArray provides a struct array GPU variable. This is an array
 * of a C struct type stored in the GPU's memory and mirrored in the CPU's
 * memory.
 * <P>
 * Class GpuStructArray supports mirroring all of the GPU's data array in the
 * CPU's memory, mirroring only a portion of the GPU's data array in the CPU's
 * memory, or mirroring none of the GPU's data array. Class GpuStructArray
 * provides operations for copying all or portions of the data array from the
 * CPU to the GPU or from the GPU to the CPU.
 * <P>
 * To use a struct array GPU variable:
 * <OL TYPE=1>
 * <P><LI>
 * Write a Java class that extends class {@linkplain Struct Struct}, to mirror
 * the fields of the C struct data type.
 * <P><LI>
 * Construct an instance of class GpuStructArray by calling either the {@link
 * Gpu#getStructArray(Class,int) getStructArray()} method on a {@linkplain Gpu
 * Gpu} object or the {@link Module#getStructArray(String,Class,int)
 * getStructArray()} method on a {@linkplain Module Module} object, specifying
 * the length of the array and (optionally) the length of the array portion
 * mirrored in the CPU.
 * <P><LI>
 * Construct new instances of the Java class with the desired states and assign
 * them to the {@link #item item} field's elements. Call the {@link #hostToDev()
 * hostToDev()} method to copy the mirrored portion of the array from CPU memory
 * to GPU memory.
 * <P><LI>
 * Pass the GpuStructArray object as an argument of a GPU kernel function call.
 * In the GPU code, this becomes a pointer (type <TT>struct T*</TT>) to an array
 * of <I>N</I> data elements, where <I>N</I> is the length of the array. The GPU
 * code sets the array's elements to the desired states.
 * <P><LI>
 * Call the {@link #devToHost() devToHost()} method to copy the mirrored portion
 * of the array from GPU memory to CPU memory. Examine the states of the {@link
 * #item item} field's elements.
 * </OL>
 *
 * @param  <T>  Java data type. Class <TT>T</TT> must extend class {@linkplain
 *              Struct Struct}.
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
public class GpuStructArray<T extends Struct>
	extends GpuArray
	{

// Exported data members.

	/**
	 * The mirrored portion of the struct array in CPU memory.
	 */
	public final T[] item;

// Hidden data members.

	private int structByteSize;
	private ByteBuffer buf;

// Hidden constructors.

	/**
	 * Construct a new dynamically allocated GPU struct array.
	 *
	 * @param  gpu     Gpu object.
	 * @param  len     Number of elements in GPU memory.
	 * @param  cpulen  Number of elements mirrored in CPU memory.
	 * @param  type    Java data type.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuStructArray
		(Gpu gpu,
		 int len,
		 int cpulen,
		 Class<T> type)
		{
		super (gpu, len, getArrayByteSize (len, type), cpulen);
		item = (T[]) Array.newInstance (type, cpulen);
		structByteSize = (int) getStructByteSize (type);
		}

	/**
	 * Construct a new statically allocated GPU struct array.
	 *
	 * @param  gpu     Gpu object.
	 * @param  len     Number of elements in GPU memory.
	 * @param  cpulen  Number of elements mirrored in CPU memory.
	 * @param  dptr    Pointer to array in GPU memory.
	 * @param  type    Java data type.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuStructArray
		(Gpu gpu,
		 int len,
		 int cpulen,
		 long dptr,
		 Class<T> type)
		{
		super (gpu, len, getArrayByteSize (len, type), cpulen, dptr);
		item = (T[]) Array.newInstance (type, cpulen);
		structByteSize = (int) getStructByteSize (type);
		}

	/**
	 * Determine the byte size of the C struct for the given Java data type.
	 *
	 * @param  type  Java data type.
	 *
	 * @return  Byte size.
	 */
	private static long getStructByteSize
		(Class<?> type)
		{
		long bytesize = 0L;
		try
			{
			Method method = type.getMethod ("sizeof");
			method.setAccessible (true);
			bytesize = (Long)(method.invoke (null));
			}
		catch (Throwable exc)
			{
			throw new IllegalArgumentException
				("GpuStructArray(): Cannot get byte size", exc);
			}
		if (bytesize > Integer.MAX_VALUE)
			throw new IllegalArgumentException (String.format
				("GpuStructArray(): Struct byte size = %d too large",
				 bytesize));
		return bytesize;
		}

	/**
	 * Determine the byte size of the C struct array for the given Java data
	 * type mirrored in the CPU.
	 *
	 * @param  len     Number of elements in GPU memory.
	 * @param  type    Java data type.
	 *
	 * @return  Byte size.
	 */
	private static long getArrayByteSize
		(int len,
		 Class<?> type)
		{
		long bytesize = getStructByteSize (type);
		if (bytesize*len > Integer.MAX_VALUE)
			throw new IllegalArgumentException (String.format
				("GpuStructArray(): Array byte size = %d too large",
				 bytesize*len));
		return bytesize*len;
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
			{
			if (buf == null)
				buf = ByteBuffer.allocate (structByteSize*cpulen);
			for (int i = 0; i < len; ++ i)
				{
				buf.clear();
				buf.order (ByteOrder.LITTLE_ENDIAN);
				buf.position (i*structByteSize);
				item[i+srcindex].toStruct (buf);
				}
			Cuda.cuMemcpyHtoD
				(gpu.ctx, dptr, dstindex*structByteSize,
				 buf.array(), 0, len*structByteSize);
			}
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
			{
			if (buf == null)
				buf = ByteBuffer.allocate (structByteSize*cpulen);
			Cuda.cuMemcpyDtoH
				(gpu.ctx, buf.array(), 0,
				 dptr, srcindex*structByteSize, len*structByteSize);
			for (int i = 0; i < len; ++ i)
				{
				buf.clear();
				buf.order (ByteOrder.LITTLE_ENDIAN);
				buf.position (i*structByteSize);
				item[i+dstindex].fromStruct (buf);
				}
			}
		}

	}
