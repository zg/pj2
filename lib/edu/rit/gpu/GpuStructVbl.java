//******************************************************************************
//
// File:    GpuStructVbl.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.GpuStructVbl
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

import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Class GpuStructVbl provides a struct GPU variable. This is a variable of a C
 * struct type stored in the GPU's memory and mirrored in the CPU's memory.
 * Class GpuStructVbl provides operations for copying the variable from the CPU
 * to the GPU or from the GPU to the CPU.
 * <P>
 * To use a struct GPU variable:
 * <OL TYPE=1>
 * <P><LI>
 * Write a Java class that extends class {@linkplain Struct Struct}, to mirror
 * the fields of the C struct data type.
 * <P><LI>
 * Construct an instance of class GpuStructVbl by calling the {@link
 * Gpu#getStructVbl(Class) getStructVbl()} method on a {@linkplain Gpu Gpu}
 * object or the {@link Module#getStructVbl(String,Class) getStructVbl()} method
 * on a {@linkplain Module Module} object.
 * <P><LI>
 * Construct a new instance of the Java class with the desired state and assign
 * it to the {@link #item item} field. Call the {@link #hostToDev() hostToDev()}
 * method to copy the variable from CPU memory to GPU memory.
 * <P><LI>
 * Pass the GpuStructVbl object as an argument of a GPU kernel function call. In
 * the GPU code, this becomes a pointer (type <TT>struct T*</TT>) to the
 * variable. The GPU code sets the variable to the desired state.
 * <P><LI>
 * Call the {@link #devToHost() devToHost()} method to copy the variable from
 * GPU memory to CPU memory. Examine the {@link #item item} field's state.
 * </OL>
 *
 * @param  <T>  Java data type. Class <TT>T</TT> must extend class {@linkplain
 *              Struct Struct}.
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
public class GpuStructVbl<T extends Struct>
	extends GpuVbl
	{

// Exported data members.

	/**
	 * The mirrored struct variable in CPU memory.
	 */
	public T item;

// Hidden data members.

	private ByteBuffer buf;

// Hidden constructors.

	/**
	 * Construct a new dynamically allocated GPU struct variable.
	 *
	 * @param  gpu   Gpu object.
	 * @param  type  Java data type.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuStructVbl
		(Gpu gpu,
		 Class<T> type)
		{
		super (gpu, getByteSize (type));
		}

	/**
	 * Construct a new statically allocated GPU struct variable.
	 *
	 * @param  gpu   Gpu object.
	 * @param  dptr  Pointer to variable in GPU memory.
	 * @param  type  Java data type.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuStructVbl
		(Gpu gpu,
		 long dptr,
		 Class<T> type)
		{
		super (gpu, getByteSize (type), dptr);
		}

	/**
	 * Determine the byte size of the C struct for the given Java data type.
	 *
	 * @param  type  Java data type.
	 *
	 * @return  Byte size.
	 */
	private static long getByteSize
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
				("GpuStructVbl(): Cannot get byte size", exc);
			}
		if (bytesize > Integer.MAX_VALUE)
			throw new IllegalArgumentException (String.format
				("GpuStructVbl(): Byte size = %d too large", bytesize));
		return bytesize;
		}

// Exported operations.

	/**
	 * Copy this GPU variable from the host CPU's memory to the GPU device's
	 * memory.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if the {@link #item item} field is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev()
		{
		if (buf == null)
			buf = ByteBuffer.allocate ((int)bytesize);
		buf.clear();
		buf.order (ByteOrder.LITTLE_ENDIAN);
		item.toStruct (buf);
		Cuda.cuMemcpyHtoD (gpu.ctx, dptr, 0, buf.array(), 0, (int)bytesize);
		}

	/**
	 * Copy this GPU variable from the GPU device's memory to the host CPU's
	 * memory.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if the {@link #item item} field is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost()
		{
		if (buf == null)
			buf = ByteBuffer.allocate ((int)bytesize);
		Cuda.cuMemcpyDtoH (gpu.ctx, buf.array(), 0, dptr, 0, (int)bytesize);
		buf.clear();
		buf.order (ByteOrder.LITTLE_ENDIAN);
		item.fromStruct (buf);
		}

	}
