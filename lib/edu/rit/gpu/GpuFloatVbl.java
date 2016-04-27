//******************************************************************************
//
// File:    GpuFloatVbl.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.GpuFloatVbl
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
 * Class GpuFloatVbl provides a float GPU variable. This is a variable of type
 * <TT>float</TT> stored in the GPU's memory and mirrored in the CPU's memory.
 * Class GpuFloatVbl provides operations for copying the variable from the CPU
 * to the GPU or from the GPU to the CPU.
 * <P>
 * To use a float GPU variable:
 * <OL TYPE=1>
 * <P><LI>
 * Construct an instance of class GpuFloatVbl by calling the {@link
 * Module#getFloatVbl(String) getFloatVbl()} method on a {@linkplain Module
 * Module} object.
 * <P><LI>
 * Set the {@link #item item} field to the desired value. Call the {@link
 * #hostToDev() hostToDev()} method to copy the variable from CPU memory to GPU
 * memory.
 * <P><LI>
 * Call the {@link #devToHost() devToHost()} method to copy the variable from
 * GPU memory to CPU memory. Examine the value of the {@link #item item} field.
 * </OL>
 *
 * @author  Alan Kaminsky
 * @version 02-Apr-2014
 */
public class GpuFloatVbl
	extends GpuVbl
	{

// Exported data members.

	/**
	 * The mirrored float variable in CPU memory.
	 */
	public float item;

// Hidden data members.

	private float[] buf = new float [1];

// Hidden constructors.

	/**
	 * Construct a new statically allocated GPU float variable.
	 *
	 * @param  gpu   Gpu object.
	 * @param  dptr  Pointer to variable in GPU memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuFloatVbl
		(Gpu gpu,
		 long dptr)
		{
		super (gpu, 4L, dptr);
		}

// Exported operations.

	/**
	 * Copy this GPU variable from the host CPU's memory to the GPU device's
	 * memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev()
		{
		buf[0] = item;
		Cuda.cuMemcpyHtoD (gpu.ctx, dptr, 0, buf, 0, 1);
		}

	/**
	 * Copy this GPU variable from the GPU device's memory to the host CPU's
	 * memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost()
		{
		Cuda.cuMemcpyDtoH (gpu.ctx, buf, 0, dptr, 0, 1);
		item = buf[0];
		}

	}
