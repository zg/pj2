//******************************************************************************
//
// File:    GpuVbl.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.GpuVbl
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
 * Class GpuVbl is the abstract base class for a GPU variable. This is a block
 * of data stored in the GPU's memory and mirrored in the CPU's memory.
 * Construct an instance of a subclass of class GpuVbl by calling a factory
 * method on a {@linkplain Gpu Gpu} object or a {@linkplain Module Module}
 * object.
 *
 * @author  Alan Kaminsky
 * @version 02-Apr-2014
 */
public abstract class GpuVbl
	{

// Hidden data members.

	Gpu gpu;          // Gpu object that created this
	long bytesize;    // Number of bytes allocated in GPU memory
	boolean dynamic;  // True if dynamically allocated
	long dptr;        // Pointer to storage in GPU memory

// Hidden constructors.

	/**
	 * Construct a new dynamically allocated GPU variable.
	 *
	 * @param  gpu          Gpu object.
	 * @param  bytesize     Number of bytes allocated in GPU memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuVbl
		(Gpu gpu,
		 long bytesize)
		{
		this (gpu, bytesize, Cuda.cuMemAlloc (gpu.ctx, bytesize));
		this.dynamic = true;
		}

	/**
	 * Construct a new statically allocated GPU variable.
	 *
	 * @param  gpu          Gpu object.
	 * @param  bytesize     Number of bytes allocated in GPU memory.
	 * @param  dptr         Pointer to storage in GPU memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuVbl
		(Gpu gpu,
		 long bytesize,
		 long dptr)
		{
		this.gpu = gpu;
		this.bytesize = bytesize;
		this.dynamic = false;
		this.dptr = dptr;
		}

// Exported operations.

	/**
	 * Copy this GPU variable from the host CPU's memory to the GPU device's
	 * memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public abstract void hostToDev();

	/**
	 * Copy this GPU variable from the GPU device's memory to the host CPU's
	 * memory.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public abstract void devToHost();

	/**
	 * Free this GPU variable's storage in the GPU device's memory. If this GPU
	 * variable was not dynamically allocated, the <TT>free()</TT> method does
	 * nothing.
	 */
	public void free()
		{
		if (dynamic)
			{
			Cuda.cuMemFree (gpu.ctx, dptr);
			dptr = 0;
			}
		}

// Hidden operations.

	/**
	 * Finalize this GPU variable.
	 */
	protected synchronized void finalize()
		{
		free();
		}

	}
