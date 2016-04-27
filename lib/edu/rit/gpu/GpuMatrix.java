//******************************************************************************
//
// File:    GpuMatrix.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.GpuMatrix
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
 * Class GpuMatrix is the abstract base class for a GPU matrix variable. This is
 * a two-dimensional array of data stored in the GPU's memory and mirrored in
 * the CPU's memory. Construct an instance of a subclass of class GpuMatrix by
 * calling a factory method on a {@linkplain Gpu Gpu} object.
 * <P>
 * Class GpuMatrix supports mirroring all of the GPU's data matrix in the CPU's
 * memory, mirroring only a portion of the GPU's data matrix in the CPU's
 * memory, or mirroring none of the GPU's data matrix. Class GpuMatrix provides
 * operations for copying all or portions of the data matrix from the CPU to the
 * GPU or from the GPU to the CPU.
 * <P>
 * In the GPU memory, the matrix is implemented as a pointer to an array of
 * <I>R</I> row pointers, where <I>R</I> is the number of rows in the matrix.
 * Each row pointer points to an array of <I>C</I> data elements, where <I>C</I>
 * is the number of columns in the matrix.
 *
 * @author  Alan Kaminsky
 * @version 05-Apr-2014
 */
public abstract class GpuMatrix
	extends GpuVbl
	{

// Hidden data members.

	int rows;          // Number of rows in GPU memory
	int cols;          // Number of columns in GPU memory
	int cpurows;       // Number of rows mirrored in CPU memory
	int cpucols;       // Number of columns mirrored in CPU memory
	long rowbytesize;  // Number of bytes in one row in GPU memory
	long elem00ptr;    // Pointer to element [0][0] in GPU memory

// Hidden constructors.

	/**
	 * Construct a new dynamically allocated GPU matrix.
	 *
	 * @param  gpu       Gpu object.
	 * @param  rows      Number of rows in GPU memory.
	 * @param  cols      Number of columns in GPU memory.
	 * @param  cpurows   Number of rows mirrored in CPU memory.
	 * @param  cpucols   Number of columns mirrored in CPU memory.
	 * @param  elemsize  Size of a matrix element in bytes.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	GpuMatrix
		(Gpu gpu,
		 int rows,
		 int cols,
		 int cpurows,
		 int cpucols,
		 long elemsize)
		{
		super (gpu, sizeof (rows, cols, elemsize));
		this.rows = rows;
		this.cols = cols;
		this.cpurows = cpurows;
		this.cpucols = cpucols;
		this.rowbytesize = elemsize*cols;
//System.out.printf ("GpuMatrix()%n");
//System.out.printf ("\tbytesize    = %d%n", this.bytesize);
//System.out.printf ("\tdynamic     = %b%n", this.dynamic);
//System.out.printf ("\tdptr        = %d%n", this.dptr);
//System.out.printf ("\trows        = %d%n", this.rows);
//System.out.printf ("\tcols        = %d%n", this.cols);
//System.out.printf ("\tcpurows     = %d%n", this.cpurows);
//System.out.printf ("\tcpucols     = %d%n", this.cpucols);
//System.out.printf ("\trowbytesize = %d%n", this.rowbytesize);

		// Initialize array of row pointers in GPU memory.
		if (Cuda.is64BitPointer())
			{
			elem00ptr = dptr + 8L*rows;
//System.out.printf ("\telem00ptr   = %d%n", this.elem00ptr);
			long[] rowptr = new long [rows];
			for (int i = 0; i < rows; ++ i)
				{
				rowptr[i] = elem00ptr + i*rowbytesize;
//System.out.printf ("\trowptr[%d] = %d%n", i, rowptr[i]);
				}
			Cuda.cuMemcpyHtoD (gpu.ctx, dptr, 0, rowptr, 0, rows);
			}
		else
			{
			elem00ptr = dptr + 4L*rows;
//System.out.printf ("\telem00ptr   = %d%n", this.elem00ptr);
				// Align to an 8-byte boundary.
			if ((elem00ptr & 7L) != 0L) elem00ptr += 4L;
			int[] rowptr = new int [rows];
			for (int i = 0; i < rows; ++ i)
				{
				rowptr[i] = (int)(elem00ptr + i*rowbytesize);
//System.out.printf ("\trowptr[%d] = %d%n", i, rowptr[i]);
				}
			Cuda.cuMemcpyHtoD (gpu.ctx, dptr, 0, rowptr, 0, rows);
			}
		}

	/**
	 * Determine the number of bytes needed for a GPU matrix.
	 *
	 * @param  rows      Number of rows in GPU memory.
	 * @param  cols      Number of columns in GPU memory.
	 * @param  elemsize  Size of a matrix element in bytes.
	 *
	 * @return  Number of bytes.
	 */
	private static long sizeof
		(int rows,
		 int cols,
		 long elemsize)
		{
		// Storage for row pointers.
		long n = (Cuda.is64BitPointer() ? 8L : 4L)*rows;

		// Align to an 8-byte boundary.
		if ((n & 7L) != 0L) n += 4L;

		// Storage for matrix elements.
		n += elemsize*rows*cols;

		return n;
		}

// Exported operations.

	/**
	 * Returns the number of rows in the GPU matrix in the GPU device's memory.
	 *
	 * @return  Number of rows in the GPU matrix.
	 */
	public int rows()
		{
		return rows;
		}

	/**
	 * Returns the number of columns in the GPU matrix in the GPU device's
	 * memory.
	 *
	 * @return  Number of columns in the GPU matrix.
	 */
	public int cols()
		{
		return cols;
		}

	/**
	 * Returns the number of rows in the GPU matrix mirrored in the host CPU's
	 * memory.
	 *
	 * @return  Number of rows in the CPU matrix.
	 */
	public int cpuRows()
		{
		return cpurows;
		}

	/**
	 * Returns the number of columns in the GPU matrix mirrored in the host
	 * CPU's memory.
	 *
	 * @return  Number of columns in the CPU matrix.
	 */
	public int cpuCols()
		{
		return cpucols;
		}

	/**
	 * Copy this GPU matrix from the host CPU's memory to the GPU device's
	 * memory. This is equivalent to the call
	 * <TT>hostToDev(0,0,0,0,cpuRows(),cpuCols())</TT>.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev()
		{
		hostToDev (0, 0, 0, 0, cpurows, cpucols);
		}

	/**
	 * Copy the initial portion of this GPU matrix from the host CPU's memory to
	 * the GPU device's memory. This is equivalent to the call
	 * <TT>hostToDev(0,0,0,0,rowlen,collen)</TT>.
	 *
	 * @param  rowlen  Number of rows to copy.
	 * @param  collen  Number of columns to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>rowlen</TT> &lt; 0,
	 *     <TT>rowlen</TT> &gt; <TT>cpuRows()</TT>, <TT>collen</TT> &lt; 0, or
	 *     <TT>collen</TT> &gt; <TT>cpuCols()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev
		(int rowlen,
		 int collen)
		{
		hostToDev (0, 0, 0, 0, rowlen, collen);
		}

	/**
	 * Copy the given portion of this GPU matrix from the host CPU's memory to
	 * the GPU device's memory. <TT>rowlen</TT>&times;<TT>collen</TT> elements
	 * starting at indexes <TT>[srcrow][srccol]</TT> in the CPU matrix are
	 * copied to the GPU matrix starting at indexes <TT>[dstrow][dstcol]</TT>.
	 *
	 * @param  dstrow  GPU matrix starting row index.
	 * @param  dstcol  GPU matrix starting column index.
	 * @param  srcrow  CPU matrix starting row index.
	 * @param  srccol  CPU matrix starting column index.
	 * @param  rowlen  Number of rows to copy.
	 * @param  collen  Number of columns to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>dstrow</TT> &lt; 0,
	 *     <TT>dstcol</TT> &lt; 0, <TT>srcrow</TT> &lt; 0, <TT>srccol</TT> &lt;
	 *     0, <TT>rowlen</TT> &lt; 0, <TT>collen</TT> &lt; 0,
	 *     <TT>dstrow+rowlen</TT> &gt; <TT>rows()</TT>, <TT>dstcol+collen</TT>
	 *     &gt; <TT>cols()</TT>, <TT>srcrow+rowlen</TT> &gt; <TT>cpuRows()</TT>,
	 *     or <TT>srccol+collen</TT> &gt; <TT>cpuCols()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public abstract void hostToDev
		(int dstrow,
		 int dstcol,
		 int srcrow,
		 int srccol,
		 int rowlen,
		 int collen);

	/**
	 * Copy this GPU matrix from the GPU device's memory to the host CPU's
	 * memory. This is equivalent to the call
	 * <TT>devToHost(0,0,0,0,cpuRows(),cpuCols())</TT>.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost()
		{
		devToHost (0, 0, 0, 0, cpurows, cpucols);
		}

	/**
	 * Copy the initial portion of this GPU matrix from the GPU device's memory
	 * to the host CPU's memory. This is equivalent to the call
	 * <TT>devToHost(0,0,0,0,rowlen,collen)</TT>.
	 *
	 * @param  rowlen  Number of rows to copy.
	 * @param  collen  Number of columns to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>rowlen</TT> &lt; 0,
	 *     <TT>rowlen</TT> &gt; <TT>cpuRows()</TT>, <TT>collen</TT> &lt; 0, or
	 *     <TT>collen</TT> &gt; <TT>cpuCols()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost
		(int rowlen,
		 int collen)
		{
		devToHost (0, 0, 0, 0, rowlen, collen);
		}

	/**
	 * Copy the given portion of this GPU matrix from the GPU device's memory to
	 * the host CPU's memory. <TT>rowlen</TT>&times;<TT>collen</TT> elements
	 * starting at indexes <TT>[srcrow][srccol]</TT> in the GPU matrix are
	 * copied to the CPU matrix starting at indexes <TT>[dstrow][dstcol]</TT>.
	 *
	 * @param  dstrow  CPU matrix starting row index.
	 * @param  dstcol  CPU matrix starting column index.
	 * @param  srcrow  GPU matrix starting row index.
	 * @param  srccol  GPU matrix starting column index.
	 * @param  rowlen  Number of rows to copy.
	 * @param  collen  Number of columns to copy.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>dstrow</TT> &lt; 0,
	 *     <TT>dstcol</TT> &lt; 0, <TT>srcrow</TT> &lt; 0, <TT>srccol</TT> &lt;
	 *     0, <TT>rowlen</TT> &lt; 0, <TT>collen</TT> &lt; 0,
	 *     <TT>dstrow+rowlen</TT> &gt; <TT>cpuRows()</TT>,
	 *     <TT>dstcol+collen</TT> &gt; <TT>cpuCols()</TT>,
	 *     <TT>srcrow+rowlen</TT> &gt; <TT>rows()</TT>, or
	 *     <TT>srccol+collen</TT> &gt; <TT>cols()</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public abstract void devToHost
		(int dstrow,
		 int dstcol,
		 int srcrow,
		 int srccol,
		 int rowlen,
		 int collen);

	}
