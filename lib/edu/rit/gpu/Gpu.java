//******************************************************************************
//
// File:    Gpu.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.Gpu
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

import java.io.IOException;

/**
 * Class Gpu encapsulates a graphics processing unit (GPU) compute device. A GPU
 * compute device, or just "device", is denoted by an integer device number. To
 * use a device:
 * <OL TYPE=1>
 * <P><LI>
 * Call the {@link #gpu() gpu()} method to get a Gpu object.
 * <P><LI>
 * Call methods on the Gpu object to perform GPU operations; such as:
 * <P><LI>
 * Informational operations &mdash;
 * <UL>
 * <LI>{@link #ensureComputeCapability(int,int) ensureComputeCapability()}
 * <LI>{@link #getMultiprocessorCount() getMultiprocessorCount()}
 * </UL>
 * <P><LI>
 * GPU configuration &mdash;
 * <UL>
 * <LI>{@link #setCacheConfig(CacheConfig) setCacheConfig()},
 *     {@link #getCacheConfig() getCacheConfig()}
 * </UL>
 * <P><LI>
 * Create a dynamically allocated primitive array or matrix in GPU memory
 * &mdash;
 * <UL>
 * <LI>{@link #getByteArray(int) getByteArray()},
 *     {@link #getByteMatrix(int,int) getByteMatrix()}
 * <LI>{@link #getShortArray(int) getShortArray()},
 *     {@link #getShortMatrix(int,int) getShortMatrix()}
 * <LI>{@link #getIntArray(int) getIntArray()},
 *     {@link #getIntMatrix(int,int) getIntMatrix()}
 * <LI>{@link #getLongArray(int) getLongArray()},
 *     {@link #getLongMatrix(int,int) getLongMatrix()}
 * <LI>{@link #getFloatArray(int) getFloatArray()},
 *     {@link #getFloatMatrix(int,int) getFloatMatrix()}
 * <LI>{@link #getDoubleArray(int) getDoubleArray()},
 *     {@link #getDoubleMatrix(int,int) getDoubleMatrix()}
 * </UL>
 * <P><LI>
 * Create a dynamically allocated struct or struct array in GPU memory &mdash;
 * <UL>
 * <LI>{@link #getStructVbl(Class) getStructVbl()}
 * <LI>{@link #getStructArray(Class,int) getStructArray()}
 * </UL>
 * <P><LI>
 * Get a module containing GPU kernel functions and data &mdash;
 * <UL>
 * <LI>{@link #getModule(String) getModule()}
 * </UL>
 * </OL>
 * <P>
 * Study the source code of the classes in package {@linkplain
 * edu.rit.gpu.example edu.rit.pj2.example} to see examples of Parallel Java 2
 * GPU programs.
 * <P>
 * <I>Note:</I> Class Gpu is not multiple thread safe. It is assumed that only
 * one thread at a time will call methods on a Gpu object.
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
public class Gpu
	{

// Hidden class data members.

	// Array of allowed device numbers for this process.
	private static int[] allowedDeviceNumbers;

	// Number of devices in use;
	private static int inUseDevices;

	// Initialize the CUDA driver API.
	static
		{
		if (Cuda.libraryLoaded)
			{
			Cuda.cuInit (0);
			setDeviceNumbers (null);
			inUseDevices = 0;
			}
		}

// Hidden data members.

	int devnum;   // Device number
	int device;   // Device handle
	String name;  // Device name
	int major;    // Compute capability major version
	int minor;    // Compute capability minor version
	int mpcount;  // Number of multiprocessors
	long ctx;     // CUDA context

// Hidden constructors.

	/**
	 * Construct a new Gpu object.
	 *
	 * @param  devnum  Device number.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	private Gpu
		(int devnum)
		{
		this.devnum = devnum;
		device = Cuda.cuDeviceGet (devnum);
		name = Cuda.cuDeviceGetName (device);
		major = Cuda.cuDeviceGetAttributeComputeCapabilityMajor (device);
		minor = Cuda.cuDeviceGetAttributeComputeCapabilityMinor (device);
		mpcount = Cuda.cuDeviceGetAttributeMultiprocessorCount (device);
		ctx = Cuda.cuCtxCreate (Cuda.J_CU_CTX_SCHED_AUTO, device);
		}

// Exported operations.

	/**
	 * Get the number of GPU devices in the system.
	 *
	 * @return  Number of GPU devices.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public static synchronized int deviceCount()
		{
		return Cuda.libraryLoaded ? Cuda.cuDeviceGetCount() : 0;
		}

	/**
	 * Specify the GPU device number or numbers this process is allowed to use.
	 * If not specified, this process is allowed to use all devices.
	 * <P>
	 * <B><I>Warning:</I></B> The <TT>setDeviceNumbers()</TT> method is intended
	 * to be called only by the {@link pj2 pj2} launcher program. Do not call
	 * the <TT>setDeviceNumbers()</TT> method yourself.
	 *
	 * @param  devnum
	 *     Array of zero or more device numbers. Each must be in the range 0
	 *     through {@link #deviceCount() deviceCount()}&minus;1. If
	 *     <TT>devnum</TT> is null, this process is allowed to use all devices.
	 *     If <TT>devnum</TT> is zero length, this process is not allowed to use
	 *     any devices.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if any device number is out of bounds.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public static synchronized void setDeviceNumbers
		(int[] devnum)
		{
		int count = deviceCount();
		if (devnum == null)
			{
			allowedDeviceNumbers = new int [count];
			for (int i = 0; i < count; ++ i)
				allowedDeviceNumbers[i] = i;
			}
		else
			{
			for (int i : devnum)
				if (0 > i || i >= count)
					throw new IndexOutOfBoundsException (String.format
						("Gpu.setDeviceNumbers(): Device number %d out of bounds",
						 i));
			allowedDeviceNumbers = (int[]) devnum.clone();
			}
		}

	/**
	 * Get the number of GPU devices this process is allowed to use.
	 *
	 * @return  Number of GPU devices.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public static synchronized int allowedDeviceCount()
		{
		return allowedDeviceNumbers.length;
		}

	/**
	 * Construct a new Gpu object. The next available GPU device is used.
	 *
	 * @return  Gpu object.
	 *
	 * @exception  IllegalStateException
	 *     (unchecked exception) Thrown if there are no available GPU devices.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public static synchronized Gpu gpu()
		{
		if (! Cuda.libraryLoaded)
			throw new GpuNotSupportedException
				("Gpu.gpu(): GPU parallel programs are not supported");
		if (inUseDevices >= allowedDeviceNumbers.length)
			throw new IllegalStateException
				("Gpu.gpu(): No GPU devices available");
		return new Gpu (allowedDeviceNumbers[inUseDevices++]);
		}

	/**
	 * Ensure that this Gpu object supports at least the given compute
	 * capability. If so, this method merely returns. If not, this method throws
	 * a {@linkplain GpuException}.
	 *
	 * @param  major  Required compute capability major version.
	 * @param  minor  Required compute capability minor version.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if this Gpu object does not support at
	 *     least the given compute capability. Thrown if a GPU error occurred.
	 */
	public void ensureComputeCapability
		(int major,
		 int minor)
		{
		if (this.major > major) return;
		if (this.major == major && this.minor >= minor) return;
		throw new GpuException (String.format
			("Gpu.ensureComputeCapability(): %d.%d required, %d.%d supported",
			 major, minor, this.major, this.minor));
		}

	/**
	 * Get the number of multiprocessors in this GPU.
	 *
	 * @return  Number of multiprocessors.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public int getMultiprocessorCount()
		{
		return mpcount;
		}

	/**
	 * Set this GPU's cache configuration. All kernel functions that do not
	 * themselves set the cache configuration will use the given cache
	 * configuration.
	 *
	 * @param  config  Cache configuration.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>config</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void setCacheConfig
		(CacheConfig config)
		{
		Cuda.cuCtxSetCacheConfig (ctx, config.value);
		}

	/**
	 * Get this GPU's cache configuration. All kernel functions that do not
	 * themselves set the cache configuration will use the returned cache
	 * configuration.
	 *
	 * @return  Cache configuration.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public CacheConfig getCacheConfig()
		{
		return CacheConfig.of (Cuda.cuCtxGetCacheConfig (ctx));
		}

	/**
	 * Create a new dynamically allocated GPU byte array. The entire GPU array
	 * is mirrored in the CPU. The array elements in CPU memory, namely the
	 * {@link GpuByteArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len  Number of array elements (&ge; 1).
	 *
	 * @return  GPU byte array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getByteArray(int,int)
	 */
	public GpuByteArray getByteArray
		(int len)
		{
		return getByteArray (len, len);
		}

	/**
	 * Create a new dynamically allocated GPU byte array, of which the given
	 * portion is mirrored in the CPU. If <TT>cpulen</TT> = 0, then none of the
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuByteArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU byte array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getByteArray(int)
	 */
	public GpuByteArray getByteArray
		(int len,
		 int cpulen)
		{
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getByteArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Gpu.getByteArray(): cpulen = %d illegal", cpulen));
		return new GpuByteArray (this, len, cpulen);
		}

	/**
	 * Create a new dynamically allocated GPU short integer array. The entire
	 * GPU array is mirrored in the CPU. The array elements in CPU memory,
	 * namely the {@link GpuShortArray#item item} field's elements, are
	 * initialized to zeroes. <I>The array elements in GPU memory are not
	 * initialized.</I>
	 *
	 * @param  len  Number of array elements (&ge; 1).
	 *
	 * @return  GPU short integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getShortArray(int,int)
	 */
	public GpuShortArray getShortArray
		(int len)
		{
		return getShortArray (len, len);
		}

	/**
	 * Create a new dynamically allocated GPU short integer array, of which the
	 * given portion is mirrored in the CPU. If <TT>cpulen</TT> = 0, then none
	 * of the array is mirrored in the CPU. The array elements in CPU memory,
	 * namely the {@link GpuShortArray#item item} field's elements, are
	 * initialized to zeroes. <I>The array elements in GPU memory are not
	 * initialized.</I>
	 *
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU short integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getShortArray(int)
	 */
	public GpuShortArray getShortArray
		(int len,
		 int cpulen)
		{
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getShortArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Gpu.getShortArray(): cpulen = %d illegal", cpulen));
		return new GpuShortArray (this, len, cpulen);
		}

	/**
	 * Create a new dynamically allocated GPU integer array. The entire GPU
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuIntArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len  Number of array elements (&ge; 1).
	 *
	 * @return  GPU integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getIntArray(int,int)
	 */
	public GpuIntArray getIntArray
		(int len)
		{
		return getIntArray (len, len);
		}

	/**
	 * Create a new dynamically allocated GPU integer array, of which the given
	 * portion is mirrored in the CPU. If <TT>cpulen</TT> = 0, then none of the
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuIntArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getIntArray(int)
	 */
	public GpuIntArray getIntArray
		(int len,
		 int cpulen)
		{
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getIntArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Gpu.getIntArray(): cpulen = %d illegal", cpulen));
		return new GpuIntArray (this, len, cpulen);
		}

	/**
	 * Create a new dynamically allocated GPU long integer array. The entire GPU
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuLongArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len  Number of array elements (&ge; 1).
	 *
	 * @return  GPU long integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getLongArray(int,int)
	 */
	public GpuLongArray getLongArray
		(int len)
		{
		return getLongArray (len, len);
		}

	/**
	 * Create a new dynamically allocated GPU long integer array, of which the
	 * given portion is mirrored in the CPU. If <TT>cpulen</TT> = 0, then none
	 * of the array is mirrored in the CPU. The array elements in CPU memory,
	 * namely the {@link GpuLongArray#item item} field's elements, are
	 * initialized to zeroes. <I>The array elements in GPU memory are not
	 * initialized.</I>
	 *
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU long integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getLongArray(int)
	 */
	public GpuLongArray getLongArray
		(int len,
		 int cpulen)
		{
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getLongArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Gpu.getLongArray(): cpulen = %d illegal", cpulen));
		return new GpuLongArray (this, len, cpulen);
		}

	/**
	 * Create a new dynamically allocated GPU float array. The entire GPU array
	 * is mirrored in the CPU. The array elements in CPU memory, namely the
	 * {@link GpuFloatArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len  Number of array elements (&ge; 1).
	 *
	 * @return  GPU float array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getFloatArray(int,int)
	 */
	public GpuFloatArray getFloatArray
		(int len)
		{
		return getFloatArray (len, len);
		}

	/**
	 * Create a new dynamically allocated GPU float array, of which the given
	 * portion is mirrored in the CPU. If <TT>cpulen</TT> = 0, then none of the
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuFloatArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU float array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getFloatArray(int)
	 */
	public GpuFloatArray getFloatArray
		(int len,
		 int cpulen)
		{
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getFloatArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Gpu.getFloatArray(): cpulen = %d illegal", cpulen));
		return new GpuFloatArray (this, len, cpulen);
		}

	/**
	 * Create a new dynamically allocated GPU double array. The entire GPU array
	 * is mirrored in the CPU. The array elements in CPU memory, namely the
	 * {@link GpuDoubleArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len  Number of array elements (&ge; 1).
	 *
	 * @return  GPU double array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getDoubleArray(int,int)
	 */
	public GpuDoubleArray getDoubleArray
		(int len)
		{
		return getDoubleArray (len, len);
		}

	/**
	 * Create a new dynamically allocated GPU double array, of which the given
	 * portion is mirrored in the CPU. If <TT>cpulen</TT> = 0, then none of the
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuDoubleArray#item item} field's elements, are initialized to
	 * zeroes. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU double array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getDoubleArray(int)
	 */
	public GpuDoubleArray getDoubleArray
		(int len,
		 int cpulen)
		{
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getDoubleArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Gpu.getDoubleArray(): cpulen = %d illegal", cpulen));
		return new GpuDoubleArray (this, len, cpulen);
		}

	/**
	 * Create a new dynamically allocated GPU byte matrix. The entire GPU
	 * matrix is mirrored in the CPU. The matrix elements in CPU memory, namely
	 * the {@link GpuByteMatrix#item item} field's elements, are initialized to
	 * zeroes. <I>The matrix elements in GPU memory are not initialized.</I>
	 *
	 * @param  rows  Number of matrix rows (&ge; 1).
	 * @param  cols  Number of matrix columns (&ge; 1).
	 *
	 * @return  GPU byte matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getByteMatrix(int,int,int,int)
	 */
	public GpuByteMatrix getByteMatrix
		(int rows,
		 int cols)
		{
		return getByteMatrix (rows, cols, rows, cols);
		}

	/**
	 * Create a new dynamically allocated GPU byte matrix, of which the given
	 * portion is mirrored in the CPU. If <TT>cpurows</TT> = 0 or
	 * <TT>cpucols</TT> = 0, then none of the matrix is mirrored in the CPU. The
	 * matrix elements in CPU memory, namely the {@link GpuByteMatrix#item item}
	 * field's elements, are initialized to zeroes. <I>The matrix elements in
	 * GPU memory are not initialized.</I>
	 *
	 * @param  rows     Number of matrix rows (&ge; 1).
	 * @param  cols     Number of matrix columns (&ge; 1).
	 * @param  cpurows  Number of matrix rows to mirror in the CPU.
	 * @param  cpucols  Number of matrix columns to mirror in the CPU.
	 *
	 * @return  GPU byte matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1, <TT>cols</TT>
	 *     &lt; 1, <TT>cpurows</TT> &lt; 0, <TT>cpurows</TT> &gt; <TT>rows</TT>,
	 *     <TT>cpucols</TT> &lt; 0, or <TT>cpucols</TT> &gt; <TT>cols</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getByteMatrix(int,int)
	 */
	public GpuByteMatrix getByteMatrix
		(int rows,
		 int cols,
		 int cpurows,
		 int cpucols)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getByteMatrix(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getByteMatrix(): cols = %d illegal", cols));
		if (0 > cpurows || cpurows > rows)
			throw new IllegalArgumentException (String.format
				("Gpu.getByteMatrix(): cpurows = %d illegal", cpurows));
		if (0 > cpucols || cpucols > cols)
			throw new IllegalArgumentException (String.format
				("Gpu.getByteMatrix(): cpucols = %d illegal", cpucols));
		return new GpuByteMatrix (this, rows, cols, cpurows, cpucols);
		}

	/**
	 * Create a new dynamically allocated GPU short integer matrix. The entire
	 * GPU matrix is mirrored in the CPU. The matrix elements in CPU memory,
	 * namely the {@link GpuShortMatrix#item item} field's elements, are
	 * initialized to zeroes. <I>The matrix elements in GPU memory are not
	 * initialized.</I>
	 *
	 * @param  rows  Number of matrix rows (&ge; 1).
	 * @param  cols  Number of matrix columns (&ge; 1).
	 *
	 * @return  GPU short integer matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getShortMatrix(int,int,int,int)
	 */
	public GpuShortMatrix getShortMatrix
		(int rows,
		 int cols)
		{
		return getShortMatrix (rows, cols, rows, cols);
		}

	/**
	 * Create a new dynamically allocated GPU short integer matrix, of which the
	 * given portion is mirrored in the CPU. If <TT>cpurows</TT> = 0 or
	 * <TT>cpucols</TT> = 0, then none of the matrix is mirrored in the CPU. The
	 * matrix elements in CPU memory, namely the {@link GpuShortMatrix#item
	 * item} field's elements, are initialized to zeroes. <I>The matrix elements
	 * in GPU memory are not initialized.</I>
	 *
	 * @param  rows     Number of matrix rows (&ge; 1).
	 * @param  cols     Number of matrix columns (&ge; 1).
	 * @param  cpurows  Number of matrix rows to mirror in the CPU.
	 * @param  cpucols  Number of matrix columns to mirror in the CPU.
	 *
	 * @return  GPU short integer matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1, <TT>cols</TT>
	 *     &lt; 1, <TT>cpurows</TT> &lt; 0, <TT>cpurows</TT> &gt; <TT>rows</TT>,
	 *     <TT>cpucols</TT> &lt; 0, or <TT>cpucols</TT> &gt; <TT>cols</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getShortMatrix(int,int)
	 */
	public GpuShortMatrix getShortMatrix
		(int rows,
		 int cols,
		 int cpurows,
		 int cpucols)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getShortMatrix(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getShortMatrix(): cols = %d illegal", cols));
		if (0 > cpurows || cpurows > rows)
			throw new IllegalArgumentException (String.format
				("Gpu.getShortMatrix(): cpurows = %d illegal", cpurows));
		if (0 > cpucols || cpucols > cols)
			throw new IllegalArgumentException (String.format
				("Gpu.getShortMatrix(): cpucols = %d illegal", cpucols));
		return new GpuShortMatrix (this, rows, cols, cpurows, cpucols);
		}

	/**
	 * Create a new dynamically allocated GPU integer matrix. The entire GPU
	 * matrix is mirrored in the CPU. The matrix elements in CPU memory, namely
	 * the {@link GpuIntMatrix#item item} field's elements, are initialized to
	 * zeroes. <I>The matrix elements in GPU memory are not initialized.</I>
	 *
	 * @param  rows  Number of matrix rows (&ge; 1).
	 * @param  cols  Number of matrix columns (&ge; 1).
	 *
	 * @return  GPU integer matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getIntMatrix(int,int,int,int)
	 */
	public GpuIntMatrix getIntMatrix
		(int rows,
		 int cols)
		{
		return getIntMatrix (rows, cols, rows, cols);
		}

	/**
	 * Create a new dynamically allocated GPU integer matrix, of which the given
	 * portion is mirrored in the CPU. If <TT>cpurows</TT> = 0 or
	 * <TT>cpucols</TT> = 0, then none of the matrix is mirrored in the CPU. The
	 * matrix elements in CPU memory, namely the {@link GpuIntMatrix#item item}
	 * field's elements, are initialized to zeroes. <I>The matrix elements in
	 * GPU memory are not initialized.</I>
	 *
	 * @param  rows     Number of matrix rows (&ge; 1).
	 * @param  cols     Number of matrix columns (&ge; 1).
	 * @param  cpurows  Number of matrix rows to mirror in the CPU.
	 * @param  cpucols  Number of matrix columns to mirror in the CPU.
	 *
	 * @return  GPU integer matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1, <TT>cols</TT>
	 *     &lt; 1, <TT>cpurows</TT> &lt; 0, <TT>cpurows</TT> &gt; <TT>rows</TT>,
	 *     <TT>cpucols</TT> &lt; 0, or <TT>cpucols</TT> &gt; <TT>cols</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getIntMatrix(int,int)
	 */
	public GpuIntMatrix getIntMatrix
		(int rows,
		 int cols,
		 int cpurows,
		 int cpucols)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getIntMatrix(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getIntMatrix(): cols = %d illegal", cols));
		if (0 > cpurows || cpurows > rows)
			throw new IllegalArgumentException (String.format
				("Gpu.getIntMatrix(): cpurows = %d illegal", cpurows));
		if (0 > cpucols || cpucols > cols)
			throw new IllegalArgumentException (String.format
				("Gpu.getIntMatrix(): cpucols = %d illegal", cpucols));
		return new GpuIntMatrix (this, rows, cols, cpurows, cpucols);
		}

	/**
	 * Create a new dynamically allocated GPU long integer matrix. The entire
	 * GPU matrix is mirrored in the CPU. The matrix elements in CPU memory,
	 * namely the {@link GpuLongMatrix#item item} field's elements, are
	 * initialized to zeroes. <I>The matrix elements in GPU memory are not
	 * initialized.</I>
	 *
	 * @param  rows  Number of matrix rows (&ge; 1).
	 * @param  cols  Number of matrix columns (&ge; 1).
	 *
	 * @return  GPU long integer matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getLongMatrix(int,int,int,int)
	 */
	public GpuLongMatrix getLongMatrix
		(int rows,
		 int cols)
		{
		return getLongMatrix (rows, cols, rows, cols);
		}

	/**
	 * Create a new dynamically allocated GPU long integer matrix, of which the
	 * given portion is mirrored in the CPU. If <TT>cpurows</TT> = 0 or
	 * <TT>cpucols</TT> = 0, then none of the matrix is mirrored in the CPU. The
	 * matrix elements in CPU memory, namely the {@link GpuLongMatrix#item item}
	 * field's elements, are initialized to zeroes. <I>The matrix elements in
	 * GPU memory are not initialized.</I>
	 *
	 * @param  rows     Number of matrix rows (&ge; 1).
	 * @param  cols     Number of matrix columns (&ge; 1).
	 * @param  cpurows  Number of matrix rows to mirror in the CPU.
	 * @param  cpucols  Number of matrix columns to mirror in the CPU.
	 *
	 * @return  GPU long integer matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1, <TT>cols</TT>
	 *     &lt; 1, <TT>cpurows</TT> &lt; 0, <TT>cpurows</TT> &gt; <TT>rows</TT>,
	 *     <TT>cpucols</TT> &lt; 0, or <TT>cpucols</TT> &gt; <TT>cols</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getLongMatrix(int,int)
	 */
	public GpuLongMatrix getLongMatrix
		(int rows,
		 int cols,
		 int cpurows,
		 int cpucols)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getLongMatrix(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getLongMatrix(): cols = %d illegal", cols));
		if (0 > cpurows || cpurows > rows)
			throw new IllegalArgumentException (String.format
				("Gpu.getLongMatrix(): cpurows = %d illegal", cpurows));
		if (0 > cpucols || cpucols > cols)
			throw new IllegalArgumentException (String.format
				("Gpu.getLongMatrix(): cpucols = %d illegal", cpucols));
		return new GpuLongMatrix (this, rows, cols, cpurows, cpucols);
		}

	/**
	 * Create a new dynamically allocated GPU float matrix. The entire GPU
	 * matrix is mirrored in the CPU. The matrix elements in CPU memory, namely
	 * the {@link GpuFloatMatrix#item item} field's elements, are initialized to
	 * zeroes. <I>The matrix elements in GPU memory are not initialized.</I>
	 *
	 * @param  rows  Number of matrix rows (&ge; 1).
	 * @param  cols  Number of matrix columns (&ge; 1).
	 *
	 * @return  GPU float matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getFloatMatrix(int,int,int,int)
	 */
	public GpuFloatMatrix getFloatMatrix
		(int rows,
		 int cols)
		{
		return getFloatMatrix (rows, cols, rows, cols);
		}

	/**
	 * Create a new dynamically allocated GPU float matrix, of which the given
	 * portion is mirrored in the CPU. If <TT>cpurows</TT> = 0 or
	 * <TT>cpucols</TT> = 0, then none of the matrix is mirrored in the CPU. The
	 * matrix elements in CPU memory, namely the {@link GpuFloatMatrix#item
	 * item} field's elements, are initialized to zeroes. <I>The matrix elements
	 * in GPU memory are not initialized.</I>
	 *
	 * @param  rows     Number of matrix rows (&ge; 1).
	 * @param  cols     Number of matrix columns (&ge; 1).
	 * @param  cpurows  Number of matrix rows to mirror in the CPU.
	 * @param  cpucols  Number of matrix columns to mirror in the CPU.
	 *
	 * @return  GPU float matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1, <TT>cols</TT>
	 *     &lt; 1, <TT>cpurows</TT> &lt; 0, <TT>cpurows</TT> &gt; <TT>rows</TT>,
	 *     <TT>cpucols</TT> &lt; 0, or <TT>cpucols</TT> &gt; <TT>cols</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getFloatMatrix(int,int)
	 */
	public GpuFloatMatrix getFloatMatrix
		(int rows,
		 int cols,
		 int cpurows,
		 int cpucols)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getFloatMatrix(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getFloatMatrix(): cols = %d illegal", cols));
		if (0 > cpurows || cpurows > rows)
			throw new IllegalArgumentException (String.format
				("Gpu.getFloatMatrix(): cpurows = %d illegal", cpurows));
		if (0 > cpucols || cpucols > cols)
			throw new IllegalArgumentException (String.format
				("Gpu.getFloatMatrix(): cpucols = %d illegal", cpucols));
		return new GpuFloatMatrix (this, rows, cols, cpurows, cpucols);
		}

	/**
	 * Create a new dynamically allocated GPU double matrix. The entire GPU
	 * matrix is mirrored in the CPU. The matrix elements in CPU memory, namely
	 * the {@link GpuDoubleMatrix#item item} field's elements, are initialized
	 * to zeroes. <I>The matrix elements in GPU memory are not initialized.</I>
	 *
	 * @param  rows  Number of matrix rows (&ge; 1).
	 * @param  cols  Number of matrix columns (&ge; 1).
	 *
	 * @return  GPU double matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1 or <TT>cols</TT>
	 *     &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getDoubleMatrix(int,int,int,int)
	 */
	public GpuDoubleMatrix getDoubleMatrix
		(int rows,
		 int cols)
		{
		return getDoubleMatrix (rows, cols, rows, cols);
		}

	/**
	 * Create a new dynamically allocated GPU double matrix, of which the given
	 * portion is mirrored in the CPU. If <TT>cpurows</TT> = 0 or
	 * <TT>cpucols</TT> = 0, then none of the matrix is mirrored in the CPU. The
	 * matrix elements in CPU memory, namely the {@link GpuDoubleMatrix#item
	 * item} field's elements, are initialized to zeroes. <I>The matrix elements
	 * in GPU memory are not initialized.</I>
	 *
	 * @param  rows     Number of matrix rows (&ge; 1).
	 * @param  cols     Number of matrix columns (&ge; 1).
	 * @param  cpurows  Number of matrix rows to mirror in the CPU.
	 * @param  cpucols  Number of matrix columns to mirror in the CPU.
	 *
	 * @return  GPU double matrix.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>rows</TT> &lt; 1, <TT>cols</TT>
	 *     &lt; 1, <TT>cpurows</TT> &lt; 0, <TT>cpurows</TT> &gt; <TT>rows</TT>,
	 *     <TT>cpucols</TT> &lt; 0, or <TT>cpucols</TT> &gt; <TT>cols</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getDoubleMatrix(int,int)
	 */
	public GpuDoubleMatrix getDoubleMatrix
		(int rows,
		 int cols,
		 int cpurows,
		 int cpucols)
		{
		if (rows < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getDoubleMatrix(): rows = %d illegal", rows));
		if (cols < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getDoubleMatrix(): cols = %d illegal", cols));
		if (0 > cpurows || cpurows > rows)
			throw new IllegalArgumentException (String.format
				("Gpu.getDoubleMatrix(): cpurows = %d illegal", cpurows));
		if (0 > cpucols || cpucols > cols)
			throw new IllegalArgumentException (String.format
				("Gpu.getDoubleMatrix(): cpucols = %d illegal", cpucols));
		return new GpuDoubleMatrix (this, rows, cols, cpurows, cpucols);
		}

	/**
	 * Create a new dynamically allocated GPU struct variable. The variable in
	 * CPU memory, namely the {@link GpuStructVbl#item item} field, is
	 * initialized to null. <I>The variable in GPU memory is not
	 * initialized.</I>
	 *
	 * @param  <T>   Java data type. Class <TT>T</TT> must extend class
	 *               {@linkplain Struct Struct}.
	 * @param  type  Class object for Java data type.
	 *
	 * @return  GPU struct variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>type</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public <T extends Struct> GpuStructVbl<T> getStructVbl
		(Class<T> type)
		{
		return new GpuStructVbl (this, type);
		}

	/**
	 * Create a new dynamically allocated GPU struct array. The entire GPU
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuStructArray#item item} field's elements, are initialized to
	 * null. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  <T>   Java data type. Class <TT>T</TT> must extend class
	 *               {@linkplain Struct Struct}.
	 * @param  type  Class object for Java data type.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU struct array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>type</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getStructArray(Class,int,int)
	 */
	public <T extends Struct> GpuStructArray<T> getStructArray
		(Class<T> type,
		 int len)
		{
		return getStructArray (type, len, len);
		}

	/**
	 * Create a new dynamically allocated GPU struct array, of which the given
	 * portion is mirrored in the CPU. If <TT>cpulen</TT> = 0, then none of the
	 * array is mirrored in the CPU. The array elements in CPU memory, namely
	 * the {@link GpuStructArray#item item} field's elements, are initialized to
	 * null. <I>The array elements in GPU memory are not initialized.</I>
	 *
	 * @param  <T>     Java data type. Class <TT>T</TT> must extend class
	 *                 {@linkplain Struct Struct}.
	 * @param  type    Class object for Java data type.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU struct array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>type</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getStructArray(Class,int)
	 */
	public <T extends Struct> GpuStructArray<T> getStructArray
		(Class<T> type,
		 int len,
		 int cpulen)
		{
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Gpu.getStructArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Gpu.getStructArray(): cpulen = %d illegal", cpulen));
		return new GpuStructArray (this, len, cpulen, type);
		}

	/**
	 * Get a module containing GPU functions and data. The module file is a
	 * resource stored in the Java class path.
	 *
	 * @param  name  Module resource name relative to the Java class path.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public Module getModule
		(String name)
		throws IOException
		{
		return new Module (this, name);
		}

	/**
	 * Returns a string version of this Gpu object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("CUDA device %d, %s, compute capability %d.%d",
			devnum, name, major, minor);
		}

// Hidden operations.

	/**
	 * Finalize this Gpu object.
	 */
	protected synchronized void finalize()
		{
		Cuda.cuCtxDestroy (ctx);
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		{
//		Gpu gpu = Gpu.gpu();
//		System.out.printf ("gpu = %s%n", gpu);
//		gpu.ensureComputeCapability (1, 99);
//		System.out.printf ("multiprocessors = %d%n", gpu.getMultiprocessorCount());
//		System.out.printf ("cache config = %s%n", gpu.getCacheConfig());
//		gpu.setCacheConfig (CacheConfig.CU_FUNC_CACHE_PREFER_L1);
//		System.out.printf ("cache config = %s%n", gpu.getCacheConfig());
//		}

	}
