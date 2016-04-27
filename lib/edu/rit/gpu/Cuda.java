//******************************************************************************
//
// File:    Cuda.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.Cuda
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

import edu.rit.io.FileResource;
import java.io.IOException;

/**
 * Class Cuda contains static native method declarations for CUDA routines.
 *
 * @author  Alan Kaminsky
 * @version 25-Mar-2014
 */
class Cuda
	{

// Load native library.

	static boolean libraryLoaded = false;

	static
		{
		try
			{
			System.loadLibrary ("EduRitGpuCuda");
			libraryLoaded = true;
			}
		catch (Throwable exc)
			{
			}
		}

// Exported constants.

	// Values for the cuCtxCreate() flags parameter.
    public static final int J_CU_CTX_SCHED_AUTO          = 0x00;
    public static final int J_CU_CTX_SCHED_SPIN          = 0x01;
    public static final int J_CU_CTX_SCHED_YIELD         = 0x02;
    public static final int J_CU_CTX_SCHED_BLOCKING_SYNC = 0x04;
    public static final int J_CU_CTX_SCHED_MASK          = 0x07;
    public static final int J_CU_CTX_MAP_HOST            = 0x08;
    public static final int J_CU_CTX_LMEM_RESIZE_TO_MAX  = 0x10;
    public static final int J_CU_CTX_FLAGS_MASK          = 0x1f;

// Exported operations.

	/**
	 * Determine if pointers are 64-bit or 32-bit.
	 *
	 * @return  True if pointers are 64-bit, false if pointers are 32-bit.
	 */
	public static native boolean is64BitPointer();

	/**
	 * Initialize the CUDA driver API.
	 *
	 * @param  flags  Initialization flags; currently must be 0.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuInit
		(int flags);

	/**
	 * Get the number of compute devices.
	 *
	 * @return  Number of compute devices.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native int cuDeviceGetCount();

	/**
	 * Get a handle to the given compute device.
	 *
	 * @param  ordinal  Device number in the range 0 through
	 *                  <TT>cuDeviceGetCount()</TT>&minus;1.
	 *
	 * @return  Device handle.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native int cuDeviceGet
		(int ordinal);

	/**
	 * Get the name of the given compute device.
	 *
	 * @param  device  Device handle.
	 *
	 * @return  Device name.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native String cuDeviceGetName
		(int device);

	/**
	 * Get the major compute capability version of the given compute device.
	 *
	 * @param  device  Device handle.
	 *
	 * @return  Major compute capability version.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native int cuDeviceGetAttributeComputeCapabilityMajor
		(int device);

	/**
	 * Get the minor compute capability version of the given compute device.
	 *
	 * @param  device  Device handle.
	 *
	 * @return  Minor compute capability version.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native int cuDeviceGetAttributeComputeCapabilityMinor
		(int device);

	/**
	 * Get the multiprocessor count of the given compute device.
	 *
	 * @param  device  Device handle.
	 *
	 * @return  Multiprocessor count.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native int cuDeviceGetAttributeMultiprocessorCount
		(int device);

	/**
	 * Create a CUDA context in the current thread.
	 *
	 * @param  flags   Context creation flags.
	 * @param  device  Device handle.
	 *
	 * @return  CUDA context.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native long cuCtxCreate
		(int flags,
		 int device);

	/**
	 * Destroy the given CUDA context.
	 *
	 * @param  ctx  CUDA context.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuCtxDestroy
		(long ctx);

	/**
	 * Set the cache configuration for the given CUDA context.
	 *
	 * @param  ctx     CUDA context.
	 * @param  config  Cache configuration enumeral value.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuCtxSetCacheConfig
		(long ctx,
		 int config);

	/**
	 * Get the cache configuration for the given CUDA context.
	 *
	 * @param  ctx  CUDA context.
	 *
	 * @return  Cache configuration enumeral value.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native int cuCtxGetCacheConfig
		(long ctx);

	/**
	 * Allocate a device memory block. The memory block is not initialized.
	 *
	 * @param  ctx       CUDA context.
	 * @param  bytesize  Number of bytes to allocate.
	 *
	 * @return  Pointer to GPU memory block.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native long cuMemAlloc
		(long ctx,
		 long bytesize);

	/**
	 * Deallocate a device memory block.
	 *
	 * @param  ctx   CUDA context.
	 * @param  dptr  Pointer to GPU memory block.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemFree
		(long ctx,
		 long dptr);

	/**
	 * Copy a portion of a byte array from host to device.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Pointer to GPU memory block.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Source array.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyHtoD
		(long ctx,
		 long dst,
		 int dstindex,
		 byte[] src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a byte array from device to host.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Destination array.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Pointer to GPU memory block.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyDtoH
		(long ctx,
		 byte[] dst,
		 int dstindex,
		 long src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a short array from host to device.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Pointer to GPU memory block.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Source array.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyHtoD
		(long ctx,
		 long dst,
		 int dstindex,
		 short[] src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a short array from device to host.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Destination array.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Pointer to GPU memory block.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyDtoH
		(long ctx,
		 short[] dst,
		 int dstindex,
		 long src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of an integer array from host to device.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Pointer to GPU memory block.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Source array.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyHtoD
		(long ctx,
		 long dst,
		 int dstindex,
		 int[] src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of an integer array from device to host.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Destination array.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Pointer to GPU memory block.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyDtoH
		(long ctx,
		 int[] dst,
		 int dstindex,
		 long src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a long array from host to device.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Pointer to GPU memory block.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Source array.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyHtoD
		(long ctx,
		 long dst,
		 int dstindex,
		 long[] src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a long array from device to host.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Destination array.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Pointer to GPU memory block.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyDtoH
		(long ctx,
		 long[] dst,
		 int dstindex,
		 long src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a float array from host to device.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Pointer to GPU memory block.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Source array.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyHtoD
		(long ctx,
		 long dst,
		 int dstindex,
		 float[] src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a float array from device to host.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Destination array.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Pointer to GPU memory block.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyDtoH
		(long ctx,
		 float[] dst,
		 int dstindex,
		 long src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a double array from host to device.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Pointer to GPU memory block.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Source array.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyHtoD
		(long ctx,
		 long dst,
		 int dstindex,
		 double[] src,
		 int srcindex,
		 int nelem);

	/**
	 * Copy a portion of a double array from device to host.
	 *
	 * @param  ctx       CUDA context.
	 * @param  dst       Destination array.
	 * @param  dstindex  Destination starting index.
	 * @param  src       Pointer to GPU memory block.
	 * @param  srcindex  Source starting index.
	 * @param  nelem     Number of elements to copy.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuMemcpyDtoH
		(long ctx,
		 double[] dst,
		 int dstindex,
		 long src,
		 int srcindex,
		 int nelem);

	/**
	 * Load a module into the given CUDA context.
	 *
	 * @param  ctx    CUDA context.
	 * @param  fname  Module file name.
	 *
	 * @return  Module handle.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native long cuModuleLoad
		(long ctx,
		 String fname);

	/**
	 * Unload the given module from the given CUDA context.
	 *
	 * @param  ctx  CUDA context.
	 * @param  mod  Module handle.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuModuleUnload
		(long ctx,
		 long mod);

	/**
	 * Get a function from the given module in the given CUDA context.
	 *
	 * @param  ctx   CUDA context.
	 * @param  mod   Module handle.
	 * @param  name  Function name.
	 *
	 * @return  Function handle.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native long cuModuleGetFunction
		(long ctx,
		 long mod,
		 String name);

	/**
	 * Get a global variable pointer from the given module in the given CUDA
	 * context.
	 *
	 * @param  ctx   CUDA context.
	 * @param  mod   Module handle.
	 * @param  name  Global variable name.
	 *
	 * @return  Global variable pointer.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native long cuModuleGetGlobal
		(long ctx,
		 long mod,
		 String name);

	/**
	 * Set the cache configuration for the given device function in the given
	 * CUDA context.
	 *
	 * @param  ctx     CUDA context.
	 * @param  func    Function handle.
	 * @param  config  Cache configuration enumeral value.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuFuncSetCacheConfig
		(long ctx,
		 long func,
		 int config);

	/**
	 * Launch a GPU kernel in the given CUDA context.
	 *
	 * @param  ctx             CUDA context.
	 * @param  func            Function handle.
	 * @param  gridDimX        Grid X dimension.
	 * @param  gridDimY        Grid Y dimension.
	 * @param  gridDimZ        Grid Z dimension.
	 * @param  blockDimX       Block X dimension.
	 * @param  blockDimY       Block Y dimension.
	 * @param  blockDimZ       Block Z dimension.
	 * @param  sharedMemBytes  Dynamically allocated shared memory size.
	 * @param  argc            Number of kernel function arguments.
	 * @param  argv            Buffer for argument values.
	 * @param  argp            Buffer for argument pointers.
	 *
	 * @exception  CudaException
	 *     (unchecked exception) Thrown if a CUDA error occurred.
	 */
	public static native void cuLaunchKernel
		(long ctx,
		 long func,
		 int gridDimX,
		 int gridDimY,
		 int gridDimZ,
		 int blockDimX,
		 int blockDimY,
		 int blockDimZ,
		 int sharedMemBytes,
		 int argc,
		 byte[] argv,
		 byte[] argp);

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		System.out.printf ("is64BitPointer() = %b%n", is64BitPointer());
//		cuInit (0);
//		System.out.printf ("cuInit() worked!%n");
//		int count = cuDeviceGetCount();
//		System.out.printf ("count = %d%n", count);
//		int device = cuDeviceGet (0);
//		System.out.printf ("device = %d%n", device);
//		String name = cuDeviceGetName (device);
//		System.out.printf ("name = %s%n", name);
//		int major = cuDeviceGetAttributeComputeCapabilityMajor (device);
//		int minor = cuDeviceGetAttributeComputeCapabilityMinor (device);
//		System.out.printf ("compute capability = %d.%d%n", major, minor);
//		long ctx = cuCtxCreate (0, device);
//		System.out.printf ("ctx = %016x%n", ctx);
//		long dptr = cuMemAlloc (ctx, 1024);
//		System.out.printf ("dptr = %08x%n", dptr);
//		cuMemFree (ctx, dptr);
//		System.out.printf ("cuMemFree() worked!%n");
//		cuCtxDestroy (ctx);
//		System.out.printf ("cuCtxDestroy() worked!%n");
//		}

	}
