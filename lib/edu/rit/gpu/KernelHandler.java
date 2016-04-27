//******************************************************************************
//
// File:    KernelHandler.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.KernelHandler
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

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Class KernelHandler provides the invocation handler for a dynamic proxy for
 * interface {@linkplain Kernel Kernel}.
 *
 * @author  Alan Kaminsky
 * @version 03-Mar-2014
 */
class KernelHandler
	implements InvocationHandler
	{

// Hidden data members.

	private Gpu gpu;     // Gpu object that created this
	private String name; // Function name
	private long func;   // Function handle
	private int argc;    // Number of arguments

	private byte[] argv;    // Buffer for argument values
	private byte[] argp;    // Buffer for argument pointers
	private ByteBuffer buf; // For writing argument values

	private int gridDimX = 1; // Grid dimensions
	private int gridDimY = 1;
	private int gridDimZ = 1;
	private int blockDimX = 1; // Block dimensions
	private int blockDimY = 1;
	private int blockDimZ = 1;
	private int sharedMemBytes = 0; // Dynamically allocated shared memory size

// Exported constructors.

	/**
	 * Construct a new kernel handler.
	 *
	 * @param  gpu   Gpu object.
	 * @param  name  Function name.
	 * @param  func  Function handle.
	 * @param  argc  Number of arguments.
	 */
	public KernelHandler
		(Gpu gpu,
		 String name,
		 long func,
		 int argc)
		{
		this.gpu = gpu;
		this.name = name;
		this.func = func;
		this.argc = argc;

		argv = new byte [8*argc];
		argp = new byte [8*argc];
		buf = ByteBuffer.wrap (argv);
		buf.order (ByteOrder.LITTLE_ENDIAN);
		}

// Exported operations.

	/**
	 * Invoke the given method.
	 *
	 * @param  proxy   Proxy object.
	 * @param  method  Method.
	 * @param  args    Method arguments.
	 */
	public Object invoke
		(Object proxy,
		 Method method,
		 Object[] args)
		{
		String methodName = method.getName();
		if (methodName.equals (name))
			return launchKernel (args);
		else if (methodName.equals ("setGridDim"))
			return setGridDim (args);
		else if (methodName.equals ("setBlockDim"))
			return setBlockDim (args);
		else if (methodName.equals ("setCacheConfig"))
			return setCacheConfig (args);
		else if (methodName.equals ("setDynamicSharedMemorySize"))
			return setDynamicSharedMemorySize (args);
		else if (methodName.equals ("equals"))
			return doEquals (proxy, args);
		else if (methodName.equals ("hashCode"))
			return doHashCode (proxy);
		else if (methodName.equals ("toString"))
			return doToString();
		else
			throw new UnsupportedOperationException (String.format
				("KernelHandler.invoke(): Unknown method \"%s\"",
				 methodName));
		}

// Hidden operations.

	/**
	 * Launch the GPU kernel function.
	 *
	 * @param  args  Method arguments.
	 */
	private Object launchKernel
		(Object[] args)
		{
		// Serialize argument values into argv byte array.
		Arrays.fill (argv, (byte)0);
		buf.clear();
		for (int i = 0; i < args.length; ++ i)
			{
			if (args[i] == null)
				throw new NullPointerException (String.format
					("Kernel.%s(): Argument %d is null", name, i));
			else if (args[i] instanceof Boolean)
				argv[8*i] = ((Boolean)(args[i])).booleanValue() ?
					(byte)1 : (byte)0;
			else if (args[i] instanceof Byte)
				argv[8*i] = ((Byte)(args[i])).byteValue();
			else if (args[i] instanceof Short)
				buf.putShort (8*i, ((Short)(args[i])).shortValue());
			else if (args[i] instanceof Character)
				buf.putChar (8*i, ((Character)(args[i])).charValue());
			else if (args[i] instanceof Integer)
				buf.putInt (8*i, ((Integer)(args[i])).intValue());
			else if (args[i] instanceof Long)
				buf.putLong (8*i, ((Long)(args[i])).longValue());
			else if (args[i] instanceof Float)
				buf.putFloat (8*i, ((Float)(args[i])).floatValue());
			else if (args[i] instanceof Double)
				buf.putDouble (8*i, ((Double)(args[i])).doubleValue());
			else if (args[i] instanceof GpuVbl)
				buf.putLong (8*i, ((GpuVbl)(args[i])).dptr);
			else
				throw new IllegalArgumentException (String.format
					("Kernel.%s(): Argument %d unknown type", name, i));
			}

		// Launch kernel.
		Cuda.cuLaunchKernel
			(gpu.ctx, func, gridDimX, gridDimY, gridDimZ,
			 blockDimX, blockDimY, blockDimZ, sharedMemBytes,
			 argc, argv, argp);

		return null;
		}

	/**
	 * Perform the proxy's <TT>setGridDim()</TT> method.
	 *
	 * @param  args  Method arguments.
	 */
	private Object setGridDim
		(Object[] args)
		{
		int x = ((Integer)(args[0])).intValue();
		int y = args.length >= 2 ? ((Integer)(args[1])).intValue() : 1;
		int z = args.length >= 3 ? ((Integer)(args[2])).intValue() : 1;
		if (x < 1)
			throw new IllegalArgumentException (String.format
				("Kernel.setGridDim(): gridDimX = %d illegal", x));
		if (y < 1)
			throw new IllegalArgumentException (String.format
				("Kernel.setGridDim(): gridDimY = %d illegal", y));
		if (z < 1)
			throw new IllegalArgumentException (String.format
				("Kernel.setGridDim(): gridDimZ = %d illegal", z));
		gridDimX = x;
		gridDimY = y;
		gridDimZ = z;
		return null;
		}

	/**
	 * Perform the proxy's <TT>setBlockDim()</TT> method.
	 *
	 * @param  args  Method arguments.
	 */
	private Object setBlockDim
		(Object[] args)
		{
		int x = ((Integer)(args[0])).intValue();
		int y = args.length >= 2 ? ((Integer)(args[1])).intValue() : 1;
		int z = args.length >= 3 ? ((Integer)(args[2])).intValue() : 1;
		if (x < 1)
			throw new IllegalArgumentException (String.format
				("Kernel.setBlockDim(): blockDimX = %d illegal", x));
		if (y < 1)
			throw new IllegalArgumentException (String.format
				("Kernel.setBlockDim(): blockDimY = %d illegal", y));
		if (z < 1)
			throw new IllegalArgumentException (String.format
				("Kernel.setBlockDim(): blockDimZ = %d illegal", z));
		blockDimX = x;
		blockDimY = y;
		blockDimZ = z;
		return null;
		}

	/**
	 * Perform the proxy's <TT>setCacheConfig()</TT> method.
	 *
	 * @param  args  Method arguments.
	 */
	private Object setCacheConfig
		(Object[] args)
		{
		Cuda.cuFuncSetCacheConfig
			(gpu.ctx, func, ((CacheConfig)(args[0])).value);
		return null;
		}

	/**
	 * Perform the proxy's <TT>setDynamicSharedMemorySize()</TT> method.
	 *
	 * @param  args  Method arguments.
	 */
	private Object setDynamicSharedMemorySize
		(Object[] args)
		{
		int n = ((Integer)(args[0])).intValue();
		if (n < 0)
			throw new IllegalArgumentException (String.format
				("Kernel.setDynamicSharedMemorySize(): bytesize = %d illegal",
				 n));
		sharedMemBytes = n;
		return null;
		}

	/**
	 * Perform the proxy's <TT>equals()</TT> method.
	 *
	 * @param  proxy  Proxy object.
	 * @param  args   Method arguments.
	 *
	 * @return  True if this proxy equals the argument, false otherwise.
	 */
	private Object doEquals
		(Object proxy,
		 Object[] args)
		{
		return new Boolean (proxy == args[0]);
		}

	/**
	 * Perform the proxy's <TT>hashCode()</TT> method.
	 *
	 * @param  proxy  Proxy object.
	 *
	 * @return  Hash code.
	 */
	private Object doHashCode
		(Object proxy)
		{
		return new Integer (System.identityHashCode (proxy));
		}

	/**
	 * Perform the proxy's <TT>toString()</TT> method.
	 *
	 * @return  String version.
	 */
	private Object doToString()
		{
		return String.format ("KernelHandler(%s)", name);
		}

	}
