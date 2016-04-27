//******************************************************************************
//
// File:    Module.java
// Package: edu.rit.gpu
// Unit:    Class edu.rit.gpu.Module
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
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.Proxy;

/**
 * Class Module encapsulates a module that contains GPU kernel functions and
 * data. To use a module:
 * <OL TYPE=1>
 * <P><LI>
 * Construct an instance of class Module by calling the {@link
 * Gpu#getModule(String) getModule()} method on a {@linkplain Gpu Gpu} object.
 * <P><LI>
 * Call methods on the Module object to perform module operations, such as:
 * <P><LI>
 * Get a proxy for a GPU kernel function &mdash;
 * <UL>
 * <LI>{@link #getKernel(Class) getKernel()}
 * </UL>
 * <P><LI>
 * Get a reference to a statically allocated variable or array in GPU memory
 * &mdash;
 * <UL>
 * <LI>{@link #getByteVbl(String) getByteVbl()},
 *     {@link #getByteArray(String,int) getByteArray()}
 * <LI>{@link #getShortVbl(String) getShortVbl()},
 *     {@link #getShortArray(String,int) getShortArray()}
 * <LI>{@link #getIntVbl(String) getIntVbl()},
 *     {@link #getIntArray(String,int) getIntArray()}
 * <LI>{@link #getLongVbl(String) getLongVbl()},
 *     {@link #getLongArray(String,int) getLongArray()}
 * <LI>{@link #getFloatVbl(String) getFloatVbl()},
 *     {@link #getFloatArray(String,int) getFloatArray()}
 * <LI>{@link #getDoubleVbl(String) getDoubleVbl()},
 *     {@link #getDoubleArray(String,int) getDoubleArray()}
 * <LI>{@link #getStructVbl(String,Class) getStructVbl()},
 *     {@link #getStructArray(String,Class,int) getStructArray()}
 * </UL>
 * </OL>
 * <P>
 * Study the source code of the classes in package {@linkplain
 * edu.rit.gpu.example edu.rit.pj2.example} to see examples of Parallel Java 2
 * GPU programs.
 * <P>
 * <I>Note:</I> Class Module is not multiple thread safe. It is assumed that
 * only one thread at a time will call methods on a Module object.
 *
 * @author  Alan Kaminsky
 * @version 03-Jun-2014
 */
public class Module
	{

// Hidden data members.

	Gpu gpu;   // Gpu object that created this
	long mod;  // Module handle

// Hidden constructors.

	/**
	 * Construct a new module. The module file is a resource stored in the Java
	 * class path.
	 *
	 * @param  gpu   Gpu object.
	 * @param  name  Module resource name relative to the Java class path.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	Module
		(Gpu gpu,
		 String name)
		throws IOException
		{
		this.gpu = gpu;
		this.mod = Cuda.cuModuleLoad
			(gpu.ctx, new FileResource (name) .filename());
		}

// Exported operations.

	/**
	 * Get a GPU kernel function from this module. The kernel function is
	 * specified by the given interface, which must extend interface {@linkplain
	 * Kernel Kernel}. The kernel function's name is that of the method
	 * declared in the given interface. The return value is a proxy object that
	 * implements the given interface. For further information, see interface
	 * {@linkplain Kernel Kernel}.
	 *
	 * @param  <T>   Kernel function interface.
	 * @param  intf  Kernel function interface's class object.
	 *
	 * @return  Proxy object for invoking kernel function.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>intf</TT> does not meet the
	 *     requirements for a kernel function interface.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public <T extends Kernel> T getKernel
		(Class<T> intf)
		{
		// Verify that the interface really is an interface.
		String intfName = intf.getSimpleName();
		if (! Modifier.isInterface (intf.getModifiers()))
			throw new IllegalArgumentException (String.format
				("Module.getKernel(): %s is not an interface", intfName));

		// Verify that the interface declares just one method.
		Method[] methods = intf.getDeclaredMethods();
		if (methods.length == 0)
			throw new IllegalArgumentException (String.format
				("Module.getKernel(): Interface %s declares no methods",
				 intfName));
		if (methods.length > 1)
			throw new IllegalArgumentException (String.format
				("Module.getKernel(): Interface %s declares more than one method",
				 intfName));

		// Verify that the method is public.
		Method method = methods[0];
		String name = method.getName();
		if (! Modifier.isPublic (method.getModifiers()))
			throw new IllegalArgumentException (String.format
				("Module.getKernel(): Method %s() is not public", name));

		// Verify that the method returns void.
		if (! method.getReturnType().equals (Void.TYPE))
			throw new IllegalArgumentException (String.format
				("Module.getKernel(): Method %s() does not return void",
				 name));

		// Verify that the method's argument types are supported.
		Class<?>[] argTypes = method.getParameterTypes();
		for (Class<?> argType : argTypes)
			if (! argType.isPrimitive() &&
					! GpuVbl.class.isAssignableFrom (argType))
				throw new IllegalArgumentException (String.format
					("Module.getKernel(): Parameter type %s not supported",
					 argType.getSimpleName()));

		// Get function handle.
		long func = Cuda.cuModuleGetFunction (gpu.ctx, mod, name);

		// Return a proxy object.
		return (T) Proxy.newProxyInstance
			(intf.getClassLoader(),
			 new Class[] { intf },
			 new KernelHandler (gpu, name, func, argTypes.length));
		}

	/**
	 * Create a new GPU byte variable mirroring the given global variable in
	 * this module. The variable in CPU memory, namely the {@link
	 * GpuByteVbl#item item} field, is initialized to zero. <I>The variable in
	 * GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 *
	 * @return  GPU byte variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public GpuByteVbl getByteVbl
		(String name)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getByteVbl(): name is null");
		return new GpuByteVbl
			(gpu, Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU short integer variable mirroring the given global
	 * variable in this module. The variable in CPU memory, namely the {@link
	 * GpuShortVbl#item item} field, is initialized to zero. <I>The variable in
	 * GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 *
	 * @return  GPU short integer variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public GpuShortVbl getShortVbl
		(String name)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getShortVbl(): name is null");
		return new GpuShortVbl
			(gpu, Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU integer variable mirroring the given global variable in
	 * this module. The variable in CPU memory, namely the {@link GpuIntVbl#item
	 * item} field, is initialized to zero. <I>The variable in GPU memory is not
	 * initialized.</I>
	 *
	 * @param  name  Global variable name.
	 *
	 * @return  GPU integer variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public GpuIntVbl getIntVbl
		(String name)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getIntVbl(): name is null");
		return new GpuIntVbl
			(gpu, Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU long integer variable mirroring the given global
	 * variable in this module. The variable in CPU memory, namely the {@link
	 * GpuLongVbl#item item} field, is initialized to zero. <I>The variable in
	 * GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 *
	 * @return  GPU long integer variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public GpuLongVbl getLongVbl
		(String name)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getLongVbl(): name is null");
		return new GpuLongVbl
			(gpu, Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU float variable mirroring the given global variable in
	 * this module. The variable in CPU memory, namely the {@link
	 * GpuFloatVbl#item item} field, is initialized to zero. <I>The variable in
	 * GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 *
	 * @return  GPU float variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public GpuFloatVbl getFloatVbl
		(String name)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getFloatVbl(): name is null");
		return new GpuFloatVbl
			(gpu, Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU double variable mirroring the given global variable in
	 * this module. The variable in CPU memory, namely the {@link
	 * GpuDoubleVbl#item item} field, is initialized to zero. <I>The variable in
	 * GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 *
	 * @return  GPU double variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public GpuDoubleVbl getDoubleVbl
		(String name)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getDoubleVbl(): name is null");
		return new GpuDoubleVbl
			(gpu, Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU byte array mirroring the given global variable in this
	 * module. The entire GPU array is mirrored in the CPU. The array in CPU
	 * memory, namely the {@link GpuByteArray#item item} field, is initialized
	 * to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU byte array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getByteArray(String,int,int)
	 */
	public GpuByteArray getByteArray
		(String name,
		 int len)
		{
		return getByteArray (name, len, len);
		}

	/**
	 * Create a new GPU byte array mirroring the given global variable in this
	 * module, of which the given portion is mirrored in the CPU. If
	 * <TT>cpulen</TT> = 0, then none of the array is mirrored in the CPU. The
	 * array in CPU memory, namely the {@link GpuByteArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name    Global variable name.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU byte array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getByteArray(String,int)
	 */
	public GpuByteArray getByteArray
		(String name,
		 int len,
		 int cpulen)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getByteArray(): name is null");
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Module.getByteArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Module.getByteArray(): cpulen = %d illegal", cpulen));
		return new GpuByteArray (gpu, len, cpulen,
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU short integer array mirroring the given global variable
	 * in this module. The entire GPU array is mirrored in the CPU. The array in
	 * CPU memory, namely the {@link GpuShortArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU short integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getShortArray(String,int,int)
	 */
	public GpuShortArray getShortArray
		(String name,
		 int len)
		{
		return getShortArray (name, len, len);
		}

	/**
	 * Create a new GPU short integer array mirroring the given global variable
	 * in this module, of which the given portion is mirrored in the CPU. If
	 * <TT>cpulen</TT> = 0, then none of the array is mirrored in the CPU. The
	 * array in CPU memory, namely the {@link GpuShortArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name    Global variable name.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU short integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getShortArray(String,int)
	 */
	public GpuShortArray getShortArray
		(String name,
		 int len,
		 int cpulen)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getShortArray(): name is null");
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Module.getShortArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Module.getShortArray(): cpulen = %d illegal", cpulen));
		return new GpuShortArray (gpu, len, cpulen,
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU integer array mirroring the given global variable in
	 * this module. The entire GPU array is mirrored in the CPU. The array in
	 * CPU memory, namely the {@link GpuIntArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getIntArray(String,int,int)
	 */
	public GpuIntArray getIntArray
		(String name,
		 int len)
		{
		return getIntArray (name, len, len);
		}

	/**
	 * Create a new GPU integer array mirroring the given global variable in
	 * this module, of which the given portion is mirrored in the CPU. If
	 * <TT>cpulen</TT> = 0, then none of the array is mirrored in the CPU. The
	 * array in CPU memory, namely the {@link GpuIntArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name    Global variable name.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getIntArray(String,int)
	 */
	public GpuIntArray getIntArray
		(String name,
		 int len,
		 int cpulen)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getIntArray(): name is null");
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Module.getIntArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Module.getIntArray(): cpulen = %d illegal", cpulen));
		return new GpuIntArray (gpu, len, cpulen,
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU long integer array mirroring the given global variable
	 * in this module. The entire GPU array is mirrored in the CPU. The array in
	 * CPU memory, namely the {@link GpuLongArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU long integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getLongArray(String,int,int)
	 */
	public GpuLongArray getLongArray
		(String name,
		 int len)
		{
		return getLongArray (name, len, len);
		}

	/**
	 * Create a new GPU long integer array mirroring the given global variable
	 * in this module, of which the given portion is mirrored in the CPU. If
	 * <TT>cpulen</TT> = 0, then none of the array is mirrored in the CPU. The
	 * array in CPU memory, namely the {@link GpuLongArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name    Global variable name.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU long integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getLongArray(String,int)
	 */
	public GpuLongArray getLongArray
		(String name,
		 int len,
		 int cpulen)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getLongArray(): name is null");
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Module.getLongArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Module.getLongArray(): cpulen = %d illegal", cpulen));
		return new GpuLongArray (gpu, len, cpulen,
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU float array mirroring the given global variable in this
	 * module. The entire GPU array is mirrored in the CPU. The array in CPU
	 * memory, namely the {@link GpuFloatArray#item item} field, is initialized
	 * to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU float array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getFloatArray(String,int,int)
	 */
	public GpuFloatArray getFloatArray
		(String name,
		 int len)
		{
		return getFloatArray (name, len, len);
		}

	/**
	 * Create a new GPU float array mirroring the given global variable in this
	 * module, of which the given portion is mirrored in the CPU. If
	 * <TT>cpulen</TT> = 0, then none of the array is mirrored in the CPU. The
	 * array in CPU memory, namely the {@link GpuFloatArray#item item} field, is
	 * initialized to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name    Global variable name.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU float array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getFloatArray(String,int)
	 */
	public GpuFloatArray getFloatArray
		(String name,
		 int len,
		 int cpulen)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getFloatArray(): name is null");
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Module.getFloatArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Module.getFloatArray(): cpulen = %d illegal", cpulen));
		return new GpuFloatArray (gpu, len, cpulen,
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU double array mirroring the given global variable in this
	 * module. The entire GPU array is mirrored in the CPU. The array in CPU
	 * memory, namely the {@link GpuDoubleArray#item item} field, is initialized
	 * to zeroes. <I>The array in GPU memory is not initialized.</I>
	 *
	 * @param  name  Global variable name.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU double array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getDoubleArray(String,int,int)
	 */
	public GpuDoubleArray getDoubleArray
		(String name,
		 int len)
		{
		return getDoubleArray (name, len, len);
		}

	/**
	 * Create a new GPU double array mirroring the given global variable in this
	 * module, of which the given portion is mirrored in the CPU. If
	 * <TT>cpulen</TT> = 0, then none of the array is mirrored in the CPU. The
	 * array in CPU memory, namely the {@link GpuDoubleArray#item item} field,
	 * is initialized to zeroes. <I>The array in GPU memory is not
	 * initialized.</I>
	 *
	 * @param  name    Global variable name.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU double array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getDoubleArray(String,int)
	 */
	public GpuDoubleArray getDoubleArray
		(String name,
		 int len,
		 int cpulen)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getDoubleArray(): name is null");
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Module.getDoubleArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Module.getDoubleArray(): cpulen = %d illegal", cpulen));
		return new GpuDoubleArray (gpu, len, cpulen,
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name));
		}

	/**
	 * Create a new GPU struct variable mirroring the given global variable in
	 * this module. The variable in CPU memory, namely the {@link
	 * GpuStructVbl#item item} field, is initialized to null. <I>The variable in
	 * GPU memory is not initialized.</I>
	 *
	 * @param  <T>   Java data type. Class <TT>T</TT> must extend class
	 *               {@linkplain Struct Struct}.
	 * @param  name  Global variable name.
	 * @param  type  Class object for Java data type.
	 *
	 * @return  GPU struct variable.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> or <TT>type</TT> is
	 *     null.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public <T extends Struct> GpuStructVbl<T> getStructVbl
		(String name,
		 Class<T> type)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getStructVbl(): name is null");
		return new GpuStructVbl (gpu, 
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name), type);
		}

	/**
	 * Create a new GPU struct array mirroring the given global variable in this
	 * module. The entire GPU array is mirrored in the CPU. The array elements
	 * in CPU memory, namely the {@link GpuStructArray#item item} field's
	 * elements, are initialized to null. <I>The array elements in GPU memory
	 * are not initialized.</I>
	 *
	 * @param  <T>   Java data type. Class <TT>T</TT> must extend class
	 *               {@linkplain Struct Struct}.
	 * @param  name  Global variable name.
	 * @param  type  Class object for Java data type.
	 * @param  len   Number of array elements (&ge; 1).
	 *
	 * @return  GPU struct array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> or <TT>type</TT> is
	 *     null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getStructArray(String,Class,int,int)
	 */
	public <T extends Struct> GpuStructArray<T> getStructArray
		(String name,
		 Class<T> type,
		 int len)
		{
		return getStructArray (name, type, len, len);
		}

	/**
	 * Create a new GPU struct array mirroring the given global variable in this
	 * module, of which the given portion is mirrored in the CPU. If
	 * <TT>cpulen</TT> = 0, then none of the array is mirrored in the CPU. The
	 * array elements in CPU memory, namely the {@link GpuStructArray#item item}
	 * field's elements, are initialized to null. <I>The array elements in GPU
	 * memory are not initialized.</I>
	 *
	 * @param  <T>     Java data type. Class <TT>T</TT> must extend class
	 *                 {@linkplain Struct Struct}.
	 * @param  name    Global variable name.
	 * @param  type    Class object for Java data type.
	 * @param  len     Number of array elements (&ge; 1).
	 * @param  cpulen  Number of array elements to mirror in the CPU.
	 *
	 * @return  GPU struct array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>name</TT> or <TT>type</TT> is
	 *     null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 1, <TT>cpulen</TT>
	 *     &lt; 0, or <TT>cpulen</TT> &gt; <TT>len</TT>.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 *
	 * @see  #getStructArray(String,Class,int)
	 */
	public <T extends Struct> GpuStructArray<T> getStructArray
		(String name,
		 Class<T> type,
		 int len,
		 int cpulen)
		{
		if (name == null)
			throw new NullPointerException
				("Module.getStructArray(): name is null");
		if (len < 1)
			throw new IllegalArgumentException (String.format
				("Module.getStructArray(): len = %d illegal", len));
		if (0 > cpulen || cpulen > len)
			throw new IllegalArgumentException (String.format
				("Module.getStructArray(): cpulen = %d illegal", cpulen));
		return new GpuStructArray (gpu, len, cpulen,
			Cuda.cuModuleGetGlobal (gpu.ctx, mod, name), type);
		}

// Hidden operations.

	/**
	 * Finalize this module.
	 */
	protected synchronized void finalize()
		{
		Cuda.cuModuleUnload (gpu.ctx, mod);
		}

	}
