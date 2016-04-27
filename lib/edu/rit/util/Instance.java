//******************************************************************************
//
// File:    Instance.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Instance
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

package edu.rit.util;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

/**
 * Class Instance provides static methods for creating instances of classes.
 *
 * @author  Alan Kaminsky
 * @version 13-Feb-2016
 */
public class Instance
	{

// Prevent construction.

	private Instance()
		{
		}

// Exported operations.

	/**
	 * Create a new instance of a class as specified by the given string.
	 * Calling this method is equivalent to calling
	 * <TT>newInstance(s,false)</TT>. See the {@link
	 * #newInstance(String,boolean) newInstance(String,boolean)} method for
	 * further information.
	 *
	 * @param  s  Constructor expression string.
	 *
	 * @return  New instance.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> does not obey the required
	 *     syntax.
	 * @exception  ClassNotFoundException
	 *     Thrown if the given class cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if a suitable constructor cannot be found in the given class.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the given constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the given constructor throws an exception.
	 */
	public static Object newInstance
		(String s)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return newInstance (s, false);
		}

	/**
	 * Create a new instance of a class as specified by the given string. The
	 * string must consist of a fully-qualified class name, a left parenthesis,
	 * zero or more comma-separated arguments, and a right parenthesis. No
	 * whitespace is allowed.
	 * <P>
	 * The <TT>newInstance()</TT> method deduces the data type and value of each
	 * argument provided in the constructor expression according to the
	 * following patterns:
	 * <UL>
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; <TT>0b</TT> or <TT>0B</TT>; one or
	 * more digits <TT>0</TT> through <TT>1</TT> &mdash; binary (base-2) value
	 * of type <TT>int</TT>.
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; <TT>0x</TT> or <TT>0X</TT>; one or
	 * more digits <TT>0</TT> through <TT>9</TT>, <TT>a</TT> through <TT>f</TT>,
	 * or <TT>A</TT> through <TT>F</TT> &mdash; hexadecimal (base-16) value of
	 * type <TT>int</TT>.
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; <TT>0</TT>; zero or more digits
	 * <TT>0</TT> through <TT>7</TT> &mdash; octal (base-8) value of type
	 * <TT>int</TT>.
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; a digit <TT>1</TT> through <TT>9</TT>;
	 * zero or more digits <TT>0</TT> through <TT>9</TT> &mdash; decimal value
	 * of type <TT>int</TT>.
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; <TT>0b</TT> or <TT>0B</TT>; one or
	 * more digits <TT>0</TT> through <TT>1</TT>; <TT>l</TT> or <TT>L</TT>
	 * &mdash; binary (base-2) value of type <TT>long</TT>.
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; <TT>0x</TT> or <TT>0X</TT>; one or
	 * more digits <TT>0</TT> through <TT>9</TT>, <TT>a</TT> through <TT>f</TT>,
	 * or <TT>A</TT> through <TT>F</TT>; <TT>l</TT> or <TT>L</TT> &mdash;
	 * hexadecimal (base-16) value of type <TT>long</TT>.
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; <TT>0</TT>; zero or more digits
	 * <TT>0</TT> through <TT>7</TT>; <TT>l</TT> or <TT>L</TT> &mdash; octal
	 * (base-8) value of type <TT>long</TT>.
	 * <P><LI>
	 * Optional <TT>+</TT> or <TT>-</TT>; a digit <TT>1</TT> through <TT>9</TT>;
	 * zero or more digits <TT>0</TT> through <TT>9</TT>; <TT>l</TT> or
	 * <TT>L</TT> &mdash; decimal value of type <TT>long</TT>.
	 * <P><LI>
	 * Otherwise &mdash; type String.
	 * </UL>
	 * <P>
	 * If a decimal argument is outside the range of type <TT>int</TT> or type
	 * <TT>long</TT>, the deduced value is not specified. If a binary, octal, or
	 * hexadecimal argument is larger than 32 bits for type <TT>int</TT> or 64
	 * bits for type <TT>long</TT>, the deduced value is not specified.
	 * <P>
	 * The <TT>newInstance()</TT> method attempts to find a constructor for the
	 * given class as follows, where <I>N</I> is the number of provided
	 * arguments:
	 * <UL>
	 * <P><LI>
	 * If <I>N</I> = 0, use a no-argument constructor.
	 * <P><LI>
	 * Else if there is a constructor with <I>N</I> arguments whose types match
	 * those provided, use that constructor.
	 * <P><LI>
	 * Else if all provided arguments are of type <TT>int</TT> and there is a
	 * constructor with one argument of type <TT>int[]</TT>, use that
	 * constructor.
	 * <P><LI>
	 * Else if all provided arguments are of type <TT>long</TT> and there is a
	 * constructor with one argument of type <TT>long[]</TT>, use that
	 * constructor.
	 * <P><LI>
	 * Else if there is a constructor with <I>N</I> arguments of type
	 * <TT>String</TT>, use that constructor; all provided arguments are passed
	 * as strings.
	 * <P><LI>
	 * Else if there is a constructor with one argument of type
	 * <TT>String[]</TT>, use that constructor; all provided arguments are
	 * passed as strings.
	 * <P><LI>
	 * Else throw a NoSuchMethodException.
	 * </UL>
	 * <P>
	 * The <TT>newInstance()</TT> method invokes the chosen constructor, passing
	 * in the provided argument values, and returns a reference to the
	 * newly-created instance.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing the instance. This means the object's class
	 * and/or the class's pertinent constructor need not be public, and a new
	 * instance will still be constructed. However, this also requires that
	 * either (a) a security manager is not installed, or (b) the security
	 * manager allows ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 * <P>
	 * <I>Note:</I> To find the given class, the calling thread's context class
	 * loader is used.
	 *
	 * @param  s  Constructor expression string.
	 * @param  disableAccessChecks  True to disable access checks, false to
	 *                              perform access checks.
	 *
	 * @return  New instance.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> does not obey the required
	 *     syntax.
	 * @exception  ClassNotFoundException
	 *     Thrown if the given class cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if a suitable constructor cannot be found in the given class.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static Object newInstance
		(String s,
		 boolean disableAccessChecks)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		// Verify no whitespace.
		if (s.matches (".*\\s.*")) syntaxError ("No whitespace allowed");

		// Break constructor expression into class name and argument list.
		int a = s.indexOf ('(');
		if (a == -1) syntaxError ("Missing (");
		if (a == 0) syntaxError ("Missing class name");
		if (s.charAt (s.length() - 1) != ')') syntaxError ("Missing )");
		String className = s.substring (0, a);
		String argList = s.substring (a + 1, s.length() - 1);

		// Get class.
		Class<?> theClass = Class.forName
			(className,
			 true,
			 Thread.currentThread().getContextClassLoader());

		// Break argument list into comma-separated arguments.
		String[] args =
			argList.length() == 0 ? new String [0] : argList.split (",");
		int N = args.length;

		// Attempt to parse arguments as int or long values.
		Class<?>[] types = new Class<?> [N];
		Integer[] intValues = new Integer [N];
		Long[] longValues = new Long [N];
		boolean allTypeInt = true;
		boolean allTypeLong = true;
		boolean minus;
		String arg;
		for (int i = 0; i < N; ++ i)
			{
			arg = new String (args[i]);
			minus = false;

			// Detect and remove initial + or -.
			if (arg.startsWith ("+"))
				{
				arg = arg.substring (1);
				}
			else if (arg.startsWith ("-"))
				{
				minus = true;
				arg = arg.substring (1);
				}

			// Detect and remove final l or L.
			if (arg.endsWith ("l") || arg.endsWith ("L"))
				{
				types[i] = Long.TYPE;
				allTypeInt = false;
				arg = arg.substring (0, arg.length() - 1);
				}
			else
				{
				types[i] = Integer.TYPE;
				allTypeLong = false;
				}

			// Detect a binary value.
			if (arg.startsWith ("0b") || arg.startsWith ("0B"))
				{
				arg = arg.substring (2);
				try
					{
					longValues[i] = parseBinary (arg, minus);
					}
				catch (NumberFormatException exc)
					{
					types[i] = String.class;
					allTypeInt = allTypeLong = false;
					}
				}

			// Detect a hexadecimal value.
			else if (arg.startsWith ("0x") || arg.startsWith ("0X"))
				{
				arg = arg.substring (2);
				try
					{
					longValues[i] = parseHex (arg, minus);
					}
				catch (NumberFormatException exc)
					{
					types[i] = String.class;
					allTypeInt = allTypeLong = false;
					}
				}

			// Detect an octal value.
			else if (arg.startsWith ("0"))
				{
				try
					{
					longValues[i] = parseOctal (arg, minus);
					}
				catch (NumberFormatException exc)
					{
					types[i] = String.class;
					allTypeInt = allTypeLong = false;
					}
				}

			// Detect a decimal value.
			else if (arg.charAt(0) >= '1' && arg.charAt(0) <= '9')
				{
				try
					{
					longValues[i] = parseDecimal (arg, minus);
					}
				catch (NumberFormatException exc)
					{
					types[i] = String.class;
					allTypeInt = allTypeLong = false;
					}
				}

			// Otherwise detect a string value.
			else
				{
				types[i] = String.class;
				longValues[i] = new Long (0L);
				allTypeInt = allTypeLong = false;
				}

			// Compute int value.
			intValues[i] = new Integer (longValues[i].intValue());
			}

//System.out.printf ("Instance.newInstance(\"%s\",%b): N = %d%n", s, disableAccessChecks, N);
//for (int i = 0; i < N; ++ i)
//	System.out.printf ("%d\t%s\t\"%s\"\t%s\t%sL%n", i, types[i], args[i], intValues[i], longValues[i]);

		// Try a constructor with the N deduced argument types.
		Constructor<?> ctor = null;
		try
			{
			ctor = theClass.getDeclaredConstructor (types);
			ctor.setAccessible (disableAccessChecks);
			Object[] ctorArgs = new Object [N];
			for (int i = 0; i < N; ++ i)
				if (types[i] == Integer.TYPE)
					ctorArgs[i] = intValues[i];
				else if (types[i] == Long.TYPE)
					ctorArgs[i] = longValues[i];
				else
					ctorArgs[i] = args[i];
			return ctor.newInstance (ctorArgs);
			}
		catch (NoSuchMethodException exc)
			{
			}

		// Try a constructor with one int array argument.
		if (allTypeInt)
			{
			try
				{
				ctor = theClass.getDeclaredConstructor (int[].class);
				ctor.setAccessible (disableAccessChecks);
				return ctor.newInstance ((Object) intValues);
				}
			catch (NoSuchMethodException exc)
				{
				}
			}

		// Try a constructor with one long array argument.
		if (allTypeLong)
			{
			try
				{
				ctor = theClass.getDeclaredConstructor (long[].class);
				ctor.setAccessible (disableAccessChecks);
				return ctor.newInstance ((Object) longValues);
				}
			catch (NoSuchMethodException exc)
				{
				}
			}

		// Try a constructor with N string arguments.
		try
			{
			for (int i = 0; i < N; ++ i)
				types[i] = String.class;
			ctor = theClass.getDeclaredConstructor (types);
			ctor.setAccessible (disableAccessChecks);
			return ctor.newInstance ((Object[]) args);
			}
		catch (NoSuchMethodException exc)
			{
			}

		// Try a constructor with one string array argument.
		try
			{
			ctor = theClass.getDeclaredConstructor (String[].class);
			ctor.setAccessible (disableAccessChecks);
			return ctor.newInstance ((Object) args);
			}
		catch (NoSuchMethodException exc)
			{
			}

		// Couldn't find a suitable constructor.
		throw new NoSuchMethodException (String.format
			("Instance.newInstance(\"%s\"): Cannot find suitable constructor",
			 s));
		}

	/**
	 * Create a new instance of the class with the given name using the class's
	 * default constructor. Calling this method is equivalent to calling
	 * <TT>newDefaultInstance(className,false)</TT>. See the {@link
	 * #newDefaultInstance(String,boolean) newDefaultInstance(String,boolean)}
	 * method for further information.
	 * <P>
	 * <I>Note:</I> To find the class with the given name, the calling thread's
	 * context class loader is used.
	 *
	 * @param  className  Class name.
	 *
	 * @return  New instance.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the class with the given name cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static Object newDefaultInstance
		(String className)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return newDefaultInstance (className, false);
		}

	/**
	 * Create a new instance of the class with the given name using the class's
	 * default constructor.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing the new instance. This means the object's
	 * class and/or the class's default constructor need not be public, and a
	 * new instance will still be constructed. However, this also requires that
	 * either (a) a security manager is not installed, or (b) the security
	 * manager allows ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 * <P>
	 * <I>Note:</I> To find the class with the given name, the calling thread's
	 * context class loader is used.
	 *
	 * @param  className  Class name.
	 * @param  disableAccessChecks
	 *     True to disable access checks, false to perform access checks.
	 *
	 * @return  New instance.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the class with the given name cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static Object newDefaultInstance
		(String className,
		 boolean disableAccessChecks)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		Class<?> theClass = Class.forName
			(className,
			 true,
			 Thread.currentThread().getContextClassLoader());
		return newDefaultInstance (theClass, disableAccessChecks);
		}

	/**
	 * Create a new instance of the given class using the class's default
	 * constructor. Calling this method is equivalent to calling
	 * <TT>newDefaultInstance(c,false)</TT>. See the {@link
	 * #newDefaultInstance(Class,boolean) newDefaultInstance(Class,boolean)}
	 * method for further information.
	 *
	 * @param  <T>  Class's data type.
	 * @param  c    Class.
	 *
	 * @return  New instance.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static <T> T newDefaultInstance
		(Class<T> c)
		throws
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return newDefaultInstance (c, false);
		}

	/**
	 * Create a new instance of the given class using the class's default
	 * constructor.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing the new instance. This means the object's
	 * class and/or the class's default constructor need not be public, and a
	 * new instance will still be constructed. However, this also requires that
	 * either (a) a security manager is not installed, or (b) the security
	 * manager allows ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 *
	 * @param  <T>  Class's data type.
	 * @param  c    Class.
	 * @param  disableAccessChecks
	 *     True to disable access checks, false to perform access checks.
	 *
	 * @return  New instance.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static <T> T newDefaultInstance
		(Class<T> c,
		 boolean disableAccessChecks)
		throws
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return
			((Constructor<T>)(getDefaultConstructor (c, disableAccessChecks)))
				.newInstance();
		}

	/**
	 * Get the given class's default constructor. Calling this method is
	 * equivalent to calling <TT>getDefaultConstructor(c,false)</TT>. See the
	 * {@link #getDefaultConstructor(Class,boolean)
	 * getDefaultConstructor(Class,boolean)} method for further information.
	 *
	 * @param  c  Class.
	 *
	 * @return  Default (no-argument) constructor for the class.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the class does not have a default constructor.
	 */
	public static Constructor<?> getDefaultConstructor
		(Class<?> c)
		throws NoSuchMethodException
		{
		return getDefaultConstructor (c, false);
		}

	/**
	 * Get the given class's default constructor.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing an instance using the returned constructor.
	 * This means the object's class and/or the class's default constructor need
	 * not be public, and a new instance will still be constructed. However,
	 * this also requires that either (a) a security manager is not installed,
	 * or (b) the security manager allows
	 * ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 *
	 * @param  c  Class.
	 * @param  disableAccessChecks  True to disable access checks, false to
	 *                              perform access checks.
	 *
	 * @return  Default (no-argument) constructor for the class.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the class does not have a default constructor.
	 */
	public static Constructor<?> getDefaultConstructor
		(Class<?> c,
		 boolean disableAccessChecks)
		throws NoSuchMethodException
		{
		for (Constructor<?> ctor : c.getDeclaredConstructors())
			if (ctor.getParameterTypes().length == 0)
				{
				ctor.setAccessible (disableAccessChecks);
				return ctor;
				}
		throw new NoSuchMethodException (String.format
			("No such method: %s.<init>()", c.getName()));
		}

// Hidden operations.

	/**
	 * Throw an exception indicating a syntax error.
	 */
	private static void syntaxError
		(String msg)
		{
		throw new IllegalArgumentException (String.format
			("Instance.newInstance(): Syntax error: %s", msg));
		}

	/**
	 * Parse the given string as a binary integer. If <TT>minus</TT> is true,
	 * negate it.
	 */
	private static Long parseBinary
		(String s,
		 boolean minus)
		{
		int n = s.length();
		if (n == 0) throw new NumberFormatException();
		long v = 0L;
		char c;
		for (int i = 0; i < n; ++ i)
			{
			c = s.charAt (i);
			if ('0' <= c && c <= '1')
				v = (v << 1) | (c - '0');
			else
				throw new NumberFormatException();
			}
		if (minus) v = -v;
		return new Long (v);
		}

	/**
	 * Parse the given string as an octal integer. If <TT>minus</TT> is true,
	 * negate it.
	 */
	private static Long parseOctal
		(String s,
		 boolean minus)
		{
		int n = s.length();
		if (n == 0) throw new NumberFormatException();
		long v = 0L;
		char c;
		for (int i = 0; i < n; ++ i)
			{
			c = s.charAt (i);
			if ('0' <= c && c <= '7')
				v = (v << 3) | (c - '0');
			else
				throw new NumberFormatException();
			}
		if (minus) v = -v;
		return new Long (v);
		}

	/**
	 * Parse the given string as a decimal integer. If <TT>minus</TT> is true,
	 * negate it.
	 */
	private static Long parseDecimal
		(String s,
		 boolean minus)
		{
		int n = s.length();
		if (n == 0) throw new NumberFormatException();
		long v = 0L;
		char c;
		for (int i = 0; i < n; ++ i)
			{
			c = s.charAt (i);
			if ('0' <= c && c <= '9')
				v = (v * 10) + (c - '0');
			else
				throw new NumberFormatException();
			}
		if (minus) v = -v;
		return new Long (v);
		}

	/**
	 * Parse the given string as a hexadecimal integer. If <TT>minus</TT> is
	 * true, negate it.
	 */
	private static Long parseHex
		(String s,
		 boolean minus)
		{
		int n = s.length();
		if (n == 0) throw new NumberFormatException();
		long v = 0L;
		char c;
		for (int i = 0; i < n; ++ i)
			{
			c = s.charAt (i);
			if ('0' <= c && c <= '9')
				v = (v << 4) | (c - '0');
			else if ('A' <= c && c <= 'F')
				v = (v << 4) | (c - 'A' + 10);
			else if ('a' <= c && c <= 'f')
				v = (v << 4) | (c - 'a' + 10);
			else
				throw new NumberFormatException();
			}
		if (minus) v = -v;
		return new Long (v);
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		System.out.println (Instance.newInstance (args[0]));
//		}

	}
