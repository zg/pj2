//******************************************************************************
//
// File:    JarClassLoader.java
// Package: edu.rit.pj2
// Unit:    Class edu.rit.pj2.JarClassLoader
//
// This Java source file is copyright (C) 2015 by Alan Kaminsky. All rights
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

package edu.rit.pj2;

import edu.rit.util.Map;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.jar.JarEntry;
import java.util.jar.JarInputStream;

/**
 * Class JarClassLoader provides a class loader that obtains class files from a
 * byte array holding the contents of a Java archive (JAR) file.
 * <P>
 * Class JarClassLoader supports loading of classes. Class JarClassLoader
 * supports loading of resources via the {@link #getResourceAsStream(String)
 * getResourceAsStream()} method. The other methods for loading resources are
 * not supported.
 * <P>
 * When running a {@linkplain Job Job} or a {@linkplain Task Task}, the {@link
 * pj2 pj2} launcher program installs a JAR class loader in the process if the
 * <TT>jar=<I>file</I></TT> option is specified. A JAR class loader is also
 * installed in the process running a job's task if the <TT>jar=<I>file</I></TT>
 * option is specified.
 *
 * @author  Alan Kaminsky
 * @version 10-Jan-2015
 */
public class JarClassLoader
	extends ClassLoader
	{

// Hidden data members.

	// Map from class or resource name to contents.
	private Map<String,byte[]> contentMap = new Map<String,byte[]>();

// Exported constructors.

	/**
	 * Construct a new JAR class loader. The parent class loader is the system
	 * class loader.
	 *
	 * @param  jar  Byte array with the contents of the JAR file.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>jar</TT> is null.
	 * @exception  IOException
	 *     Thrown if the JAR file's contents could not be extracted from the
	 *     <TT>jar</TT> byte array.
	 */
	public JarClassLoader
		(byte[] jar)
		throws IOException
		{
		super();
		readJarContents (jar);
		}

	/**
	 * Construct a new JAR class loader with the given parent class loader.
	 *
	 * @param  parent  Parent class loader.
	 * @param  jar     Byte array with the contents of the JAR file.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>jar</TT> is null.
	 * @exception  IOException
	 *     Thrown if the JAR file's contents could not be extracted from the
	 *     <TT>jar</TT> byte array.
	 */
	public JarClassLoader
		(ClassLoader parent, 
		 byte[] jar)
		throws IOException
		{
		super (parent);
		readJarContents (jar);
		}

	private void readJarContents
		(byte[] jar)
		throws IOException
		{
		if (jar == null)
			throw new NullPointerException
				("JarClassLoader(): jar is null");
		JarInputStream in = new JarInputStream
			(new ByteArrayInputStream (jar));
		JarEntry jarEntry;
		String name;
		ByteArrayOutputStream out;
		byte[] buf = new byte [1024];
		int n;
		while ((jarEntry = in.getNextJarEntry()) != null)
			{
			name = jarEntry.getName();
			out = new ByteArrayOutputStream();
			while ((n = in.read (buf, 0, 1024)) != -1)
				out.write (buf, 0, n);
			contentMap.put (name, out.toByteArray());
			}
		}

// Hidden operations.

	/**
	 * Find the class with the given name.
	 *
	 * @param  name  Class name.
	 *
	 * @return  Class object.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the class could not be found.
	 */
	protected Class<?> findClass
		(String name)
		throws ClassNotFoundException
		{
		String contentName = name.replace ('.', '/') + ".class";
		byte[] content = contentMap.get (contentName);
		if (content == null)
			throw new ClassNotFoundException (String.format
				("JarClassLoader.findClass(): Class %s not found", name));
		return defineClass (name, content, 0, content.length);
		}

	/**
	 * Get an input stream for reading the given resource.
	 *
	 * @param  name  Resource name.
	 *
	 * @return  Input stream, or null if the resource could not be found.
	 */
	public InputStream getResourceAsStream
		(String name)
		{
		byte[] content = contentMap.get (name);
		return content == null ? null : new ByteArrayInputStream (content);
		}

	}
