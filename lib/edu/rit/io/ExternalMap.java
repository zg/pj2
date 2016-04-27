//******************************************************************************
//
// File:    ExternalMap.java
// Package: edu.rit.io
// Unit:    Class edu.rit.io.ExternalMap
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

package edu.rit.io;

import edu.rit.util.Action;
import edu.rit.util.Map;
import edu.rit.util.Pair;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

/**
 * Class ExternalMap provides a mapping from keys to values, where the keys and
 * values are stored in an external file. The key must be an object suitable for
 * use in a hashed data structure. The value may be any object. Keys and values
 * must not be null. Keys and values must be streamable or serializable.
 * <P>
 * When an external map is constructed, the keys are read from the file, but the
 * values are not. Subsequently, the {@link #get(Object) get(key)} method
 * retrieves the value associated with the given key; the value is read from the
 * file the first time it is retrieved. The {@link #add(Object,Object)
 * add(key,value)} method adds a new (key, value) mapping, which is written to
 * the file.
 * <P>
 * <I>Note:</I> Class ExternalMap is multiple thread safe.
 *
 * @param  <K>  Key data type.
 * @param  <V>  Value data type.
 *
 * @author  Alan Kaminsky
 * @version 28-Apr-2015
 */
public class ExternalMap<K,V>
	{

// FILE FORMAT
// The file consists of a series of records. Each record consists of:
// - Key length (KL), 4 bytes
// - Value length (VL), 4 bytes
// - Byte array containing streamed key, KL bytes
// - Byte array containing streamed value, VL bytes

// Hidden helper classes.

	/**
	 * Class Info is a record of information about one value in the file.
	 */
	private static class Info<V>
		{
		public long offset;
		public int length;
		public V value;
		public Info
			(long offset,
			 int length,
			 V value)
			{
			this.offset = offset;
			this.length = length;
			this.value = value;
			}
		}

// Hidden data members.

	private File file;
	private RandomAccessFile raf;
	private Map<K,Info<V>> map = new Map<K,Info<V>>();

// Exported constructors.

	/**
	 * Construct a new external map for the given file. The file must be
	 * readable and writeable.
	 *
	 * @param  file  File.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public ExternalMap
		(File file)
		throws IOException
		{
		this.file = file;
		this.raf = new RandomAccessFile (file, "rwd");
		readKeys();
		}

	/**
	 * Construct a new external map for the given file name. The file must be
	 * readable and writeable.
	 *
	 * @param  filename  File name.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public ExternalMap
		(String filename)
		throws IOException
		{
		this (new File (filename));
		}

	/**
	 * Read the keys from the file and populate the map.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private synchronized void readKeys()
		throws IOException
		{
		long fileLength = raf.length();
		byte[] keyBytes = new byte [0];
		long offset = 0L;
		while (offset < fileLength)
			{
			raf.seek (offset);
			int keyLength = raf.readInt();
			int valueLength = raf.readInt();
			if (keyBytes.length < keyLength)
				keyBytes = new byte [keyLength];
			raf.readFully (keyBytes, 0, keyLength);
			ByteArrayInputStream bais =
				new ByteArrayInputStream (keyBytes, 0, keyLength);
			InStream in = new InStream (bais);
			K key = (K) in.readObject();
			map.put (key, new Info<V> (offset + 8 + keyLength,
				valueLength, null));
			offset = offset + 8 + keyLength + valueLength;
			}
		}

// Exported operations.

	/**
	 * Returns the file in which this external map is stored.
	 *
	 * @return  File.
	 */
	public File file()
		{
		return file;
		}

	/**
	 * Determine if this external map contains the given key.
	 *
	 * @param  key  Key.
	 *
	 * @return  True if this external map contains <TT>key</TT>, false
	 *          otherwise.
	 */
	public synchronized boolean contains
		(K key)
		{
		return map.contains (key);
		}

	/**
	 * Get the value associated with the given key in this external map. If this
	 * external map does not contain <TT>key</TT>, null is returned.
	 *
	 * @param  key  Key.
	 *
	 * @return  Value associated with <TT>key</TT>, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized V get
		(K key)
		throws IOException
		{
		Info<V> info = map.get (key);
		if (info == null) return null;
		if (info.value == null)
			{
			byte[] valueBytes = new byte [info.length];
			raf.seek (info.offset);
			raf.readFully (valueBytes);
			ByteArrayInputStream bais = new ByteArrayInputStream (valueBytes);
			InStream in = new InStream (bais);
			info.value = (V) in.readObject();
			}
		return info.value;
		}

	/**
	 * Add the given key and associated value to this external map. The key and
	 * the value must not be null. The key must not already exist in this
	 * external map. The key and the value must be streamable or serializable.
	 *
	 * @param  key    Key.
	 * @param  value  Value.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null. Thrown if
	 *     <TT>value</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>key</TT> already exists in this
	 *     external map.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void add
		(K key,
		 V value)
		throws IOException
		{
		if (key == null)
			throw new NullPointerException
				("ExternalMap.add(): key is null");
		if (value == null)
			throw new NullPointerException
				("ExternalMap.add(): value is null");
		if (map.contains (key))
			throw new IllegalArgumentException
				("ExternalMap.add(): key already exists");

		ByteArrayOutputStream baos;
		OutStream out;

		baos = new ByteArrayOutputStream();
		out = new OutStream (baos);
		out.writeObject (key);
		out.close();
		byte[] keyBytes = baos.toByteArray();

		baos = new ByteArrayOutputStream();
		out = new OutStream (baos);
		out.writeObject (value);
		out.close();
		byte[] valueBytes = baos.toByteArray();

		long offset = raf.length();
		raf.seek (offset);
		raf.writeInt (keyBytes.length);
		raf.writeInt (valueBytes.length);
		raf.write (keyBytes);
		raf.write (valueBytes);

		map.put (key, new Info<V> (offset + 8 + keyBytes.length,
			valueBytes.length, value));
		}

	/**
	 * Close this external map.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public synchronized void close()
		throws IOException
		{
		raf.close();
		raf = null;
		map = null;
		}

	/**
	 * Finalize this external map.
	 */
	protected void finalize()
		throws Throwable
		{
		close();
		}

	/**
	 * Main program to print the keys in an external map file.
	 * <P>
	 * Usage: <TT>java edu.rit.io.ExternalMap <I>file</I></TT>
	 */
	public static void main
		(String[] args)
		throws Exception
		{
		if (args.length != 1) usage();
		ExternalMap<Object,Object> emap =
			new ExternalMap<Object,Object> (args[0]);
		emap.map.forEachItemDo (new Action<Pair<Object,Info<Object>>>()
			{
			public void run (Pair<Object,Info<Object>> pair)
				{
				Object key = pair.key();
				Info<Object> value = pair.value();
				System.out.printf ("%s (offset=%d, length=%d)%n",
					key, value.offset, value.length);
				}
			});
		}

	/**
	 * Print a usage message and exit.
	 */
	private static void usage()
		{
		System.err.println ("Usage: java edu.rit.io.ExternalMap <file>");
		System.exit (1);
		}

	}
