//******************************************************************************
//
// File:    InStream.java
// Package: edu.rit.io
// Unit:    Class edu.rit.io.InStream
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

import edu.rit.util.Instance;
import edu.rit.util.Map;
import java.io.ByteArrayInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;

/**
 * Class InStream provides an object that reads primitive data types, strings,
 * objects, and arrays in binary form from an underlying input stream. It reads
 * a byte stream that was written using class {@linkplain OutStream} as
 * described in more detail in the individual methods.
 * <P>
 * The methods for reading integer types and strings read a variable number of
 * bytes, as described in more detail in the individual methods of class
 * {@linkplain OutStream}. This can save space in the byte stream if small
 * integer values are written more frequently than large integer values.
 * <P>
 * Several methods for reading objects are provided:
 * <UL>
 * <P><LI>
 * {@link #readFields(Streamable) readFields()} reads just the fields of a
 * {@linkplain Streamable} object. Use this method when the reader knows the
 * class of the object ahead of time and has already created an instance of that
 * class.
 * <P><LI>
 * {@link #readObject() readObject()} reads both the class name and the fields
 * of a {@linkplain Streamable} or {@linkplain java.io.Serializable
 * Serializable} object. Use this method when the reader does not know the class
 * of the object ahead of time.
 * <P><LI>
 * {@link #readReference() readReference()} reads the class name and the fields
 * of a {@linkplain Streamable} or {@linkplain java.io.Serializable
 * Serializable} object, and also keeps a reference to the object. If the same
 * object is read by {@link #readReference() readReference()} again, just a
 * reference to the previous object is returned. Use this method when the reader
 * needs to end up with multiple references to the same object.
 * </UL>
 * <P>
 * Classes {@linkplain OutStream} and InStream provide a <I>lightweight</I>
 * object serialization capability&mdash;one that generates fewer bytes than
 * Java Object Serialization in package java.io. However, the programmer is
 * responsible for writing all the serialization and deserialization code in
 * each class that implements interface {@linkplain Streamable}.
 * <P>
 * To support interoperability with classes designed to work with Java Object
 * Serialization, {@link #readObject() readObject()} and {@link #readReference()
 * readReference()} will work both on {@linkplain Streamable} objects and on
 * {@linkplain java.io.Serializable Serializable} objects. In the latter case, a
 * byte array is read, and the byte array is converted to an object using Java
 * Object Serialization.
 * <P>
 * Methods for reading arrays of primitive types, strings, and objects are
 * provided. Methods for reading multidimensional arrays are not provided; you
 * can read a multidimensional array by reading a series of single-dimensional
 * arrays.
 * <P>
 * Class InStream includes buffering. Blocks of bytes are read from the
 * underlying input stream and stored in an internal buffer. Items are then read
 * from the internal buffer.
 * <P>
 * <I>Note:</I> Class InStream is not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 13-Jan-2015
 */
public class InStream
	{

// Hidden helper classes.

	private static class ClassInfo
		{
		// Class.
		public Class<?> c;

		// True if class implements interface Streamable.
		public boolean isStreamable;

		// No-argument constructor for a streamable class, or null if none.
		public Constructor<?> ctor;

		public ClassInfo
			(Class<?> c)
			{
			this.c = c;
			this.isStreamable = Streamable.class.isAssignableFrom (c);
			if (this.isStreamable)
				{
				try
					{
					this.ctor = Instance.getDefaultConstructor (c, true);
					}
				catch (NoSuchMethodException exc)
					{
					}
				}
			}
		}

// Hidden data members.

	// Underlying input stream.
	private InputStream in;

	// Buffer for incoming bytes.
	private byte[] buf;
	private int buflen;
	private int bufindex;

	// Cache of classes read by readObject() and readReference().
	private Map<Integer,ClassInfo> indexToClassMap;

	// Cache of objects read by readReference().
	private Map<Integer,Object> indexToObjectMap;

// Hidden operations.

	/**
	 * Verify that this in stream is open.
	 *
	 * @exception  IOException
	 *     Thrown if this in stream is closed.
	 */
	private void verifyOpen()
		throws IOException
		{
		if (in == null)
			throw new IOException ("InStream is closed");
		}

	/**
	 * Verify that this in stream is not at end-of-stream.
	 *
	 * @exception  IOException
	 *     Thrown if this in stream is at end-of-stream.
	 */
	private void verifyMoreBytes()
		throws IOException
		{
		if (buflen < 0)
			throw new EOFException ("Unexpected end-of-stream");
		}

	/**
	 * Read the given byte from the underlying output stream. The byte is
	 * returned as an integer in the range 0..255.
	 *
	 * @return  Byte.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private int read()
		throws IOException
		{
		verifyOpen();
		if (buflen >= 0 && bufindex == buflen) readBuf();
		verifyMoreBytes();
		return buf[bufindex++] & 0xFF;
		}

	/**
	 * Read the contents of the buffer from the underlying input stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private void readBuf()
		throws IOException
		{
		buflen = in.read (buf);
		bufindex = 0;
		}

	/**
	 * Read a class from the underlying input stream. If a null reference was
	 * written, null is returned.
	 *
	 * @return  Class information object, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  ClassNotFoundException
	 *     Thrown if the class could not be found.
	 */
	private ClassInfo readClass()
		throws IOException, ClassNotFoundException
		{
		int i = readUnsignedInt();
		if (i == 0) return null;
		ClassInfo info = classForIndex (i);
		if (info == null)
			{
			String name = readString();
			Class<?> c = Class.forName (name, true, 
				Thread.currentThread().getContextClassLoader());
			info = addClassToCache (i, c);
			}
		return info;
		}

	/**
	 * Returns the class information for the given class index.
	 *
	 * @param  i  Class index.
	 *
	 * @return  Class information, or null if the class is not in the cache.
	 */
	private ClassInfo classForIndex
		(int i)
		{
		if (indexToClassMap == null)
			indexToClassMap = new Map<Integer,ClassInfo>();
		return indexToClassMap.get (i);
		}

	/**
	 * Add the given class to the class cache. Assumes the class is not in the
	 * cache.
	 *
	 * @param  i  Class index.
	 * @param  c  Class.
	 *
	 * @return  Class information.
	 */
	private ClassInfo addClassToCache
		(int i,
		 Class<?> c)
		{
		ClassInfo info = new ClassInfo (c);
		indexToClassMap.put (i, info);
		return info;
		}

	/**
	 * Returns the object for the given object index.
	 *
	 * @param  i  Object index.
	 *
	 * @return  Object, or null if the object is not in the cache.
	 */
	private Object objectForIndex
		(int i)
		{
		if (indexToObjectMap == null)
			indexToObjectMap = new Map<Integer,Object>();
		return indexToObjectMap.get (i);
		}

	/**
	 * Add the given object to the object cache. Assumes the object is not in
	 * the cache.
	 *
	 * @param  i  Object index.
	 * @param  o  Object.
	 */
	private void addObjectToCache
		(int i,
		 Object o)
		{
		indexToObjectMap.put (i, o);
		}

// Exported constructors.

	/**
	 * Construct a new in stream. The internal buffer size is the default (8192
	 * bytes).
	 *
	 * @param  in  Underlying input stream.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>in</TT> is null.
	 */
	public InStream
		(InputStream in)
		{
		this (in, 8192);
		}

	/**
	 * Construct a new in stream with the given internal buffer size.
	 *
	 * @param  in    Underlying input stream.
	 * @param  size  Internal buffer size &ge; 1 (bytes).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>in</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>size</TT> &lt; 1.
	 */
	public InStream
		(InputStream in,
		 int size)
		{
		if (in == null)
			throw new NullPointerException
				("InStream(): in is null");
		if (size < 1)
			throw new IllegalArgumentException (String.format
				("InStream(): size=%d illegal", size));

		this.in = in;
		this.buf = new byte [size];
		this.buflen = 0;
		this.bufindex = 0;
		}

// Exported operations.

	/**
	 * Read a Boolean value from this in stream. One byte is read, either 0
	 * (false) or 1 (true).
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeBoolean(boolean)}.
	 *
	 * @return  Boolean value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public boolean readBoolean()
		throws IOException
		{
		int b = read();
		switch (b)
			{
			case 0:
				return false;
			case 1:
				return true;
			default:
				throw new IllegalDataException (String.format
					("InStream.readBoolean(): Byte=%d illegal", b));
			}
		}

	/**
	 * Read a byte value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeByte(byte)}.
	 *
	 * @return  Byte value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public byte readByte()
		throws IOException
		{
		return (byte) read();
		}

	/**
	 * Read an unsigned byte value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeUnsignedByte(byte)}.
	 *
	 * @return  Unsigned byte value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public byte readUnsignedByte()
		throws IOException
		{
		return (byte) read();
		}

	/**
	 * Read a short value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeShort(short)}.
	 *
	 * @return  Short value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public short readShort()
		throws IOException
		{
		return (short) readInt();
		}

	/**
	 * Read an unsigned short value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeUnsignedShort(short)}.
	 *
	 * @return  Unsigned short value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public short readUnsignedShort()
		throws IOException
		{
		return (short) readUnsignedInt();
		}

	/**
	 * Read a character value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeChar(char)}.
	 *
	 * @return  Character value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public char readChar()
		throws IOException
		{
		return (char) readUnsignedInt();
		}

	/**
	 * Read an integer value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link OutStream#writeInt(int)}.
	 *
	 * @return  Integer value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public int readInt()
		throws IOException
		{
		int v = read();
		if ((v & 0x80) == 0x00)
			{
			// Bytes read (s = sign bit, v = value bit):
			// 0svvvvvv
			v <<= 25;
			v >>= 25;
			}
		else if ((v & 0xC0) == 0x80)
			{
			// 10svvvvv vvvvvvvv
			v = (v << 8) | read();
			v <<= 18;
			v >>= 18;
			}
		else if ((v & 0xE0) == 0xC0)
			{
			// 110svvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 11;
			v >>= 11;
			}
		else if ((v & 0xF0) == 0xE0)
			{
			// 1110svvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 4;
			v >>= 4;
			}
		else
			{
			// 1111ssss svvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		return v;
		}

	/**
	 * Read an unsigned integer value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeUnsignedInt(int)}.
	 *
	 * @return  Unsigned integer value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public int readUnsignedInt()
		throws IOException
		{
		int v = read();
		if ((v & 0x80) == 0x00)
			{
			// Bytes read (v = value bit):
			// 0vvvvvvv
			v &= 0x7F;
			}
		else if ((v & 0xC0) == 0x80)
			{
			// 10vvvvvv vvvvvvvv
			v &= 0x3F;
			v = (v << 8) | read();
			}
		else if ((v & 0xE0) == 0xC0)
			{
			// 110vvvvv vvvvvvvv vvvvvvvv
			v &= 0x1F;
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else if ((v & 0xF0) == 0xE0)
			{
			// 1110vvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v &= 0x0F;
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else
			{
			// 11110000 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		return v;
		}

	/**
	 * Read a long value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeLong(long)}.
	 *
	 * @return  Long value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public long readLong()
		throws IOException
		{
		int b = read();
		long v = b;
		if ((b & 0x80) == 0x00)
			{
			// Bytes read (s = sign bit, v = value bit):
			// 0svvvvvv
			v <<= 57;
			v >>= 57;
			}
		else if ((b & 0xC0) == 0x80)
			{
			// 10svvvvv vvvvvvvv
			v = (v << 8) | read();
			v <<= 50;
			v >>= 50;
			}
		else if ((b & 0xE0) == 0xC0)
			{
			// 110svvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 43;
			v >>= 43;
			}
		else if ((b & 0xF0) == 0xE0)
			{
			// 1110svvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 36;
			v >>= 36;
			}
		else if ((b & 0xF8) == 0xF0)
			{
			// 11110svv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 29;
			v >>= 29;
			}
		else if ((b & 0xFC) == 0xF8)
			{
			// 111110sv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 22;
			v >>= 22;
			}
		else if ((b & 0xFE) == 0xFC)
			{
			// 1111110s vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 15;
			v >>= 15;
			}
		else if (b == 0xFE)
			{
			// 11111110 svvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v <<= 8;
			v >>= 8;
			}
		else
			{
			// 11111111 svvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		return v;
		}

	/**
	 * Read an unsigned long value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeUnsignedLong(long)}.
	 *
	 * @return  Unsigned long value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public long readUnsignedLong()
		throws IOException
		{
		int b = read();
		long v = b;
		if ((b & 0x80) == 0x00)
			{
			// Bytes read (v = value bit):
			// 0vvvvvvv
			v &= 0x7FL;
			}
		else if ((b & 0xC0) == 0x80)
			{
			// 10vvvvvv vvvvvvvv
			v &= 0x3FL;
			v = (v << 8) | read();
			}
		else if ((b & 0xE0) == 0xC0)
			{
			// 110vvvvv vvvvvvvv vvvvvvvv
			v &= 0x1FL;
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else if ((b & 0xF0) == 0xE0)
			{
			// 1110vvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v &= 0x0FL;
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else if ((b & 0xF8) == 0xF0)
			{
			// 11110vvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v &= 0x07L;
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else if ((b & 0xFC) == 0xF8)
			{
			// 111110vv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v &= 0x03L;
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else if ((b & 0xFE) == 0xFC)
			{
			// 1111110v vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v &= 0x01L;
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else if (b == 0xFE)
			{
			// 11111110 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = read() & 0xFFL;
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		else
			{
			// 11111111 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			v = read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			v = (v << 8) | read();
			}
		return v;
		}

	/**
	 * Read a float value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeFloat(float)}.
	 *
	 * @return  Float value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public float readFloat()
		throws IOException
		{
		int vv = read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		return Float.intBitsToFloat (vv);
		}

	/**
	 * Read a double value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeDouble(double)}.
	 *
	 * @return  Double value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public double readDouble()
		throws IOException
		{
		long vv = read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		vv = (vv << 8) | read();
		return Double.longBitsToDouble (vv);
		}

	/**
	 * Read a string value from this in stream.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeString(String)}.
	 *
	 * @return  String value, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public String readString()
		throws IOException
		{
		String v = null;
		int n = readInt();
		if (n >= 0)
			{
			char[] c = new char [n];
			for (int i = 0; i < n; ++ i)
				c[i] = readChar();
			v = new String (c);
			}
		return v;
		}

	/**
	 * Read the given object's fields from this in stream. The fields are read
	 * by calling <TT>v.</TT>{@link Streamable#readIn(InStream) readIn()}.
	 * <TT>v</TT> must not be null.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeFields(Streamable) OutStream.writeFields()}.
	 *
	 * @param  <T>  Data type of the object.
	 * @param  v    Object.
	 *
	 * @return  Object that was read, namely <TT>v</TT>.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public <T extends Streamable> T readFields
		(T v)
		throws IOException
		{
		v.readIn (this);
		return v;
		}

	/**
	 * Read an object from this in stream. An index is read using
	 * <TT>readUnsignedInt()</TT>. If the index is 0, null is returned.
	 * Otherwise, the index designates the object's class; if this is the first
	 * occurrence of the class, the class name is read using
	 * <TT>readString()</TT>; either (a) for a streamable object, a new object
	 * is constructed using the class's no-argument constructor and the object's
	 * fields are read by calling <TT>readFields()</TT>, or (b) for a
	 * serializable object, a byte array is read and the byte array is converted
	 * to an object using Java Object Serialization; and the object is returned.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeObject(Object) OutStream.writeObject()}.
	 * <P>
	 * <I>Note:</I> If the same object is written multiple times using {@link
	 * OutStream#writeObject(Object) writeObject()}, the reader will end up with
	 * multiple <I>different</I> objects (that are copies of each other). If the
	 * reader needs to end up with multiple references to the <I>same</I>
	 * object, use {@link OutStream#writeReference(Object) writeReference()}.
	 * <P>
	 * <I>Note:</I> The calling thread's context class loader is used to
	 * retrieve the class, given the class name.
	 * <P>
	 * <I>Note:</I> When constructing a new instance of a streamable object,
	 * access checks are suppressed. This means the object's class and/or the
	 * class's no-argument constructor need not be public, and a new instance
	 * will still be constructed. However, this also requires that either (a) a
	 * security manager is not installed, or (b) the security manager allows
	 * ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 *
	 * @return  Object, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object could not be
	 *     deserialized.
	 */
	public Object readObject()
		throws IOException
		{
		try
			{
			ClassInfo info = readClass();
			if (info == null)
				{
				return null;
				}
			else if (! info.isStreamable)
				{
				ContextObjectInputStream ois =
					new ContextObjectInputStream
						(new ByteArrayInputStream (readByteArray()));
				return ois.readObject();
				}
			else if (info.ctor != null)
				{
				return readFields ((Streamable) info.ctor.newInstance());
				}
			else
				{
				throw new NoSuchMethodException (String.format
					("%s.<init>()", info.c.getName()));
				}
			}
		catch (Throwable exc)
			{
			throw new DeserializeException
				("InStream.readObject(): Object could not be deserialized",
				 exc);
			}
		}

	/**
	 * Read a reference to an object from this in stream. An index is read using
	 * <TT>readUnsignedInt()</TT>. If the index is 0, null is returned.
	 * Otherwise, the index designates the object reference; if this is the
	 * first occurrence of the object, the object is read using
	 * <TT>readObject()</TT>, and the object is saved in a cache so that later
	 * calls to <TT>readReference()</TT> will return a reference to the saved
	 * object; and the object reference is returned.
	 * <P>
	 * To write the value, the writer must call {@link
	 * OutStream#writeReference(Object) OutStream.writeReference()}.
	 * <P>
	 * <I>Note:</I> If the same object is written multiple times using {@link
	 * OutStream#writeReference(Object) writeReference()}, the reader will end
	 * up with multiple references to the <I>same</I> object. If the reader
	 * needs to end up with multiple <I>different</I> objects (that are copies
	 * of each other), use {@link OutStream#writeObject(Object) writeObject()}.
	 *
	 * @return  Object reference, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object could not be
	 *     deserialized.
	 */
	public Object readReference()
		throws IOException
		{
		int i = readUnsignedInt();
		if (i == 0) return null;
		Object v = objectForIndex (i);
		if (v == null)
			{
			v = readObject();
			addObjectToCache (i, v);
			}
		return v;
		}

	/**
	 * Clear the cache of objects that have been read using {@link
	 * #readReference()}. Afterwards, this in stream will act as though no
	 * objects had been previously read using {@link #readReference()}.
	 * <P>
	 * The reader must call <TT>clearCache()</TT> at the same point as the
	 * writer called {@link OutStream#clearCache()}.
	 */
	public void clearCache()
		{
		indexToObjectMap.clear();
		}

	/**
	 * Read a Boolean array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readBoolean()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeBooleanArray(boolean[])} or {@link
	 * OutStream#writeBooleanArray(boolean[],int,int)}.
	 *
	 * @return  Boolean array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public boolean[] readBooleanArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readBooleanArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		boolean[] v = new boolean [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readBoolean();
		return v;
		}

	/**
	 * Read the given Boolean array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readBoolean()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeBooleanArray(boolean[])} or {@link
	 * OutStream#writeBooleanArray(boolean[],int,int)}.
	 *
	 * @param  v  Boolean array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readBooleanArray
		(boolean[] v)
		throws IOException
		{
		readBooleanArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given Boolean array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readBoolean()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeBooleanArray(boolean[])} or {@link
	 * OutStream#writeBooleanArray(boolean[],int,int)}.
	 *
	 * @param  v    Boolean array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readBooleanArray
		(boolean[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readBooleanArray(): Index out of bounds, v=boolean[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readBooleanArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readBoolean();
		}

	/**
	 * Read a byte array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readByte()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeByteArray(byte[])} or {@link
	 * OutStream#writeByteArray(byte[],int,int)}.
	 *
	 * @return  Byte array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public byte[] readByteArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readByteArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		byte[] v = new byte [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readByte();
		return v;
		}

	/**
	 * Read the given byte array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readByte()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeByteArray(byte[])} or {@link
	 * OutStream#writeByteArray(byte[],int,int)}.
	 *
	 * @param  v  Byte array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readByteArray
		(byte[] v)
		throws IOException
		{
		readByteArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given byte array from this in stream. The length is
	 * read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readByte()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeByteArray(byte[])} or {@link
	 * OutStream#writeByteArray(byte[],int,int)}.
	 *
	 * @param  v    Byte array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readByteArray
		(byte[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readByteArray(): Index out of bounds, v=byte[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readByteArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readByte();
		}

	/**
	 * Read an unsigned byte array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readUnsignedByte()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedByteArray(byte[])} or {@link
	 * OutStream#writeUnsignedByteArray(byte[],int,int)}.
	 *
	 * @return  Unsigned byte array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public byte[] readUnsignedByteArray()
		throws IOException
		{
		return readByteArray();
		}

	/**
	 * Read the given unsigned byte array from this in stream. The length is
	 * read using <TT>readInt()</TT>. If the length is not the same as the
	 * length of <TT>v</TT>, an exception is thrown. Otherwise, each element of
	 * <TT>v</TT> is read using <TT>readUnsignedByte()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedByteArray(byte[])} or {@link
	 * OutStream#writeUnsignedByteArray(byte[],int,int)}.
	 *
	 * @param  v  Unsigned byte array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedByteArray
		(byte[] v)
		throws IOException
		{
		readByteArray (v);
		}

	/**
	 * Read a portion of the given unsigned byte array from this in stream. The
	 * length is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readUnsignedByte()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedByteArray(byte[])} or {@link
	 * OutStream#writeUnsignedByteArray(byte[],int,int)}.
	 *
	 * @param  v    Unsigned byte array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedByteArray
		(byte[] v,
		 int off,
		 int len)
		throws IOException
		{
		readByteArray (v, off, len);
		}

	/**
	 * Read a short array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readShort()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeShortArray(short[])} or {@link
	 * OutStream#writeShortArray(short[],int,int)}.
	 *
	 * @return  Short array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public short[] readShortArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readShortArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		short[] v = new short [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readShort();
		return v;
		}

	/**
	 * Read the given short array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readShort()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeShortArray(short[])} or {@link
	 * OutStream#writeShortArray(short[],int,int)}.
	 *
	 * @param  v  Short array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readShortArray
		(short[] v)
		throws IOException
		{
		readShortArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given short array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readShort()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeShortArray(short[])} or {@link
	 * OutStream#writeShortArray(short[],int,int)}.
	 *
	 * @param  v    Short array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readShortArray
		(short[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readShortArray(): Index out of bounds, v=short[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readShortArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readShort();
		}

	/**
	 * Read an unsigned short array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readUnsignedShort()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedShortArray(short[])} or {@link
	 * OutStream#writeUnsignedShortArray(short[],int,int)}.
	 *
	 * @return  Unsigned short array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public short[] readUnsignedShortArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readUnsignedShortArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		short[] v = new short [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readUnsignedShort();
		return v;
		}

	/**
	 * Read the given unsigned short array from this in stream. The length is
	 * read using <TT>readInt()</TT>. If the length is not the same as the
	 * length of <TT>v</TT>, an exception is thrown. Otherwise, each element of
	 * <TT>v</TT> is read using <TT>readUnsignedShort()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedShortArray(short[])} or {@link
	 * OutStream#writeUnsignedShortArray(short[],int,int)}.
	 *
	 * @param  v  Unsigned short array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedShortArray
		(short[] v)
		throws IOException
		{
		readUnsignedShortArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given unsigned short array from this in stream. The
	 * length is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using
	 * <TT>readUnsignedShort()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedShortArray(short[])} or {@link
	 * OutStream#writeUnsignedShortArray(short[],int,int)}.
	 *
	 * @param  v    Unsigned short array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedShortArray
		(short[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readUnsignedShortArray(): Index out of bounds, v=short[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readUnsignedShortArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readUnsignedShort();
		}

	/**
	 * Read a character array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readChar()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeCharArray(char[])} or {@link
	 * OutStream#writeCharArray(char[],int,int)}.
	 *
	 * @return  Character array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public char[] readCharArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readCharArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		char[] v = new char [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readChar();
		return v;
		}

	/**
	 * Read the given character array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readChar()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeCharArray(char[])} or {@link
	 * OutStream#writeCharArray(char[],int,int)}.
	 *
	 * @param  v  Character array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readCharArray
		(char[] v)
		throws IOException
		{
		readCharArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given character array from this in stream. The
	 * length is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readChar()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeCharArray(char[])} or {@link
	 * OutStream#writeCharArray(char[],int,int)}.
	 *
	 * @param  v    Character array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readCharArray
		(char[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readCharArray(): Index out of bounds, v=char[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readCharArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readChar();
		}

	/**
	 * Read an integer array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readInt()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeIntArray(int[])} or {@link
	 * OutStream#writeIntArray(int[],int,int)}.
	 *
	 * @return  Integer array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public int[] readIntArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readIntArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		int[] v = new int [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readInt();
		return v;
		}

	/**
	 * Read the given integer array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readInt()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeIntArray(int[])} or {@link
	 * OutStream#writeIntArray(int[],int,int)}.
	 *
	 * @param  v  Integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIntArray
		(int[] v)
		throws IOException
		{
		readIntArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given integer array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readInt()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeIntArray(int[])} or {@link
	 * OutStream#writeIntArray(int[],int,int)}.
	 *
	 * @param  v    Integer array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIntArray
		(int[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readIntArray(): Index out of bounds, v=int[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readIntArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readInt();
		}

	/**
	 * Read an unsigned integer array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readUnsignedInt()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedIntArray(int[])} or {@link
	 * OutStream#writeUnsignedIntArray(int[],int,int)}.
	 *
	 * @return  Unsigned integer array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public int[] readUnsignedIntArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readUnsignedIntArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		int[] v = new int [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readUnsignedInt();
		return v;
		}

	/**
	 * Read the given unsigned integer array from this in stream. The length is
	 * read using <TT>readInt()</TT>. If the length is not the same as the
	 * length of <TT>v</TT>, an exception is thrown. Otherwise, each element of
	 * <TT>v</TT> is read using <TT>readUnsignedInt()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedIntArray(int[])} or {@link
	 * OutStream#writeUnsignedIntArray(int[],int,int)}.
	 *
	 * @param  v  Unsigned integer array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedIntArray
		(int[] v)
		throws IOException
		{
		readUnsignedIntArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given unsigned integer array from this in stream.
	 * The length is read using <TT>readInt()</TT>. If the length is not the
	 * same as <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT>
	 * elements starting from <TT>v[off]</TT> are read using
	 * <TT>readUnsignedInt()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedIntArray(int[])} or {@link
	 * OutStream#writeUnsignedIntArray(int[],int,int)}.
	 *
	 * @param  v    Unsigned integer array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedIntArray
		(int[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readUnsignedIntArray(): Index out of bounds, v=int[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readUnsignedIntArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readUnsignedInt();
		}

	/**
	 * Read a long array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readLong()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeLongArray(long[])} or {@link
	 * OutStream#writeLongArray(long[],int,int)}.
	 *
	 * @return  Long array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public long[] readLongArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readLongArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		long[] v = new long [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readLong();
		return v;
		}

	/**
	 * Read the given long array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readLong()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeLongArray(long[])} or {@link
	 * OutStream#writeLongArray(long[],int,int)}.
	 *
	 * @param  v  Long array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readLongArray
		(long[] v)
		throws IOException
		{
		readLongArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given long array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readLong()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeLongArray(long[])} or {@link
	 * OutStream#writeLongArray(long[],int,int)}.
	 *
	 * @param  v    Long array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readLongArray
		(long[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readLongArray(): Index out of bounds, v=long[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readLongArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readLong();
		}

	/**
	 * Read an unsigned long array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readUnsignedLong()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedLongArray(long[])} or {@link
	 * OutStream#writeUnsignedLongArray(long[],int,int)}.
	 *
	 * @return  Unsigned long array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public long[] readUnsignedLongArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readUnsignedLongArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		long[] v = new long [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readUnsignedLong();
		return v;
		}

	/**
	 * Read the given unsigned long array from this in stream. The length is
	 * read using <TT>readInt()</TT>. If the length is not the same as the
	 * length of <TT>v</TT>, an exception is thrown. Otherwise, each element of
	 * <TT>v</TT> is read using <TT>readUnsignedLong()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedLongArray(long[])} or {@link
	 * OutStream#writeUnsignedLongArray(long[],int,int)}.
	 *
	 * @param  v  Unsigned long array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedLongArray
		(long[] v)
		throws IOException
		{
		readUnsignedLongArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given unsigned long array from this in stream.
	 * The length is read using <TT>readInt()</TT>. If the length is not the
	 * same as <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT>
	 * elements starting from <TT>v[off]</TT> are read using
	 * <TT>readUnsignedLong()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeUnsignedLongArray(long[])} or {@link
	 * OutStream#writeUnsignedLongArray(long[],int,int)}.
	 *
	 * @param  v    Unsigned long array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readUnsignedLongArray
		(long[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readUnsignedLongArray(): Index out of bounds, v=long[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readUnsignedLongArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readUnsignedLong();
		}

	/**
	 * Read a float array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readFloat()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeFloatArray(float[])} or {@link
	 * OutStream#writeFloatArray(float[],int,int)}.
	 *
	 * @return  Float array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public float[] readFloatArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readFloatArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		float[] v = new float [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readFloat();
		return v;
		}

	/**
	 * Read the given float array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readFloat()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeFloatArray(float[])} or {@link
	 * OutStream#writeFloatArray(float[],int,int)}.
	 *
	 * @param  v  Float array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readFloatArray
		(float[] v)
		throws IOException
		{
		readFloatArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given float array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readFloat()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeFloatArray(float[])} or {@link
	 * OutStream#writeFloatArray(float[],int,int)}.
	 *
	 * @param  v    Float array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readFloatArray
		(float[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readFloatArray(): Index out of bounds, v=float[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readFloatArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readFloat();
		}

	/**
	 * Read a double array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readDouble()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeDoubleArray(double[])} or {@link
	 * OutStream#writeDoubleArray(double[],int,int)}.
	 *
	 * @return  Double array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public double[] readDoubleArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readDoubleArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		double[] v = new double [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readDouble();
		return v;
		}

	/**
	 * Read the given double array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readDouble()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeDoubleArray(double[])} or {@link
	 * OutStream#writeDoubleArray(double[],int,int)}.
	 *
	 * @param  v  Double array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readDoubleArray
		(double[] v)
		throws IOException
		{
		readDoubleArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given double array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readDouble()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeDoubleArray(double[])} or {@link
	 * OutStream#writeDoubleArray(double[],int,int)}.
	 *
	 * @param  v    Double array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readDoubleArray
		(double[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readDoubleArray(): Index out of bounds, v=double[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readDoubleArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readDouble();
		}

	/**
	 * Read a string array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, a new array of the given length is created, and each element
	 * of the array is read using <TT>readString()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeStringArray(String[])} or {@link
	 * OutStream#writeStringArray(String[],int,int)}.
	 *
	 * @return  String array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public String[] readStringArray()
		throws IOException
		{
		int n = readInt();
		if (n < -1)
			throw new IllegalDataException (String.format
				("InStream.readStringArray(): Length=%d illegal", n));
		if (n == -1)
			return null;
		String[] v = new String [n];
		for (int i = 0; i < n; ++ i)
			v[i] = readString();
		return v;
		}

	/**
	 * Read the given string array from this in stream. The length is read
	 * using <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readString()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeStringArray(String[])} or {@link
	 * OutStream#writeStringArray(String[],int,int)}.
	 *
	 * @param  v  String array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readStringArray
		(String[] v)
		throws IOException
		{
		readStringArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given string array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readString()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeStringArray(String[])} or {@link
	 * OutStream#writeStringArray(String[],int,int)}.
	 *
	 * @param  v    String array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readStringArray
		(String[] v,
		 int off,
		 int len)
		throws IOException
		{
		int vlen = v.length;
		if (off < 0 || len < 0 || off + len > vlen)
			throw new IndexOutOfBoundsException (String.format
				("InStream.readStringArray(): Index out of bounds, v=String[%d], off=%d, len=%d",
				 vlen, off, len));
		int n = readInt();
		if (n != len)
			throw new IllegalDataException (String.format
				("InStream.readStringArray(): Length=%d illegal, len=%d",
				 n, len));
		for (int i = 0; i < len; ++ i)
			v[off+i] = readString();
		}

	/**
	 * Read an array of objects from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is &minus;1, null is returned.
	 * Otherwise, the array element class is read, a new array of the given
	 * length and the given element class is created, and each element of the
	 * array is read using <TT>readObject()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeObjectArray(Object[])} or {@link
	 * OutStream#writeObjectArray(Object[],int,int)}.
	 *
	 * @return  Object array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object array could not
	 *     be deserialized.
	 */
	public Object[] readObjectArray()
		throws IOException
		{
		try
			{
			int n = readInt();
			if (n < -1)
				throw new IllegalDataException (String.format
					("InStream.readObjectArray(): Length=%d illegal", n));
			if (n == -1)
				return null;
			ClassInfo info = readClass();
			Object[] v = (Object[]) Array.newInstance (info.c, n);
			for (int i = 0; i < n; ++ i)
				v[i] = readObject();
			return v;
			}
		catch (Throwable exc)
			{
			throw new DeserializeException
				("InStream.readObjectArray(): Object could not be deserialized",
				 exc);
			}
		}

	/**
	 * Read the given object array from this in stream. The length is read using
	 * <TT>readInt()</TT>. If the length is not the same as the length of
	 * <TT>v</TT>, an exception is thrown. Otherwise, the array element class is
	 * read. If this is not the same as (or a subclass of) the element class of
	 * <TT>v</TT>, an exception is thrown. Otherwise, each element of <TT>v</TT>
	 * is read using <TT>readObject()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeObjectArray(Object[])} or {@link
	 * OutStream#writeObjectArray(Object[],int,int)}.
	 *
	 * @param  v  Object array.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object array could not
	 *     be deserialized.
	 */
	public void readObjectArray
		(Object[] v)
		throws IOException
		{
		readObjectArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given object array from this in stream. The length
	 * is read using <TT>readInt()</TT>. If the length is not the same as
	 * <TT>len</TT>, an exception is thrown. Otherwise, the array element class
	 * is read. If this is not the same as (or a subclass of) the element class
	 * of <TT>v</TT>, an exception is thrown. Otherwise, <TT>len</TT> elements
	 * starting from <TT>v[off]</TT> are read using <TT>readObject()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeObjectArray(Object[])} or {@link
	 * OutStream#writeObjectArray(Object[],int,int)}.
	 *
	 * @param  v    Object array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object array could not
	 *     be deserialized.
	 */
	public void readObjectArray
		(Object[] v,
		 int off,
		 int len)
		throws IOException
		{
		try
			{
			int vlen = v.length;
			if (off < 0 || len < 0 || off + len > vlen)
				throw new IndexOutOfBoundsException (String.format
					("InStream.readObjectArray(): Index out of bounds, v=Object[%d], off=%d, len=%d",
					 vlen, off, len));
			int n = readInt();
			if (n != len)
				throw new IllegalDataException (String.format
					("InStream.readObjectArray(): Length=%d illegal, len=%d",
					 n, len));
			ClassInfo info = readClass();
			Class<?> elementClass = v.getClass().getComponentType();
			if (! elementClass.isAssignableFrom (info.c))
				throw new ClassCastException (String.format
					("InStream.readObjectArray(): Elements in stream (%s) cannot be cast to elements of v (%s)",
					 info.c, elementClass));
			for (int i = 0; i < len; ++ i)
				v[off+i] = readObject();
			}
		catch (Throwable exc)
			{
			throw new DeserializeException
				("InStream.readObjectArray(): Object could not be deserialized",
				 exc);
			}
		}

	/**
	 * Read an array of object references from this in stream. The length is
	 * read using <TT>readInt()</TT>. If the length is &minus;1, null is
	 * returned. Otherwise, the array element class is read, a new array of the
	 * given length and the given element class is created, and each element of
	 * the array is read using <TT>readReference()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeObjectArray(Object[])} or {@link
	 * OutStream#writeObjectArray(Object[],int,int)}.
	 *
	 * @return  Object array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object reference array
	 *     could not be deserialized.
	 */
	public Object[] readReferenceArray()
		throws IOException
		{
		try
			{
			int n = readInt();
			if (n < -1)
				throw new IllegalDataException (String.format
					("InStream.readReferenceArray(): Length=%d illegal", n));
			if (n == -1)
				return null;
			ClassInfo info = readClass();
			Object[] v = (Object[]) Array.newInstance (info.c, n);
			for (int i = 0; i < n; ++ i)
				v[i] = readReference();
			return v;
			}
		catch (Throwable exc)
			{
			throw new DeserializeException
				("InStream.readReferenceArray(): Object could not be deserialized",
				 exc);
			}
		}

	/**
	 * Read the given object reference array from this in stream. The length is
	 * read using <TT>readInt()</TT>. If the length is not the same as the
	 * length of <TT>v</TT>, an exception is thrown. Otherwise, the array
	 * element class is read. If this is not the same as (or a subclass of) the
	 * element class of <TT>v</TT>, an exception is thrown. Otherwise, each
	 * element of <TT>v</TT> is read using <TT>readReference()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeObjectArray(Object[])} or {@link
	 * OutStream#writeObjectArray(Object[],int,int)}.
	 *
	 * @param  v  Object array.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object reference array
	 *     could not be deserialized.
	 */
	public void readReferenceArray
		(Object[] v)
		throws IOException
		{
		readReferenceArray (v, 0, v.length);
		}

	/**
	 * Read a portion of the given object reference array from this in stream.
	 * The length is read using <TT>readInt()</TT>. If the length is not the
	 * same as <TT>len</TT>, an exception is thrown. Otherwise, the array
	 * element class is read. If this is not the same as (or a subclass of) the
	 * element class of <TT>v</TT>, an exception is thrown. Otherwise,
	 * <TT>len</TT> elements starting from <TT>v[off]</TT> are read using
	 * <TT>readReference()</TT>.
	 * <P>
	 * To write the array, the writer must call {@link
	 * OutStream#writeObjectArray(Object[])} or {@link
	 * OutStream#writeObjectArray(Object[],int,int)}.
	 *
	 * @param  v    Object array.
	 * @param  off  Index of first element to read.
	 * @param  len  Number of elements to read.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred. Thrown if the object reference array
	 *     could not be deserialized.
	 */
	public void readReferenceArray
		(Object[] v,
		 int off,
		 int len)
		throws IOException
		{
		try
			{
			int vlen = v.length;
			if (off < 0 || len < 0 || off + len > vlen)
				throw new IndexOutOfBoundsException (String.format
					("InStream.readReferenceArray(): Index out of bounds, v=Object[%d], off=%d, len=%d",
					 vlen, off, len));
			int n = readInt();
			if (n != len)
				throw new IllegalDataException (String.format
					("InStream.readReferenceArray(): Length=%d illegal, len=%d",
					 n, len));
			ClassInfo info = readClass();
			Class<?> elementClass = v.getClass().getComponentType();
			if (! elementClass.isAssignableFrom (info.c))
				throw new ClassCastException (String.format
					("InStream.readReferenceArray(): Elements in stream (%s) cannot be cast to elements of v (%s)",
					 info.c, elementClass));
			for (int i = 0; i < len; ++ i)
				v[off+i] = readReference();
			}
		catch (Throwable exc)
			{
			throw new DeserializeException
				("InStream.readReferenceArray(): Object could not be deserialized",
				 exc);
			}
		}

	/**
	 * Close this in stream. The underlying stream is closed.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void close()
		throws IOException
		{
		try
			{
			in.close();
			}
		finally
			{
			in = null;
			buf = null;
			buflen = 0;
			bufindex = -1;
			indexToClassMap = null;
			indexToObjectMap = null;
			}
		}

	/**
	 * Finalize this in stream.
	 */
	protected void finalize()
		throws Throwable
		{
		close();
		}

	}
