//******************************************************************************
//
// File:    OutStream.java
// Package: edu.rit.io
// Unit:    Class edu.rit.io.OutStream
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

import edu.rit.util.IdentityMap;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

/**
 * Class OutStream provides an object that writes primitive data types, strings,
 * objects, and arrays in binary form to an underlying output stream. The
 * resulting byte stream can be read using class {@linkplain InStream} as
 * described in more detail in the individual methods.
 * <P>
 * The methods for writing integer types and strings write a variable number of
 * bytes, as described in more detail in the individual methods. This can save
 * space in the byte stream if small integer values are written more frequently
 * than large integer values.
 * <P>
 * Several methods for writing objects are provided:
 * <UL>
 * <P><LI>
 * {@link #writeFields(Streamable) writeFields()} writes just the fields of a
 * {@linkplain Streamable} object. Use this method when the reader knows the
 * class of the object ahead of time and has already created an instance of that
 * class.
 * <P><LI>
 * {@link #writeObject(Object) writeObject()} writes both the class name and the
 * fields of a {@linkplain Streamable} or {@linkplain java.io.Serializable
 * Serializable} object. Use this method when the reader does not know the class
 * of the object ahead of time.
 * <P><LI>
 * {@link #writeReference(Object) writeReference()} writes the class name and
 * the fields of a {@linkplain Streamable} or {@linkplain java.io.Serializable
 * Serializable} object, and also keeps a reference to the object. If the same
 * object is passed to {@link #writeReference(Object) writeReference()} again,
 * just a reference to the previous object is written. Use this method when the
 * reader needs to end up with multiple references to the same object.
 * </UL>
 * <P>
 * Classes OutStream and {@linkplain InStream} provide a <I>lightweight</I>
 * object serialization capability&mdash;one that generates fewer bytes than
 * Java Object Serialization in package java.io. However, the programmer is
 * responsible for writing all the serialization and deserialization code in
 * each class that implements interface {@linkplain Streamable}.
 * <P>
 * To support interoperability with classes designed to work with Java Object
 * Serialization, {@link #writeObject(Object) writeObject()} and {@link
 * #writeReference(Object) writeReference()} will work both on {@linkplain
 * Streamable} objects and on {@linkplain java.io.Serializable Serializable}
 * objects. In the latter case, the object is converted to a byte array using
 * Java Object Serialization, and the byte array is written.
 * <P>
 * Methods for writing arrays of primitive types, strings, and objects are
 * provided. Methods for writing multidimensional arrays are not provided; you
 * can write a multidimensional array by writing a series of single-dimensional
 * arrays.
 * <P>
 * Class OutStream includes buffering. All bytes written are stored in an
 * internal buffer. The buffer is flushed to the underlying output stream when
 * the buffer fills up, the {@link #flush() flush()} method is called, or the
 * {@link #close() close()} method is called.
 * <P>
 * <I>Note:</I> Class OutStream is not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 14-Jan-2015
 */
public class OutStream
	{

// Hidden data members.

	// Underlying output stream.
	private OutputStream out;

	// Buffer for outgoing bytes.
	private byte[] buf;
	private int buflen;

	// Cache of classes written by writeObject() and writeReference().
	private IdentityMap<Class<?>,Integer> classToIndexMap;

	// Cache of objects written by writeReference().
	private IdentityMap<Object,Integer> objectToIndexMap;

// Hidden operations.

	/**
	 * Verify that this out stream is open.
	 *
	 * @exception  IOException
	 *     Thrown if this out stream is closed.
	 */
	private void verifyOpen()
		throws IOException
		{
		if (out == null)
			throw new IOException ("OutStream is closed");
		}

	/**
	 * Write the given byte to the underlying output stream. Only bits 0..7 are
	 * written.
	 *
	 * @param  b  Byte.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private void write
		(int b)
		throws IOException
		{
		verifyOpen();
		buf[buflen++] = (byte) b;
		if (buflen == buf.length) writeBuf();
		}

	/**
	 * Write the contents of the buffer to the underlying output stream. Assumes
	 * the buffer length is greater than 0.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private void writeBuf()
		throws IOException
		{
		verifyOpen();
		out.write (buf, 0, buflen);
		buflen = 0;
		}

	/**
	 * Write the given class to the underlying output stream.
	 *
	 * @param  c  Class.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	private void writeClass
		(Class<?> c)
		throws IOException
		{
		int i = indexForClass (c);
		if (i == 0)
			{
			i = addClassToCache (c);
			writeUnsignedInt (i);
			writeString (c.getName());
			}
		else
			{
			writeUnsignedInt (i);
			}
		}

	/**
	 * Returns the class index for the given class.
	 *
	 * @param  c  Class.
	 *
	 * @return  Class index, or 0 if the class is not in the cache.
	 */
	private int indexForClass
		(Class<?> c)
		{
		if (classToIndexMap == null)
			classToIndexMap = new IdentityMap<Class<?>,Integer>();
		Integer i = classToIndexMap.get (c);
		return i == null ? 0 : i;
		}

	/**
	 * Add the given class to the class cache. Assumes the class is not in the
	 * cache.
	 *
	 * @param  c  Class.
	 *
	 * @return  Class index.
	 */
	private int addClassToCache
		(Class<?> c)
		{
		int i = classToIndexMap.size() + 1;
		classToIndexMap.put (c, i);
		return i;
		}

	/**
	 * Returns the object index for the given object.
	 *
	 * @param  o  Object.
	 *
	 * @return  Object index, or 0 if the object is not in the cache.
	 */
	private int indexForObject
		(Object o)
		{
		if (objectToIndexMap == null)
			objectToIndexMap = new IdentityMap<Object,Integer>();
		Integer i = objectToIndexMap.get (o);
		return i == null ? 0 : i;
		}

	/**
	 * Add the given object to the object cache. Assumes the object is not in
	 * the cache.
	 *
	 * @param  o  Object.
	 *
	 * @return  Object index.
	 */
	private int addObjectToCache
		(Object o)
		{
		int i = objectToIndexMap.size() + 1;
		objectToIndexMap.put (o, i);
		return i;
		}

// Exported constructors.

	/**
	 * Construct a new out stream. The internal buffer size is the default (8192
	 * bytes).
	 *
	 * @param  out  Underlying output stream.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 */
	public OutStream
		(OutputStream out)
		{
		this (out, 8192);
		}

	/**
	 * Construct a new out stream with the given internal buffer size.
	 *
	 * @param  out   Underlying output stream.
	 * @param  size  Internal buffer size &ge; 1 (bytes).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null.
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>size</TT> &lt; 1.
	 */
	public OutStream
		(OutputStream out,
		 int size)
		{
		if (out == null)
			throw new NullPointerException
				("OutStream(): out is null");
		if (size < 1)
			throw new IllegalArgumentException (String.format
				("OutStream(): size=%d illegal", size));

		this.out = out;
		this.buf = new byte [size];
		this.buflen = 0;
		}

// Exported operations.

	/**
	 * Write the given Boolean value to this out stream. One byte is written,
	 * either 0 (if <TT>v</TT> is false) or 1 (if <TT>v</TT> is true).
	 * <P>
	 * To read the value, the reader must call {@link InStream#readBoolean()}.
	 *
	 * @param  v  Boolean value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeBoolean
		(boolean v)
		throws IOException
		{
		write (v ? 1 : 0);
		}

	/**
	 * Write the given byte value to this out stream.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readByte()}.
	 *
	 * @param  v  Byte value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeByte
		(byte v)
		throws IOException
		{
		write (v);
		}

	/**
	 * Write the given unsigned byte value to this out stream.
	 * <P>
	 * To read the value, the reader must call {@link
	 * InStream#readUnsignedByte()}.
	 *
	 * @param  v  Unsigned byte value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedByte
		(byte v)
		throws IOException
		{
		write (v);
		}

	/**
	 * Write the given short value to this out stream. From one to three bytes
	 * are written, depending on the value; smaller absolute values require
	 * fewer bytes.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readShort()}.
	 *
	 * @param  v  Short value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeShort
		(short v)
		throws IOException
		{
		writeInt (v);
		}

	/**
	 * Write the given unsigned short value to this out stream. From one to
	 * three bytes are written, depending on the value; smaller absolute values
	 * require fewer bytes.
	 * <P>
	 * To read the value, the reader must call {@link
	 * InStream#readUnsignedShort()}.
	 *
	 * @param  v  Unsigned short value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedShort
		(short v)
		throws IOException
		{
		writeUnsignedInt (v);
		}

	/**
	 * Write the given character value to this out stream. From one to three
	 * bytes are written, depending on the value; smaller absolute values
	 * require fewer bytes.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readChar()}.
	 *
	 * @param  v  Character value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeChar
		(char v)
		throws IOException
		{
		writeUnsignedInt (v);
		}

	/**
	 * Write the given integer value to this out stream. From one to five bytes
	 * are written, depending on the value; smaller absolute values require
	 * fewer bytes.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readInt()}.
	 *
	 * @param  v  Integer value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeInt
		(int v)
		throws IOException
		{
		if (0xFFFFFFC0 <= v && v <= 0x0000003F)
			{
			// Bytes written (s = sign bit, v = value bit):
			// 0svvvvvv
			write (v & 0x7F);
			}
		else if (0xFFFFE000 <= v && v <= 0x00001FFF)
			{
			// 10svvvvv vvvvvvvv
			write (((v >> 8) & 0x3F) | 0x80);
			write (v);
			}
		else if (0xFFF00000 <= v && v <= 0x000FFFFF)
			{
			// 110svvvv vvvvvvvv vvvvvvvv
			write (((v >> 16) & 0x1F) | 0xC0);
			write (v >> 8);
			write (v);
			}
		else if (0xF8000000 <= v && v <= 0x07FFFFFF)
			{
			// 1110svvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (((v >> 24) & 0x0F) | 0xE0);
			write (v >> 16);
			write (v >> 8);
			write (v);
			}
		else
			{
			// 1111ssss svvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write ((v >> 31) | 0xF0);
			write (v >> 24);
			write (v >> 16);
			write (v >> 8);
			write (v);
			}
		}

	/**
	 * Write the given unsigned integer value to this out stream. From one to
	 * five bytes are written, depending on the value; smaller absolute values
	 * require fewer bytes.
	 * <P>
	 * To read the value, the reader must call {@link
	 * InStream#readUnsignedInt()}.
	 *
	 * @param  v  Unsigned integer value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedInt
		(int v)
		throws IOException
		{
		if (v < 0)
			{
			// Bytes written (v = value bit):
			// 11110000 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (0xF0);
			write (v >> 24);
			write (v >> 16);
			write (v >> 8);
			write (v);
			}
		else if (v <= 0x0000007F)
			{
			// 0vvvvvvv
			write (v);
			}
		else if (v <= 0x00003FFF)
			{
			// 10vvvvvv vvvvvvvv
			write ((v >> 8) | 0x80);
			write (v);
			}
		else if (v <= 0x001FFFFF)
			{
			// 110vvvvv vvvvvvvv vvvvvvvv
			write ((v >> 16) | 0xC0);
			write (v >> 8);
			write (v);
			}
		else if (v <= 0x0FFFFFFF)
			{
			// 1110vvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write ((v >> 24) | 0xE0);
			write (v >> 16);
			write (v >> 8);
			write (v);
			}
		else
			{
			// 11110000 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (0xF0);
			write (v >> 24);
			write (v >> 16);
			write (v >> 8);
			write (v);
			}
		}

	/**
	 * Write the given long value to this out stream. From one to nine bytes are
	 * written, depending on the value; smaller absolute values require fewer
	 * bytes.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readLong()}.
	 *
	 * @param  v  Long value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeLong
		(long v)
		throws IOException
		{
		if (0xFFFFFFFFFFFFFFC0L <= v && v <= 0x000000000000003FL)
			{
			// Bytes written (s = sign bit, v = value bit):
			// 0svvvvvv
			write ((int)(v) & 0x7F);
			}
		else if (0xFFFFFFFFFFFFE000L <= v && v <= 0x0000000000001FFFL)
			{
			// 10svvvvv vvvvvvvv
			write (((int)(v >> 8) & 0x3F) | 0x80);
			write ((int)(v));
			}
		else if (0xFFFFFFFFFFF00000L <= v && v <= 0x00000000000FFFFFL)
			{
			// 110svvvv vvvvvvvv vvvvvvvv
			write (((int)(v >> 16) & 0x1F) | 0xC0);
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (0xFFFFFFFFF8000000L <= v && v <= 0x0000000007FFFFFFL)
			{
			// 1110svvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (((int)(v >> 24) & 0x0F) | 0xE0);
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (0xFFFFFFFC00000000L <= v && v <= 0x00000003FFFFFFFFL)
			{
			// 11110svv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (((int)(v >> 32) & 0x07) | 0xF0);
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (0xFFFFFE0000000000L <= v && v <= 0x000001FFFFFFFFFFL)
			{
			// 111110sv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (((int)(v >> 40) & 0x03) | 0xF8);
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (0xFFFF000000000000L <= v && v <= 0x0000FFFFFFFFFFFFL)
			{
			// 1111110s vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (((int)(v >> 48) & 0x01) | 0xFC);
			write ((int)(v >> 40));
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (0xFF80000000000000L <= v && v <= 0x007FFFFFFFFFFFFFL)
			{
			// 11111110 svvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (0xFE);
			write ((int)(v >> 48));
			write ((int)(v >> 40));
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else
			{
			// 11111111 svvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (0xFF);
			write ((int)(v >> 56));
			write ((int)(v >> 48));
			write ((int)(v >> 40));
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		}

	/**
	 * Write the given unsigned long value to this out stream. From one to nine
	 * bytes are written, depending on the value; smaller absolute values
	 * require fewer bytes.
	 * <P>
	 * To read the value, the reader must call {@link
	 * InStream#readUnsignedLong()}.
	 *
	 * @param  v  Unsigned long value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedLong
		(long v)
		throws IOException
		{
		if (v < 0L)
			{
			// Bytes written (v = value bit):
			// 11111111 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (0xFF);
			write ((int)(v >> 56));
			write ((int)(v >> 48));
			write ((int)(v >> 40));
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (v <= 0x000000000000007FL)
			{
			// 0vvvvvvv
			write ((int)(v));
			}
		else if (v <= 0x0000000000003FFFL)
			{
			// 10vvvvvv vvvvvvvv
			write ((int)(v >> 8) | 0x80);
			write ((int)(v));
			}
		else if (v <= 0x00000000001FFFFFL)
			{
			// 110vvvvv vvvvvvvv vvvvvvvv
			write ((int)(v >> 16) | 0xC0);
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (v <= 0x000000000FFFFFFFL)
			{
			// 1110vvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write ((int)(v >> 24) | 0xE0);
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (v <= 0x00000007FFFFFFFFL)
			{
			// 11110vvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write ((int)(v >> 32) | 0xF0);
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (v <= 0x000003FFFFFFFFFFL)
			{
			// 111110vv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write ((int)(v >> 40) | 0xF8);
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (v <= 0x0001FFFFFFFFFFFFL)
			{
			// 1111110v vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write ((int)(v >> 48) | 0xFC);
			write ((int)(v >> 40));
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else if (v <= 0x00FFFFFFFFFFFFFFL)
			{
			// 11111110 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (0xFE);
			write ((int)(v >> 48));
			write ((int)(v >> 40));
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		else
			{
			// 11111111 vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv vvvvvvvv
			write (0xFF);
			write ((int)(v >> 56));
			write ((int)(v >> 48));
			write ((int)(v >> 40));
			write ((int)(v >> 32));
			write ((int)(v >> 24));
			write ((int)(v >> 16));
			write ((int)(v >> 8));
			write ((int)(v));
			}
		}

	/**
	 * Write the given float value to this out stream. Four bytes are written.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readFloat()}.
	 *
	 * @param  v  Float value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeFloat
		(float v)
		throws IOException
		{
		int vv = Float.floatToRawIntBits (v);
		write (vv >> 24);
		write (vv >> 16);
		write (vv >> 8);
		write (vv);
		}

	/**
	 * Write the given double value to this out stream. Eight bytes are written.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readDouble()}.
	 *
	 * @param  v  Double value.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeDouble
		(double v)
		throws IOException
		{
		long vv = Double.doubleToRawLongBits (v);
		write ((int)(vv >> 56));
		write ((int)(vv >> 48));
		write ((int)(vv >> 40));
		write ((int)(vv >> 32));
		write ((int)(vv >> 24));
		write ((int)(vv >> 16));
		write ((int)(vv >> 8));
		write ((int)(vv));
		}

	/**
	 * Write the given string value to this out stream. The length of the string
	 * is written using <TT>writeInt()</TT>, then each character of the string
	 * is written using <TT>writeChar()</TT>. If <TT>v</TT> is null, then a
	 * length of &minus;1 is written.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readString()}.
	 *
	 * @param  v  String value, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeString
		(String v)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length();
			writeInt (n);
			for (int i = 0; i < n; ++ i)
				writeChar (v.charAt (i));
			}
		}

	/**
	 * Write the given object's fields to this out stream. The fields are
	 * written by calling <TT>v.</TT>{@link Streamable#writeOut(OutStream)
	 * writeOut()}. <TT>v</TT> must not be null.
	 * <P>
	 * To read the value, the reader must call {@link
	 * InStream#readFields(Streamable) InStream.readFields()}.
	 *
	 * @param  v  Object.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>v</TT> is null.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeFields
		(Streamable v)
		throws IOException
		{
		v.writeOut (this);
		}

	/**
	 * Write the given object to this out stream. If <TT>v</TT> is null, a 0
	 * byte is written. Otherwise, a nonzero index designating the object's
	 * class is written using <TT>writeUnsignedInt()</TT>; if this is the first
	 * occurrence of the class, the class name is written using
	 * <TT>writeString()</TT>; and either (a) for a streamable object, the
	 * object's fields are written by calling <TT>writeFields()</TT>, or (b) for
	 * a serializable object, the object is converted to a byte array using Java
	 * Object Serialization and the byte array is written.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readObject()}.
	 * <P>
	 * <I>Note:</I> If the same object is written multiple times using
	 * <TT>writeObject()</TT>, the reader will end up with multiple
	 * <I>different</I> objects (that are copies of each other). If the reader
	 * needs to end up with multiple references to the <I>same</I> object, use
	 * {@link #writeReference(Object) writeReference()}.
	 *
	 * @param  v  Object, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeObject
		(Object v)
		throws IOException
		{
		if (v == null)
			writeUnsignedInt (0);
		else
			{
			Class<?> c = v.getClass();
			writeClass (c);
			if (Streamable.class.isAssignableFrom (c))
				{
				writeFields ((Streamable) v);
				}
			else
				{
				ByteArrayOutputStream baos = new ByteArrayOutputStream();
				ObjectOutputStream oos = new ObjectOutputStream (baos);
				oos.writeObject (v);
				oos.close();
				writeByteArray (baos.toByteArray());
				}
			}
		}

	/**
	 * Write a reference to the given object to this out stream. If <TT>v</TT>
	 * is null, a 0 byte is written. Otherwise, a nonzero index designating the
	 * object reference is written using <TT>writeUnsignedInt()</TT>; if this is
	 * the first occurrence of the object, the object is written using
	 * <TT>writeObject()</TT>, and the object is saved in a cache so that later
	 * calls to <TT>writeReference()</TT> will write just the reference index.
	 * To clear the cache, call {@link #clearCache()}.
	 * <P>
	 * To read the value, the reader must call {@link InStream#readReference()}.
	 * <P>
	 * <I>Note:</I> If the same object is written multiple times using
	 * <TT>writeReference()</TT>, the reader will end up with multiple
	 * references to the <I>same</I> object. If the reader needs to end up with
	 * multiple <I>different</I> objects (that are copies of each other), use
	 * {@link #writeObject(Object) writeObject()}.
	 *
	 * @param  v  Object, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeReference
		(Object v)
		throws IOException
		{
		if (v == null)
			writeUnsignedInt (0);
		else
			{
			int i = indexForObject (v);
			if (i == 0)
				{
				i = addObjectToCache (v);
				writeUnsignedInt (i);
				writeObject (v);
				}
			else
				{
				writeUnsignedInt (i);
				}
			}
		}

	/**
	 * Clear the cache of objects that have been written using {@link
	 * #writeReference(Object) writeReference()}. Afterwards, this out stream
	 * will act as though no objects had been previously written using {@link
	 * #writeReference(Object) writeReference()}.
	 * <P>
	 * The reader must call {@link InStream#clearCache()} at the same point as
	 * the writer called <TT>clearCache()</TT>.
	 */
	public void clearCache()
		{
		objectToIndexMap.clear();
		}

	/**
	 * Write the given Boolean array to this out stream. The length of the array
	 * is written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeBoolean()</TT>. If <TT>v</TT> is null, then a
	 * length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readBooleanArray()}, {@link
	 * InStream#readBooleanArray(boolean[])}, or {@link
	 * InStream#readBooleanArray(boolean[],int,int)}.
	 *
	 * @param  v  Boolean array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeBooleanArray
		(boolean[] v)
		throws IOException
		{
		writeBooleanArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given Boolean array to this out stream. The length
	 * of the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeBoolean()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readBooleanArray()},
	 * {@link InStream#readBooleanArray(boolean[])}, or
	 * {@link InStream#readBooleanArray(boolean[],int,int)}.
	 *
	 * @param  v    Boolean array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeBooleanArray
		(boolean[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeBooleanArray(): Index out of bounds, v=boolean[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeBoolean (v[i]);
			}
		}

	/**
	 * Write the given byte array to this out stream. The length of the array is
	 * written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeByte()</TT>. If <TT>v</TT> is null, then a length
	 * of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readByteArray()},
	 * {@link InStream#readByteArray(byte[])}, or
	 * {@link InStream#readByteArray(byte[],int,int)}.
	 *
	 * @param  v  Byte array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeByteArray
		(byte[] v)
		throws IOException
		{
		writeByteArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given byte array to this out stream. The length of
	 * the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeByte()</TT>. If <TT>v</TT> is null,
	 * then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readByteArray()},
	 * {@link InStream#readByteArray(byte[])}, or
	 * {@link InStream#readByteArray(byte[],int,int)}.
	 *
	 * @param  v    Byte array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeByteArray
		(byte[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeByteArray(): Index out of bounds, v=byte[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeByte (v[i]);
			}
		}

	/**
	 * Write the given unsigned byte array to this out stream. The length of the
	 * array is written using <TT>writeInt()</TT>, then each element of the
	 * array is written using <TT>writeUnsignedByte()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readUnsignedByteArray()}, {@link
	 * InStream#readUnsignedByteArray(byte[])}, or {@link
	 * InStream#readUnsignedByteArray(byte[],int,int)}.
	 *
	 * @param  v  Unsigned byte array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedByteArray
		(byte[] v)
		throws IOException
		{
		writeByteArray (v);
		}

	/**
	 * Write a portion of the given unsigned byte array to this out stream. The
	 * length of the portion is written using <TT>writeInt()</TT>, then each
	 * element of the portion is written using <TT>writeUnsignedByte()</TT>. If
	 * <TT>v</TT> is null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readUnsignedByteArray()}, {@link
	 * InStream#readUnsignedByteArray(byte[])}, or {@link
	 * InStream#readUnsignedByteArray(byte[],int,int)}.
	 *
	 * @param  v    Unsigned byte array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedByteArray
		(byte[] v,
		 int off,
		 int len)
		throws IOException
		{
		writeByteArray (v, off, len);
		}

	/**
	 * Write the given short array to this out stream. The length of the array
	 * is written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeShort()</TT>. If <TT>v</TT> is null, then a length
	 * of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readShortArray()}, {@link InStream#readShortArray(short[])}, or
	 * {@link InStream#readShortArray(short[],int,int)}.
	 *
	 * @param  v  Short array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeShortArray
		(short[] v)
		throws IOException
		{
		writeShortArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given short array to this out stream. The length
	 * of the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeShort()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readShortArray()}, {@link InStream#readShortArray(short[])}, or
	 * {@link InStream#readShortArray(short[],int,int)}.
	 *
	 * @param  v    Short array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeShortArray
		(short[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeShortArray(): Index out of bounds, v=short[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeShort (v[i]);
			}
		}

	/**
	 * Write the given unsigned short array to this out stream. The length of
	 * the array is written using <TT>writeInt()</TT>, then each element of the
	 * array is written using <TT>writeUnsignedShort()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readUnsignedShortArray()}, {@link
	 * InStream#readUnsignedShortArray(short[])}, or {@link
	 * InStream#readUnsignedShortArray(short[],int,int)}.
	 *
	 * @param  v  Unsigned short array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedShortArray
		(short[] v)
		throws IOException
		{
		writeUnsignedShortArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given unsigned short array to this out stream. The
	 * length of the portion is written using <TT>writeInt()</TT>, then each
	 * element of the portion is written using <TT>writeUnsignedShort()</TT>. If
	 * <TT>v</TT> is null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readUnsignedShortArray()}, {@link
	 * InStream#readUnsignedShortArray(short[])}, or {@link
	 * InStream#readUnsignedShortArray(short[],int,int)}.
	 *
	 * @param  v    Unsigned short array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedShortArray
		(short[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeUnsignedShortArray(): Index out of bounds, v=short[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeUnsignedShort (v[i]);
			}
		}

	/**
	 * Write the given character array to this out stream. The length of the
	 * array is written using <TT>writeInt()</TT>, then each element of the
	 * array is written using <TT>writeChar()</TT>. If <TT>v</TT> is null, then
	 * a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readCharArray()},
	 * {@link InStream#readCharArray(char[])}, or {@link
	 * InStream#readCharArray(char[],int,int)}.
	 *
	 * @param  v  Character array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeCharArray
		(char[] v)
		throws IOException
		{
		writeCharArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given character array to this out stream. The
	 * length of the portion is written using <TT>writeInt()</TT>, then each
	 * element of the portion is written using <TT>writeChar()</TT>. If
	 * <TT>v</TT> is null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readCharArray()}, {@link InStream#readCharArray(char[])}, or
	 * {@link InStream#readCharArray(char[],int,int)}.
	 *
	 * @param  v    Character array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeCharArray
		(char[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeCharArray(): Index out of bounds, v=char[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeChar (v[i]);
			}
		}

	/**
	 * Write the given integer array to this out stream. The length of the array
	 * is written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeInt()</TT>. If <TT>v</TT> is null, then a length
	 * of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readIntArray()},
	 * {@link InStream#readIntArray(int[])}, or {@link
	 * InStream#readIntArray(int[],int,int)}.
	 *
	 * @param  v  Integer array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeIntArray
		(int[] v)
		throws IOException
		{
		writeIntArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given integer array to this out stream. The length
	 * of the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeInt()</TT>. If <TT>v</TT> is null,
	 * then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readIntArray()},
	 * {@link InStream#readIntArray(int[])}, or {@link
	 * InStream#readIntArray(int[],int,int)}.
	 *
	 * @param  v    Integer array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeIntArray
		(int[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeIntArray(): Index out of bounds, v=int[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeInt (v[i]);
			}
		}

	/**
	 * Write the given unsigned integer array to this out stream. The length of
	 * the array is written using <TT>writeInt()</TT>, then each element of the
	 * array is written using <TT>writeUnsignedInt()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readUnsignedIntArray()}, {@link
	 * InStream#readUnsignedIntArray(int[])}, or {@link
	 * InStream#readUnsignedIntArray(int[],int,int)}.
	 *
	 * @param  v  Unsigned integer array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedIntArray
		(int[] v)
		throws IOException
		{
		writeUnsignedIntArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given unsigned integer array to this out stream.
	 * The length of the portion is written using <TT>writeInt()</TT>, then each
	 * element of the portion is written using <TT>writeUnsignedInt()</TT>. If
	 * <TT>v</TT> is null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readUnsignedIntArray()},
	 * {@link InStream#readUnsignedIntArray(int[])}, or {@link
	 * InStream#readUnsignedIntArray(int[],int,int)}.
	 *
	 * @param  v    Unsigned integer array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedIntArray
		(int[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeUnsignedIntArray(): Index out of bounds, v=int[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeUnsignedInt (v[i]);
			}
		}

	/**
	 * Write the given long array to this out stream. The length of the array is
	 * written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeLong()</TT>. If <TT>v</TT> is null, then a length
	 * of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readLongArray()},
	 * {@link InStream#readLongArray(long[])}, or {@link
	 * InStream#readLongArray(long[],int,int)}.
	 *
	 * @param  v  Long array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeLongArray
		(long[] v)
		throws IOException
		{
		writeLongArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given long array to this out stream. The length of
	 * the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeLong()</TT>. If <TT>v</TT> is null,
	 * then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readLongArray()},
	 * {@link InStream#readLongArray(long[])}, or {@link
	 * InStream#readLongArray(long[],int,int)}.
	 *
	 * @param  v    Long array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeLongArray
		(long[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeLongArray(): Index out of bounds, v=long[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeLong (v[i]);
			}
		}

	/**
	 * Write the given unsigned long array to this out stream. The length of the
	 * array is written using <TT>writeInt()</TT>, then each element of the
	 * array is written using <TT>writeUnsignedLong()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readUnsignedLongArray()}, {@link
	 * InStream#readUnsignedLongArray(long[])}, or {@link
	 * InStream#readUnsignedLongArray(long[],int,int)}.
	 *
	 * @param  v  Unsigned long array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedLongArray
		(long[] v)
		throws IOException
		{
		writeUnsignedLongArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given unsigned long array to this out stream. The
	 * length of the portion is written using <TT>writeInt()</TT>, then each
	 * element of the portion is written using <TT>writeUnsignedLong()</TT>. If
	 * <TT>v</TT> is null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link InStream#readUnsignedLongArray()},
	 * {@link InStream#readUnsignedLongArray(long[])}, or {@link
	 * InStream#readUnsignedLongArray(long[],int,int)}.
	 *
	 * @param  v    Unsigned long array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeUnsignedLongArray
		(long[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeUnsignedLongArray(): Index out of bounds, v=long[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeUnsignedLong (v[i]);
			}
		}

	/**
	 * Write the given float array to this out stream. The length of the array
	 * is written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeFloat()</TT>. If <TT>v</TT> is null, then a length
	 * of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readFloatArray()}, {@link InStream#readFloatArray(float[])}, or
	 * {@link InStream#readFloatArray(float[],int,int)}.
	 *
	 * @param  v  Float array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeFloatArray
		(float[] v)
		throws IOException
		{
		writeFloatArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given float array to this out stream. The length
	 * of the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeFloat()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readFloatArray()}, {@link InStream#readFloatArray(float[])}, or
	 * {@link InStream#readFloatArray(float[],int,int)}.
	 *
	 * @param  v    Float array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeFloatArray
		(float[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeFloatArray(): Index out of bounds, v=float[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeFloat (v[i]);
			}
		}

	/**
	 * Write the given double array to this out stream. The length of the array
	 * is written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeDouble()</TT>. If <TT>v</TT> is null, then a
	 * length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readDoubleArray()}, {@link InStream#readDoubleArray(double[])},
	 * or {@link InStream#readDoubleArray(double[],int,int)}.
	 *
	 * @param  v  Double array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeDoubleArray
		(double[] v)
		throws IOException
		{
		writeDoubleArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given double array to this out stream. The length
	 * of the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeDouble()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readDoubleArray()}, {@link InStream#readDoubleArray(double[])},
	 * or {@link InStream#readDoubleArray(double[],int,int)}.
	 *
	 * @param  v    Double array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeDoubleArray
		(double[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeDoubleArray(): Index out of bounds, v=double[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeDouble (v[i]);
			}
		}

	/**
	 * Write the given string array to this out stream. The length of the array
	 * is written using <TT>writeInt()</TT>, then each element of the array is
	 * written using <TT>writeString()</TT>. If <TT>v</TT> is null, then a
	 * length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readStringArray()}, {@link InStream#readStringArray(String[])},
	 * or {@link InStream#readStringArray(String[],int,int)}.
	 *
	 * @param  v  String array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeStringArray
		(String[] v)
		throws IOException
		{
		writeStringArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given string array to this out stream. The length
	 * of the portion is written using <TT>writeInt()</TT>, then each element of
	 * the portion is written using <TT>writeString()</TT>. If <TT>v</TT> is
	 * null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readStringArray()}, {@link InStream#readStringArray(String[])},
	 * or {@link InStream#readStringArray(String[],int,int)}.
	 *
	 * @param  v    String array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeStringArray
		(String[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeStringArray(): Index out of bounds, v=String[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeString (v[i]);
			}
		}

	/**
	 * Write the given object array to this out stream. The length of the array
	 * is written using <TT>writeInt()</TT>, the array element class is written,
	 * then each element of the array is written using <TT>writeObject()</TT>.
	 * If <TT>v</TT> is null, then a length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readObjectArray()}, {@link
	 * InStream#readObjectArray(Object[])}, or {@link
	 * InStream#readObjectArray(Object[],int,int)}.
	 *
	 * @param  v  Object array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeObjectArray
		(Object[] v)
		throws IOException
		{
		writeObjectArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given object array to this out stream. The length
	 * of the portion is written using <TT>writeInt()</TT>, the array element
	 * class is written, then each element of the portion is written using
	 * <TT>writeObject()</TT>. If <TT>v</TT> is null, then a length of &minus;1
	 * is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readObjectArray()}, {@link
	 * InStream#readObjectArray(Object[])}, or {@link
	 * InStream#readObjectArray(Object[],int,int)}.
	 *
	 * @param  v    Object array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeObjectArray
		(Object[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeObjectArray(): Index out of bounds, v=Object[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			writeClass (v.getClass().getComponentType());
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeObject (v[i]);
			}
		}

	/**
	 * Write the given array of object references to this out stream. The length
	 * of the array is written using <TT>writeInt()</TT>, the array element
	 * class is written, then each element of the array is written using
	 * <TT>writeReference()</TT>. If <TT>v</TT> is null, then a length of
	 * &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readReferenceArray()}, {@link
	 * InStream#readReferenceArray(Object[])}, or {@link
	 * InStream#readReferenceArray(Object[],int,int)}.
	 *
	 * @param  v  Object array, or null.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeReferenceArray
		(Object[] v)
		throws IOException
		{
		writeReferenceArray (v, 0, v == null ? 0 : v.length);
		}

	/**
	 * Write a portion of the given array of object references to this out
	 * stream. The length of the portion is written using <TT>writeInt()</TT>,
	 * the array element class is written, then each element of the portion is
	 * written using <TT>writeReference()</TT>. If <TT>v</TT> is null, then a
	 * length of &minus;1 is written.
	 * <P>
	 * To read the array, the reader must call {@link
	 * InStream#readReferenceArray()}, {@link
	 * InStream#readReferenceArray(Object[])}, or {@link
	 * InStream#readReferenceArray(Object[],int,int)}.
	 *
	 * @param  v    Object array, or null.
	 * @param  off  Index of first element to write.
	 * @param  len  Number of elements to write.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>v</TT> is non-null and
	 *     <TT>off</TT> &lt; 0, <TT>len</TT> &lt; 0, or <TT>off+len</TT> &gt;
	 *     <TT>v.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeReferenceArray
		(Object[] v,
		 int off,
		 int len)
		throws IOException
		{
		if (v == null)
			writeInt (-1);
		else
			{
			int n = v.length;
			if (off < 0 || len < 0 || off + len > n)
				throw new IndexOutOfBoundsException (String.format
					("OutStream.writeReferenceArray(): Index out of bounds, v=Object[%d], off=%d, len=%d",
					 n, off, len));
			writeInt (len);
			writeClass (v.getClass().getComponentType());
			n = off + len;
			for (int i = off; i < n; ++ i)
				writeReference (v[i]);
			}
		}

	/**
	 * Flush this out stream to the underlying output stream. Any buffered bytes
	 * are written to the underlying output stream, and the underlying output
	 * stream is flushed.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void flush()
		throws IOException
		{
		if (buflen > 0) writeBuf();
		out.flush();
		}

	/**
	 * Close this out stream. Any buffered bytes are written to the underlying
	 * output stream, and the underlying output stream is flushed and closed.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void close()
		throws IOException
		{
		try
			{
			flush();
			out.close();
			}
		finally
			{
			out = null;
			buf = null;
			buflen = 0;
			classToIndexMap = null;
			objectToIndexMap = null;
			}
		}

	/**
	 * Finalize this out stream.
	 */
	protected void finalize()
		throws Throwable
		{
		close();
		}

	}
