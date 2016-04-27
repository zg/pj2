//******************************************************************************
//
// File:    Packing.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Packing
//
// This Java source file is copyright (C) 2013 by Alan Kaminsky. All rights
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

/**
 * Class Packing provides static methods for packing and unpacking arrays of
 * bytes, short integers, and integers into and out of short integers, integers,
 * long integers, and arrays thereof.
 * <P>
 * <I>Note:</I> The operations in class Packing are not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 21-Aug-2013
 */
public class Packing
	{

// Prevent construction.

	private Packing()
		{
		}

//*****************************************************************************
// Exported operations for byte-to-short packing.

	/**
	 * Pack bytes from the given array into a short integer in little-endian
	 * order.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+1]</TT>
	 *          packed into a short integer in little-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static short packShortLittleEndian
		(byte[] src,
		 int srcPos)
		{
		if (srcPos + 2 > src.length) throw new IndexOutOfBoundsException();
		short rv = (short) 0;
		for (int i = 0; i <= 1; ++ i)
			rv |= (src[srcPos+i] & 0xFF) << (i*8);
		return rv;
		}

	/**
	 * Pack bytes from the given array into a short integer in big-endian order.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+1]</TT>
	 *          packed into a short integer in big-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static short packShortBigEndian
		(byte[] src,
		 int srcPos)
		{
		if (srcPos + 2 > src.length) throw new IndexOutOfBoundsException();
		short rv = (short) 0;
		for (int i = 0; i <= 1; ++ i)
			rv |= (src[srcPos+i] & 0xFF) << ((1 - i)*8);
		return rv;
		}

	/**
	 * Pack bytes from the given array into the given array of short integers in
	 * little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+2*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 * @param  dst     Destination array of packed short integers.
	 * @param  dstPos  Index of first packed short integer.
	 * @param  len     Number of short integers (not bytes!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packShortLittleEndian
		(byte[] src,
		 int srcPos,
		 short[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 2*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packShortLittleEndian (src, srcPos + 2*i);
		}

	/**
	 * Pack bytes from the given array into the given array of short integers in
	 * big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+2*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 * @param  dst     Destination array of packed short integers.
	 * @param  dstPos  Index of first packed integer.
	 * @param  len     Number of short integers (not bytes!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packShortBigEndian
		(byte[] src,
		 int srcPos,
		 short[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 2*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packShortBigEndian (src, srcPos + 2*i);
		}

//*****************************************************************************
// Exported operations for byte-to-int packing.

	/**
	 * Pack bytes from the given array into an integer in little-endian order.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+3]</TT>
	 *          packed into an integer in little-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static int packIntLittleEndian
		(byte[] src,
		 int srcPos)
		{
		if (srcPos + 4 > src.length) throw new IndexOutOfBoundsException();
		int rv = 0;
		for (int i = 0; i <= 3; ++ i)
			rv |= (src[srcPos+i] & 0xFF) << (i*8);
		return rv;
		}

	/**
	 * Pack bytes from the given array into an integer in big-endian order.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+3]</TT>
	 *          packed into an integer in big-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static int packIntBigEndian
		(byte[] src,
		 int srcPos)
		{
		if (srcPos + 4 > src.length) throw new IndexOutOfBoundsException();
		int rv = 0;
		for (int i = 0; i <= 3; ++ i)
			rv |= (src[srcPos+i] & 0xFF) << ((3 - i)*8);
		return rv;
		}

	/**
	 * Pack bytes from the given array into the given array of integers in
	 * little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+4*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 * @param  dst     Destination array of packed integers.
	 * @param  dstPos  Index of first packed integer.
	 * @param  len     Number of integers (not bytes!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packIntLittleEndian
		(byte[] src,
		 int srcPos,
		 int[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 4*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packIntLittleEndian (src, srcPos + 4*i);
		}

	/**
	 * Pack bytes from the given array into the given array of integers in
	 * big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+4*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 * @param  dst     Destination array of packed integers.
	 * @param  dstPos  Index of first packed integer.
	 * @param  len     Number of integers (not bytes!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packIntBigEndian
		(byte[] src,
		 int srcPos,
		 int[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 4*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packIntBigEndian (src, srcPos + 4*i);
		}

//*****************************************************************************
// Exported operations for byte-to-long packing.

	/**
	 * Pack bytes from the given array into a long integer in little-endian
	 * order.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+7]</TT>
	 *          packed into a long integer in little-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static long packLongLittleEndian
		(byte[] src,
		 int srcPos)
		{
		if (srcPos + 8 > src.length) throw new IndexOutOfBoundsException();
		long rv = 0L;
		for (int i = 0; i <= 7; ++ i)
			rv |= (src[srcPos+i] & 0xFFL) << (i*8);
		return rv;
		}

	/**
	 * Pack bytes from the given array into a long integer in big-endian order.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+7]</TT>
	 *          packed into a long integer in big-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static long packLongBigEndian
		(byte[] src,
		 int srcPos)
		{
		if (srcPos + 8 > src.length) throw new IndexOutOfBoundsException();
		long rv = 0L;
		for (int i = 0; i <= 7; ++ i)
			rv |= (src[srcPos+i] & 0xFFL) << ((7 - i)*8);
		return rv;
		}

	/**
	 * Pack bytes from the given array into the given array of long integers in
	 * little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+8*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 * @param  dst     Destination array of packed long integers.
	 * @param  dstPos  Index of first packed long integer.
	 * @param  len     Number of long integers (not bytes!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packLongLittleEndian
		(byte[] src,
		 int srcPos,
		 long[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 8*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packLongLittleEndian (src, srcPos + 8*i);
		}

	/**
	 * Pack bytes from the given array into the given array of long integers in
	 * big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+8*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of bytes to pack.
	 * @param  srcPos  Index of first byte to pack.
	 * @param  dst     Destination array of packed long integers.
	 * @param  dstPos  Index of first packed long integer.
	 * @param  len     Number of long integers (not bytes!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packLongBigEndian
		(byte[] src,
		 int srcPos,
		 long[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 8*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packLongBigEndian (src, srcPos + 8*i);
		}

//*****************************************************************************
// Exported operations for short-to-int packing.

	/**
	 * Pack short integers from the given array into an integer in little-endian
	 * order.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+1]</TT>
	 *          packed into an integer in little-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static int packIntLittleEndian
		(short[] src,
		 int srcPos)
		{
		if (srcPos + 2 > src.length) throw new IndexOutOfBoundsException();
		int rv = 0;
		for (int i = 0; i <= 1; ++ i)
			rv |= (src[srcPos+i] & 0xFFFF) << (i*16);
		return rv;
		}

	/**
	 * Pack short integers from the given array into an integer in big-endian
	 * order.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+1]</TT>
	 *          packed into an integer in big-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static int packIntBigEndian
		(short[] src,
		 int srcPos)
		{
		if (srcPos + 2 > src.length) throw new IndexOutOfBoundsException();
		int rv = 0;
		for (int i = 0; i <= 1; ++ i)
			rv |= (src[srcPos+i] & 0xFFFF) << ((1 - i)*16);
		return rv;
		}

	/**
	 * Pack short integers from the given array into the given array of integers
	 * in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+2*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 * @param  dst     Destination array of packed integers.
	 * @param  dstPos  Index of first packed integer.
	 * @param  len     Number of integers (not short integers!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packIntLittleEndian
		(short[] src,
		 int srcPos,
		 int[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 2*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packIntLittleEndian (src, srcPos + 2*i);
		}

	/**
	 * Pack short integers from the given array into the given array of integers
	 * in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+2*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 * @param  dst     Destination array of packed integers.
	 * @param  dstPos  Index of first packed integer.
	 * @param  len     Number of integers (not short integers!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packIntBigEndian
		(short[] src,
		 int srcPos,
		 int[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 2*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packIntBigEndian (src, srcPos + 2*i);
		}

//*****************************************************************************
// Exported operations for short-to-long packing.

	/**
	 * Pack short integers from the given array into a long integer in
	 * little-endian order.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+3]</TT>
	 *          packed into a long integer in little-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static long packLongLittleEndian
		(short[] src,
		 int srcPos)
		{
		if (srcPos + 4 > src.length) throw new IndexOutOfBoundsException();
		long rv = 0;
		for (int i = 0; i <= 3; ++ i)
			rv |= (src[srcPos+i] & 0xFFFFL) << (i*16);
		return rv;
		}

	/**
	 * Pack short integers from the given array into a long integer in
	 * big-endian order.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+3]</TT>
	 *          packed into a long integer in big-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static long packLongBigEndian
		(short[] src,
		 int srcPos)
		{
		if (srcPos + 4 > src.length) throw new IndexOutOfBoundsException();
		long rv = 0;
		for (int i = 0; i <= 3; ++ i)
			rv |= (src[srcPos+i] & 0xFFFFL) << ((3 - i)*16);
		return rv;
		}

	/**
	 * Pack short integers from the given array into the given array of long
	 * integers in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+4*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 * @param  dst     Destination array of packed long integers.
	 * @param  dstPos  Index of first packed long integer.
	 * @param  len     Number of long integers (not short integers!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packLongLittleEndian
		(short[] src,
		 int srcPos,
		 long[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 4*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packLongLittleEndian (src, srcPos + 4*i);
		}

	/**
	 * Pack short integers from the given array into the given array of long
	 * integers in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+4*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of short integers to pack.
	 * @param  srcPos  Index of first short integer to pack.
	 * @param  dst     Destination array of packed long integers.
	 * @param  dstPos  Index of first packed long integer.
	 * @param  len     Number of long integers (not short integers!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packLongBigEndian
		(short[] src,
		 int srcPos,
		 long[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 4*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packLongBigEndian (src, srcPos + 4*i);
		}

//*****************************************************************************
// Exported operations for int-to-long packing.

	/**
	 * Pack integers from the given array into a long integer in little-endian
	 * order.
	 *
	 * @param  src     Source array of integers to pack.
	 * @param  srcPos  Index of first integer to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+1]</TT>
	 *          packed into a long integer in little-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static long packLongLittleEndian
		(int[] src,
		 int srcPos)
		{
		if (srcPos + 2 > src.length) throw new IndexOutOfBoundsException();
		long rv = 0;
		for (int i = 0; i <= 1; ++ i)
			rv |= (src[srcPos+i] & 0xFFFFFFFFL) << (i*32);
		return rv;
		}

	/**
	 * Pack integers from the given array into a long integer in big-endian
	 * order.
	 *
	 * @param  src     Source array of integers to pack.
	 * @param  srcPos  Index of first integer to pack.
	 *
	 * @return  Elements <TT>src[srcPos]</TT> through <TT>src[srcPos+1]</TT>
	 *          packed into a long integer in big-endian order.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds.
	 */
	public static long packLongBigEndian
		(int[] src,
		 int srcPos)
		{
		if (srcPos + 2 > src.length) throw new IndexOutOfBoundsException();
		long rv = 0;
		for (int i = 0; i <= 1; ++ i)
			rv |= (src[srcPos+i] & 0xFFFFFFFFL) << ((1 - i)*32);
		return rv;
		}

	/**
	 * Pack integers from the given array into the given array of long integers
	 * in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+2*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of integers to pack.
	 * @param  srcPos  Index of first integer to pack.
	 * @param  dst     Destination array of packed long integers.
	 * @param  dstPos  Index of first packed long integer.
	 * @param  len     Number of long integers (not integers!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packLongLittleEndian
		(int[] src,
		 int srcPos,
		 long[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 2*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packLongLittleEndian (src, srcPos + 2*i);
		}

	/**
	 * Pack integers from the given array into the given array of long integers
	 * in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+2*len-1]</TT> are packed into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+len-1]</TT>.
	 *
	 * @param  src     Source array of integers to pack.
	 * @param  srcPos  Index of first integer to pack.
	 * @param  dst     Destination array of packed long integers.
	 * @param  dstPos  Index of first packed long integer.
	 * @param  len     Number of long integers (not integers!) to pack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if packing would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void packLongBigEndian
		(int[] src,
		 int srcPos,
		 long[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + 2*len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			dst[dstPos+i] = packLongBigEndian (src, srcPos + 2*i);
		}

//*****************************************************************************
// Exported operations for short-to-byte unpacking.

	/**
	 * Unpack the given short integer into the given array of bytes in
	 * little-endian order. The short integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+1]</TT>.
	 *
	 * @param  src     Source short integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackShortLittleEndian
		(short src,
		 byte[] dst,
		 int dstPos)
		{
		if (dstPos + 2 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 1; ++ i)
			dst[dstPos+i] = (byte)(src >> (i*8));
		}

	/**
	 * Unpack the given short integer into the given array of bytes in
	 * big-endian order. The short integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+1]</TT>.
	 *
	 * @param  src     Source short integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackShortBigEndian
		(short src,
		 byte[] dst,
		 int dstPos)
		{
		if (dstPos + 2 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 1; ++ i)
			dst[dstPos+i] = (byte)(src >> ((1 - i)*8));
		}

	/**
	 * Unpack short integers from the given array into the given array of bytes
	 * in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+2*len-1]</TT>.
	 *
	 * @param  src     Source array of short integers to unpack.
	 * @param  srcPos  Index of first short integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 * @param  len     Number of short integers (not bytes!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackShortLittleEndian
		(short[] src,
		 int srcPos,
		 byte[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 2*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackShortLittleEndian (src[srcPos+i], dst, dstPos + 2*i);
		}

	/**
	 * Unpack short integers from the given array into the given array of bytes
	 * in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+2*len-1]</TT>.
	 *
	 * @param  src     Source array of short integers to unpack.
	 * @param  srcPos  Index of first short integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 * @param  len     Number of short integers (not bytes!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackShortBigEndian
		(short[] src,
		 int srcPos,
		 byte[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 2*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackShortBigEndian (src[srcPos+i], dst, dstPos + 2*i);
		}

//*****************************************************************************
// Exported operations for int-to-byte unpacking.

	/**
	 * Unpack the given integer into the given array of bytes in little-endian
	 * order. The integer is unpacked into elements <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+3]</TT>.
	 *
	 * @param  src     Source integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntLittleEndian
		(int src,
		 byte[] dst,
		 int dstPos)
		{
		if (dstPos + 4 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 3; ++ i)
			dst[dstPos+i] = (byte)(src >> (i*8));
		}

	/**
	 * Unpack the given integer into the given array of bytes in big-endian
	 * order. The integer is unpacked into elements <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+3]</TT>.
	 *
	 * @param  src     Source integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntBigEndian
		(int src,
		 byte[] dst,
		 int dstPos)
		{
		if (dstPos + 4 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 3; ++ i)
			dst[dstPos+i] = (byte)(src >> ((3 - i)*8));
		}

	/**
	 * Unpack integers from the given array into the given array of bytes in
	 * little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+4*len-1]</TT>.
	 *
	 * @param  src     Source array of integers to unpack.
	 * @param  srcPos  Index of first integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 * @param  len     Number of integers (not bytes!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntLittleEndian
		(int[] src,
		 int srcPos,
		 byte[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 4*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackIntLittleEndian (src[srcPos+i], dst, dstPos + 4*i);
		}

	/**
	 * Unpack integers from the given array into the given array of bytes in
	 * big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+4*len-1]</TT>.
	 *
	 * @param  src     Source array of integers to unpack.
	 * @param  srcPos  Index of first integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 * @param  len     Number of integers (not bytes!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntBigEndian
		(int[] src,
		 int srcPos,
		 byte[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 4*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackIntBigEndian (src[srcPos+i], dst, dstPos + 4*i);
		}

//*****************************************************************************
// Exported operations for long-to-byte unpacking.

	/**
	 * Unpack the given long integer into the given array of bytes in
	 * little-endian order. The long integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+7]</TT>.
	 *
	 * @param  src     Source long integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongLittleEndian
		(long src,
		 byte[] dst,
		 int dstPos)
		{
		if (dstPos + 8 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 7; ++ i)
			{
			dst[dstPos+i] = (byte)(src >> (i*8));
			}
		}

	/**
	 * Unpack the given long integer into the given array of bytes in big-endian
	 * order. The long integer is unpacked into elements <TT>dst[dstPos]</TT>
	 * through <TT>dst[dstPos+7]</TT>.
	 *
	 * @param  src     Source long integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongBigEndian
		(long src,
		 byte[] dst,
		 int dstPos)
		{
		if (dstPos + 8 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 7; ++ i)
			{
			dst[dstPos+i] = (byte)(src >> ((7 - i)*8));
			}
		}

	/**
	 * Unpack long integers from the given array into the given array of bytes
	 * in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+8*len-1]</TT>.
	 *
	 * @param  src     Source array of long integers to unpack.
	 * @param  srcPos  Index of first long integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 * @param  len     Number of integers (not bytes!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongLittleEndian
		(long[] src,
		 int srcPos,
		 byte[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 8*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackLongLittleEndian (src[srcPos+i], dst, dstPos + 8*i);
		}

	/**
	 * Unpack long integers from the given array into the given array of bytes
	 * in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+8*len-1]</TT>.
	 *
	 * @param  src     Source array of long integers to unpack.
	 * @param  srcPos  Index of first long integer to unpack.
	 * @param  dst     Destination array to receive unpacked bytes.
	 * @param  dstPos  Index of first unpacked byte.
	 * @param  len     Number of integers (not bytes!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongBigEndian
		(long[] src,
		 int srcPos,
		 byte[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 8*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackLongBigEndian (src[srcPos+i], dst, dstPos + 8*i);
		}

//*****************************************************************************
// Exported operations for int-to-short unpacking.

	/**
	 * Unpack the given integer into the given array of short integers in
	 * little-endian order. The integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+1]</TT>.
	 *
	 * @param  src     Source integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntLittleEndian
		(int src,
		 short[] dst,
		 int dstPos)
		{
		if (dstPos + 2 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 1; ++ i)
			dst[dstPos+i] = (short)(src >> (i*16));
		}

	/**
	 * Unpack the given integer into the given array of short integers in
	 * big-endian order. The integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+1]</TT>.
	 *
	 * @param  src     Source integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntBigEndian
		(int src,
		 short[] dst,
		 int dstPos)
		{
		if (dstPos + 2 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 1; ++ i)
			dst[dstPos+i] = (short)(src >> ((1 - i)*16));
		}

	/**
	 * Unpack integers from the given array into the given array of short
	 * integers in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+2*len-1]</TT>.
	 *
	 * @param  src     Source array of integers to unpack.
	 * @param  srcPos  Index of first integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 * @param  len     Number of integers (not short integers!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntLittleEndian
		(int[] src,
		 int srcPos,
		 short[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 2*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackIntLittleEndian (src[srcPos+i], dst, dstPos + 2*i);
		}

	/**
	 * Unpack integers from the given array into the given array of short
	 * integers in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+2*len-1]</TT>.
	 *
	 * @param  src     Source array of integers to unpack.
	 * @param  srcPos  Index of first integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 * @param  len     Number of integers (not short integers!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackIntBigEndian
		(int[] src,
		 int srcPos,
		 short[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 2*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackIntBigEndian (src[srcPos+i], dst, dstPos + 2*i);
		}

//*****************************************************************************
// Exported operations for long-to-short unpacking.

	/**
	 * Unpack the given long integer into the given array of short integers in
	 * little-endian order. The long integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+3]</TT>.
	 *
	 * @param  src     Source long integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongLittleEndian
		(long src,
		 short[] dst,
		 int dstPos)
		{
		if (dstPos + 4 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 3; ++ i)
			dst[dstPos+i] = (short)(src >> (i*16));
		}

	/**
	 * Unpack the given long integer into the given array of short integers in
	 * big-endian order. The long integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+3]</TT>.
	 *
	 * @param  src     Source long integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongBigEndian
		(long src,
		 short[] dst,
		 int dstPos)
		{
		if (dstPos + 4 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 3; ++ i)
			dst[dstPos+i] = (short)(src >> ((3 - i)*16));
		}

	/**
	 * Unpack long integers from the given array into the given array of short
	 * integers in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+4*len-1]</TT>.
	 *
	 * @param  src     Source array of long integers to unpack.
	 * @param  srcPos  Index of first long integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 * @param  len     Number of long integers (not short integers!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongLittleEndian
		(long[] src,
		 int srcPos,
		 short[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 4*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackLongLittleEndian (src[srcPos+i], dst, dstPos + 4*i);
		}

	/**
	 * Unpack long integers from the given array into the given array of short
	 * integers in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+4*len-1]</TT>.
	 *
	 * @param  src     Source array of long integers to unpack.
	 * @param  srcPos  Index of first long integer to unpack.
	 * @param  dst     Destination array to receive unpacked short integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 * @param  len     Number of long integers (not short integers!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongBigEndian
		(long[] src,
		 int srcPos,
		 short[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 4*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackLongBigEndian (src[srcPos+i], dst, dstPos + 4*i);
		}

//*****************************************************************************
// Exported operations for long-to-int unpacking.

	/**
	 * Unpack the given long integer into the given array of integers in
	 * little-endian order. The long integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+1]</TT>.
	 *
	 * @param  src     Source long integer to unpack.
	 * @param  dst     Destination array to receive unpacked integers.
	 * @param  dstPos  Index of first unpacked integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongLittleEndian
		(long src,
		 int[] dst,
		 int dstPos)
		{
		if (dstPos + 2 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 1; ++ i)
			dst[dstPos+i] = (int)(src >> (i*32));
		}

	/**
	 * Unpack the given long integer into the given array of integers in
	 * big-endian order. The long integer is unpacked into elements
	 * <TT>dst[dstPos]</TT> through <TT>dst[dstPos+1]</TT>.
	 *
	 * @param  src     Source long integer to unpack.
	 * @param  dst     Destination array to receive unpacked integers.
	 * @param  dstPos  Index of first unpacked short integer.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongBigEndian
		(long src,
		 int[] dst,
		 int dstPos)
		{
		if (dstPos + 2 > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i <= 1; ++ i)
			dst[dstPos+i] = (int)(src >> ((1 - i)*32));
		}

	/**
	 * Unpack long integers from the given array into the given array of
	 * integers in little-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+2*len-1]</TT>.
	 *
	 * @param  src     Source array of long integers to unpack.
	 * @param  srcPos  Index of first long integer to unpack.
	 * @param  dst     Destination array to receive unpacked integers.
	 * @param  dstPos  Index of first unpacked integer.
	 * @param  len     Number of long integers (not integers!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongLittleEndian
		(long[] src,
		 int srcPos,
		 int[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 2*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackLongLittleEndian (src[srcPos+i], dst, dstPos + 2*i);
		}

	/**
	 * Unpack long integers from the given array into the given array of
	 * integers in big-endian order. Elements <TT>src[srcPos]</TT> through
	 * <TT>src[srcPos+len-1]</TT> are unpacked into <TT>dst[dstPos]</TT> through
	 * <TT>dst[dstPos+2*len-1]</TT>.
	 *
	 * @param  src     Source array of long integers to unpack.
	 * @param  srcPos  Index of first long integer to unpack.
	 * @param  dst     Destination array to receive unpacked integers.
	 * @param  dstPos  Index of first unpacked integer.
	 * @param  len     Number of long integers (not integers!) to unpack.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>src</TT> is null. Thrown if
	 *     <TT>dst</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if unpacking would cause accessing array
	 *     elements out of bounds; in this case <TT>dst</TT> is not altered.
	 */
	public static void unpackLongBigEndian
		(long[] src,
		 int srcPos,
		 int[] dst,
		 int dstPos,
		 int len)
		{
		if (srcPos + len > src.length) throw new IndexOutOfBoundsException();
		if (dstPos + 2*len > dst.length) throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			unpackLongBigEndian (src[srcPos+i], dst, dstPos + 2*i);
		}

	}
