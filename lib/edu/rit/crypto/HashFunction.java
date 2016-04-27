//******************************************************************************
//
// File:    HashFunction.java
// Package: edu.rit.crypto
// Unit:    Interface edu.rit.crypto.HashFunction
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

package edu.rit.crypto;

/**
 * Interface HashFunction specifies the interface for a cryptographic hash
 * function.
 *
 * @author  Alan Kaminsky
 * @version 16-Jul-2013
 */
public interface HashFunction
	{

	/**
	 * Returns this hash function's digest size in bytes.
	 *
	 * @return  Digest size.
	 */
	public int digestSize();

	/**
	 * Append the given byte to the message being hashed. Only the least
	 * significant 8 bits of <TT>b</TT> are used.
	 *
	 * @param  b  Message byte.
	 */
	public void hash
		(int b);

	/**
	 * Append the given byte array to the message being hashed.
	 *
	 * @param  buf  Array of message bytes.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>buf</TT> is null.
	 */
	public void hash
		(byte[] buf);

	/**
	 * Append a portion of the given byte array to the message being hashed.
	 *
	 * @param  buf  Array of message bytes.
	 * @param  off  Index of first message byte to hash.
	 * @param  len  Number of message bytes to hash.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>buf</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>buf.length</TT>.
	 */
	public void hash
		(byte[] buf,
		 int off,
		 int len);

	/**
	 * Obtain the message digest. <TT>digest</TT> must be an array of bytes
	 * whose length is equal to <TT>digestSize()</TT>. The message consists of
	 * the series of bytes provided to the <TT>hash()</TT> methods. The digest
	 * of the message is stored in the <TT>digest</TT> array. Afterwards, this
	 * hash function is reset.
	 *
	 * @param  digest  Message digest (output).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>digest</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>digest.length</TT> &ne;
	 *     <TT>digestSize()</TT>.
	 */
	public void digest
		(byte[] digest);

	/**
	 * Reset this hash function. Any accumulated message bytes are discarded,
	 * and the digest computation starts afresh.
	 */
	public void reset();

	}
