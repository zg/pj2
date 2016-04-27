//******************************************************************************
//
// File:    RC4.java
// Package: edu.rit.crypto
// Unit:    Class edu.rit.crypto.RC4
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

package edu.rit.crypto;

/**
 * Class RC4 provides an object that computes the RC4 stream cipher.
 *
 * @author  Alan Kaminsky
 * @version 20-Jan-2014
 */
public class RC4
	implements StreamCipher
	{

// Kludge to avert false sharing in multithreaded programs.

	// Padding fields.
	volatile long p0 = 1000L;
	volatile long p1 = 1001L;
	volatile long p2 = 1002L;
	volatile long p3 = 1003L;
	volatile long p4 = 1004L;
	volatile long p5 = 1005L;
	volatile long p6 = 1006L;
	volatile long p7 = 1007L;
	volatile long p8 = 1008L;
	volatile long p9 = 1009L;
	volatile long pa = 1010L;
	volatile long pb = 1011L;
	volatile long pc = 1012L;
	volatile long pd = 1013L;
	volatile long pe = 1014L;
	volatile long pf = 1015L;

	// Method to prevent the JDK from optimizing away the padding fields.
	long preventOptimization()
		{
		return p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 +
			p8 + p9 + pa + pb + pc + pd + pe + pf;
		}

// Hidden data members.

	private byte[] S = new byte [384]; // 256 + 128 bytes padding
	private int x;
	private int y;

// Exported constructors.

	/**
	 * Construct a new RC4 stream cipher object and set the key. <TT>key</TT>
	 * must be an array of bytes whose length is equal to <TT>keySize()</TT>.
	 * Class RC4 supports a fixed key size of 16 bytes (128 bits). The keystream
	 * generator is initialized, such that successive calls to
	 * <TT>encrypt()</TT> will encrypt or decrypt a series of bytes.
	 *
	 * @param  key  Key.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>key.length</TT> &ne;
	 *     <TT>keySize()</TT>.
	 */
	public RC4
		(byte[] key)
		{
		setKey (key);
		}

// Exported operations.

	/**
	 * Returns this stream cipher's key size in bytes. Class RC4 supports a
	 * fixed key size of 16 bytes (128 bits).
	 *
	 * @return  Key size.
	 */
	public int keySize()
		{
		return 16;
		}

	/**
	 * Set the key for this stream cipher. <TT>key</TT> must be an array of
	 * bytes whose length is equal to <TT>keySize()</TT>. Class RC4 supports a
	 * fixed key size of 16 bytes (128 bits). The keystream generator is
	 * initialized, such that successive calls to <TT>encrypt()</TT> will
	 * encrypt or decrypt a series of bytes.
	 *
	 * @param  key  Key.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>key</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>key.length</TT> &ne;
	 *     <TT>keySize()</TT>.
	 */
	public void setKey
		(byte[] key)
		{
		if (key.length != 16)
			throw new IndexOutOfBoundsException
				("RC4.setKey(): key.length != 16");

		for (int i = 0; i <= 255; ++ i)
			S[i] = (byte)i;
		int j = 0;
		for (int i = 0; i <= 255; ++ i)
			{
			j = (j + S[i] + key[i & 15]) & 255;
			swap (i, j);
			}
		x = 0;
		y = 0;
		}

	/**
	 * Encrypt or decrypt the given byte. Only the least significant 8 bits of
	 * <TT>b</TT> are used. If <TT>b</TT> is a plaintext byte, the ciphertext
	 * byte is returned as a value from 0 to 255. If <TT>b</TT> is a ciphertext
	 * byte, the plaintext byte is returned as a value from 0 to 255.
	 *
	 * @param  b  Plaintext byte (if encrypting), ciphertext byte (if
	 *            decrypting).
	 *
	 * @return  Ciphertext byte (if encrypting), plaintext byte (if decrypting).
	 */
	public int encrypt
		(int b)
		{
		x = (x + 1) & 255;
		y = (y + S[x]) & 255;
		swap (x, y);
		return (S[(S[x] + S[y]) & 255] & 255) ^ b;
		}

// Hidden operations.

	/**
	 * Swap S[i] with S[j].
	 */
	private void swap
		(int i,
		 int j)
		{
		byte t = S[i];
		S[i] = S[j];
		S[j] = t;
		}

	}
