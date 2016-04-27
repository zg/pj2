//******************************************************************************
//
// File:    StreamCipher.java
// Package: edu.rit.crypto
// Unit:    Interface edu.rit.crypto.StreamCipher
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
 * Interface StreamCipher specifies the interface for a stream cipher.
 *
 * @author  Alan Kaminsky
 * @version 09-Aug-2013
 */
public interface StreamCipher
	{

	/**
	 * Returns this stream cipher's key size in bytes. If the stream cipher
	 * includes both a key and a nonce, <TT>keySize()</TT> returns the size of
	 * the key plus the nonce in bytes.
	 *
	 * @return  Key size.
	 */
	public int keySize();

	/**
	 * Set the key for this stream cipher. <TT>key</TT> must be an array of
	 * bytes whose length is equal to <TT>keySize()</TT>. If the stream cipher
	 * includes both a key and a nonce, <TT>key</TT> contains the bytes of the
	 * key followed by the bytes of the nonce. The keystream generator is
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
		(byte[] key);

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
		(int b);

	}
