//******************************************************************************
//
// File:    CipherInputStream.java
// Package: edu.rit.crypto
// Unit:    Class edu.rit.crypto.CipherInputStream
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

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Class CipherInputStream provides a {@linkplain java.io.FilterInputStream
 * FilterInputStream} that passes all bytes read through a {@linkplain
 * StreamCipher StreamCipher}. If the bytes read from the underlying input
 * stream are plaintext, the bytes read from the cipher input stream are
 * ciphertext (encryption). If the bytes read from the underlying input stream
 * are ciphertext, the bytes read from the cipher input stream are plaintext
 * (decryption).
 *
 * @author  Alan Kaminsky
 * @version 09-Aug-2013
 */
public class CipherInputStream
	extends FilterInputStream
	{

// Hidden data members.

	private StreamCipher cipher;

// Exported constructors.

	/**
	 * Construct a new cipher input stream.
	 *
	 * @param  in      Underlying input stream.
	 * @param  cipher  Stream cipher.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>in</TT> is null or
	 *     <TT>cipher</TT> is null.
	 */
	public CipherInputStream
		(InputStream in,
		 StreamCipher cipher)
		{
		super (validate (in));
		this.cipher = validate (cipher);
		}

	private static InputStream validate
		(InputStream in)
		{
		if (in == null)
			throw new NullPointerException
				("CipherInputStream(): in is null");
		return in;
		}

	private static StreamCipher validate
		(StreamCipher cipher)
		{
		if (cipher == null)
			throw new NullPointerException
				("CipherInputStream(): cipher is null");
		return cipher;
		}

// Exported operations.

	/**
	 * Read the next byte from this cipher input stream. If the end of stream
	 * was not encountered, an <TT>int</TT> in the range 0 through 255 is
	 * returned. If the end of stream was encountered, &minus;1 is returned.
	 *
	 * @return  Plaintext byte (if encrypting), ciphertext byte (if decrypting),
	 *          or &minus;1 (if end of stream).
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public int read()
		throws IOException
		{
		int b = in.read();
		return b == -1 ? -1 : cipher.encrypt (b);
		}

	/**
	 * Read bytes from this cipher input stream into a portion of an array. If
	 * <TT>len</TT> &gt; 0, then this method blocks until input is available or
	 * the end of stream is encountered. If <TT>len</TT> = 0, then this method
	 * does nothing and returns 0.
	 *
	 * @param  buf  Buffer.
	 * @param  off  Index of first byte to store in buffer.
	 * @param  len  Maximum number of bytes to store in buffer.
	 *
	 * @return  The number of bytes actually read, or &minus;1 if the end of
	 *          stream was encountered.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>buf</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off+len</TT> &gt; <TT>buf.length</TT>.
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public int read
		(byte[] buf,
		 int off,
		 int len)
		throws IOException
		{
		int n = in.read (buf, off, len);
		for (int i = 0; i < n; ++ i)
			buf[off+i] = (byte) cipher.encrypt (buf[off+i]);
		return n;
		}

	/**
	 * Class CipherInputStream does not support the <TT>skip()</TT> operation.
	 * This method throws an UnsupportedOperationException.
	 */
	public long skip
		(long n)
		{
		throw new UnsupportedOperationException
			("CipherInputStream.skip(): Skip not supported");
		}

	/**
	 * Class CipherInputStream does not support the <TT>mark()</TT> operation.
	 * This method throws an UnsupportedOperationException.
	 */
	public void mark
		(int readlimit)
		{
		throw new UnsupportedOperationException
			("CipherInputStream.mark(): Mark not supported");
		}

	/**
	 * Class CipherInputStream does not support the <TT>reset()</TT> operation.
	 * This method throws an UnsupportedOperationException.
	 */
	public void reset()
		{
		throw new UnsupportedOperationException
			("CipherInputStream.reset(): Reset not supported");
		}

	/**
	 * Determine if this cipher input stream supports the <TT>mark()</TT> and
	 * <TT>reset()</TT> operations. (It does not.)
	 *
	 * @return  False.
	 */
	public boolean markSupported()
		{
		return false;
		}

	}
