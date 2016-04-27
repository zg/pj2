//******************************************************************************
//
// File:    CipherOutputStream.java
// Package: edu.rit.crypto
// Unit:    Class edu.rit.crypto.CipherOutputStream
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

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Class CipherOutputStream provides a {@linkplain java.io.FilterOutputStream
 * FilterOutputStream} that passes all bytes written through a {@linkplain
 * StreamCipher StreamCipher}. If the bytes written to the cipher output stream
 * are plaintext, the bytes written to the underlying output stream are
 * ciphertext (encryption). If the bytes written to the cipher output stream are
 * ciphertext, the bytes written to the underlying output stream are plaintext
 * (decryption).
 *
 * @author  Alan Kaminsky
 * @version 09-Aug-2013
 */
public class CipherOutputStream
	extends FilterOutputStream
	{

// Hidden data members.

	private StreamCipher cipher;

// Exported constructors.

	/**
	 * Construct a new cipher output stream.
	 *
	 * @param  out     Underlying output stream.
	 * @param  cipher  Stream cipher.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>out</TT> is null or
	 *     <TT>cipher</TT> is null.
	 */
	public CipherOutputStream
		(OutputStream out,
		 StreamCipher cipher)
		{
		super (validate (out));
		this.cipher = validate (cipher);
		}

	private static OutputStream validate
		(OutputStream out)
		{
		if (out == null)
			throw new NullPointerException
				("CipherOutputStream(): out is null");
		return out;
		}

	private static StreamCipher validate
		(StreamCipher cipher)
		{
		if (cipher == null)
			throw new NullPointerException
				("CipherOutputStream(): cipher is null");
		return cipher;
		}

// Exported operations.

	/**
	 * Write the given byte to this cipher output stream. Only the least
	 * significant 8 bits of <TT>b</TT> are used.
	 *
	 * @param  b  Plaintext byte (if encrypting), ciphertext byte (if
	 *            decrypting).
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(int b)
		throws IOException
		{
		out.write (cipher.encrypt (b));
		}

	}
