//******************************************************************************
//
// File:    SHA256.java
// Package: edu.rit.crypto
// Unit:    Class edu.rit.crypto.SHA256
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

import edu.rit.util.Hex;
import edu.rit.util.Packing;

/**
 * Class SHA256 provides an object that computes the SHA-256 cryptographic hash
 * function.
 *
 * @author  Alan Kaminsky
 * @version 20-Jan-2014
 */
public class SHA256
	implements HashFunction
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

	private byte[] messageBlock = new byte [192]; // 64 + 128 bytes padding
	private long byteCount;
	private int[] H = new int [40]; // 8 + 32 ints padding
	private int[] W = new int [96]; // 64 + 32 ints padding

// Exported constructors.

	/**
	 * Construct a new SHA-256 hash function.
	 */
	public SHA256()
		{
		reset();
		}

// Exported operations.

	/**
	 * Returns this hash function's digest size in bytes.
	 *
	 * @return  Digest size.
	 */
	public int digestSize()
		{
		return 32;
		}

	/**
	 * Append the given byte to the message being hashed. Only the least
	 * significant 8 bits of <TT>b</TT> are used.
	 *
	 * @param  b  Message byte.
	 */
	public void hash
		(int b)
		{
		hash ((byte)b);
		}

	/**
	 * Append the given byte array to the message being hashed.
	 *
	 * @param  buf  Array of message bytes.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>buf</TT> is null.
	 */
	public void hash
		(byte[] buf)
		{
		hash (buf, 0, buf.length);
		}

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
		 int len)
		{
		if (off < 0 || len < 0 || off + len > buf.length)
			throw new IndexOutOfBoundsException();
		for (int i = 0; i < len; ++ i)
			hash (buf[off+i]);
		}

	private void hash
		(byte b)
		{
		messageBlock[(int)(byteCount & 63)] = b;
		++ byteCount;
		if ((byteCount & 63) == 0)
			compress();
		}

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
		(byte[] digest)
		{
		if (digest.length != 32)
			throw new IndexOutOfBoundsException();

		// Pad the message.
		long bitlen = 8*byteCount;
		hash ((byte)0x80);
		while ((byteCount & 63) != 56)
			hash ((byte)0);
		Packing.unpackLongBigEndian (bitlen, messageBlock, 56);

		// One final compression.
		compress();

		// Output digest and reset.
		Packing.unpackIntBigEndian (H, 0, digest, 0, 8);
		reset();
		}

	/**
	 * Reset this hash function. Any accumulated message bytes are discarded,
	 * and the digest computation starts afresh.
	 */
	public void reset()
		{
		byteCount = 0;
		H[0] = 0x6a09e667;
		H[1] = 0xbb67ae85;
		H[2] = 0x3c6ef372;
		H[3] = 0xa54ff53a;
		H[4] = 0x510e527f;
		H[5] = 0x9b05688c;
		H[6] = 0x1f83d9ab;
		H[7] = 0x5be0cd19;
		}

// Hidden operations.

	// SHA-256 functions.
	private static int Ch (int x, int y, int z)
		{
		return (x & y) ^ (~x & z);
		}
	private static int Maj (int x, int y, int z)
		{
		return (x & y) ^ (x & z) ^ (y & z);
		}
	private static int Sigma_0 (int x)
		{
		return Integer.rotateRight (x, 2) ^
			Integer.rotateRight (x, 13) ^
			Integer.rotateRight (x, 22);
		}
	private static int Sigma_1 (int x)
		{
		return Integer.rotateRight (x, 6) ^
			Integer.rotateRight (x, 11) ^
			Integer.rotateRight (x, 25);
		}
	private static int sigma_0 (int x)
		{
		return Integer.rotateRight (x, 7) ^
			Integer.rotateRight (x, 18) ^
			(x >>> 3);
		}
	private static int sigma_1 (int x)
		{
		return Integer.rotateRight (x, 17) ^
			Integer.rotateRight (x, 19) ^
			(x >>> 10);
		}

	// SHA-256 constants.
	private static int[] K = new int[]
		{
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 
		0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 
		0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 
		0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 
		0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 
		0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 
		0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2, 
		};

	// SHA-256 compression function.
	private void compress()
		{
		// Prepare the message schedule W.
		Packing.packIntBigEndian (messageBlock, 0, W, 0, 16);
		for (int t = 16; t <= 63; ++ t)
			W[t] = sigma_1(W[t-2]) + W[t-7] + sigma_0(W[t-15]) + W[t-16];

		// Initialize working variables.
		int a = H[0];
		int b = H[1];
		int c = H[2];
		int d = H[3];
		int e = H[4];
		int f = H[5];
		int g = H[6];
		int h = H[7];

		// Do 64 rounds.
		int T_1, T_2;
		for (int t = 0; t <= 63; ++ t)
			{
			T_1 = h + Sigma_1(e) + Ch(e,f,g) + K[t] + W[t];
			T_2 = Sigma_0(a) + Maj(a,b,c);
			h = g;
			g = f;
			f = e;
			e = d + T_1;
			d = c;
			c = b;
			b = a;
			a = T_1 + T_2;
			}

		// Output chaining value.
		H[0] += a;
		H[1] += b;
		H[2] += c;
		H[3] += d;
		H[4] += e;
		H[5] += f;
		H[6] += g;
		H[7] += h;
		}

// Unit test main program.

//	/**
//	 * Unit test main program. Test vectors from:
//	 * http://csrc.nist.gov/groups/ST/toolkit/documents/Examples/SHA256.pdf
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		SHA256 sha256 = new SHA256();
//		byte[] db = new byte [32];
//		String d;
//		for (int i = 0; i < testMessage.length; ++ i)
//			{
//			sha256.hash (testMessage[i].getBytes ("US-ASCII"));
//			sha256.digest (db);
//			d = Hex.toString (db);
//			System.out.printf ("Message = \"%s\"%n", testMessage[i]);
//			System.out.printf ("Digest  = %s%n", d);
//			System.out.printf ("Correct = %s%n", testDigest[i]);
//			if (d.compareTo (testDigest[i]) == 0)
//				System.out.printf ("Correct%n");
//			else
//				System.out.printf ("WRONG%n");
//			}
//		}
//
//	private static String[] testMessage = new String[]
//		{
//		"abc",
//		"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
//		};
//
//	private static String[] testDigest = new String[]
//		{
//		"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
//		"248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
//		};

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		{
//		if (args.length != 1) usage();
//		byte[] digest = new byte [32];
//		SHA256 hash = new SHA256();
//		hash.hash (Hex.toByteArray (args[0]));
//		hash.digest (digest);
//		System.out.printf ("%s%n", Hex.toString (digest));
//		}
//
//	/**
//	 * Print a usage message and exit.
//	 */
//	private static void usage()
//		{
//		System.err.println ("Usage: java edu.rit.crypto.SHA256 <message>");
//		System.err.println ("<message> = Message bytes (hexadecimal)");
//		System.exit (1);
//		}

	}
