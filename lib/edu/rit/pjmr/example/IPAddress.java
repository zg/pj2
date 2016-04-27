//******************************************************************************
//
// File:    IPAddress.java
// Package: edu.rit.pjmr.example
// Unit:    Class edu.rit.pjmr.example.IPAddress
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

package edu.rit.pjmr.example;

import java.io.Serializable;
import java.util.Scanner;

/**
 * Class IPAddress encapsulates a textual IP address.
 *
 * @author  Alan Kaminsky
 * @version 19-Nov-2013
 */
public class IPAddress
	implements Comparable<IPAddress>, Serializable
	{

	private int addr;

	/**
	 * Construct a new IP address.
	 *
	 * @param  s  IP address string in dotted decimal notation.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> is not a valid IP address.
	 */
	public IPAddress
		(String s)
		{
		Scanner scanner = new Scanner (s);
		scanner.useDelimiter ("\\.");
		addr = 0;
		for (int i = 0; i < 4; ++ i)
			{
			if (! scanner.hasNextInt())
				throw new IllegalArgumentException (String.format
					("IPAddress(): s = \"%s\" illegal", s));
			int b = scanner.nextInt();
			if (0 > b || b > 255)
				throw new IllegalArgumentException (String.format
					("IPAddress(): s = \"%s\" illegal", s));
			addr = (addr << 8) | b;
			}
		if (scanner.hasNext())
			throw new IllegalArgumentException (String.format
				("IPAddress(): s = \"%s\" illegal", s));
		}

	/**
	 * Determine if this IP address is equal to the given object.
	 *
	 * @param  obj  Object to test.
	 *
	 * @return  True if this IP address equals <TT>obj</TT>, false otherwise.
	 */
	public boolean equals
		(Object obj)
		{
		return
			(obj instanceof IPAddress) &&
			(this.addr == ((IPAddress)obj).addr);
		}

	/**
	 * Returns a hash code for this IP address.
	 *
	 * @return  Hash code.
	 */
	public int hashCode()
		{
		return addr;
		}

	/**
	 * Returns a string version of this IP address.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return String.format ("%d.%d.%d.%d",
			(addr >> 24) & 0xFF,
			(addr >> 16) & 0xFF,
			(addr >>  8) & 0xFF,
			(addr      ) & 0xFF);
		}

	/**
	 * Compare this IP address to the given IP address.
	 *
	 * @param  ipaddr  IP address to compare.
	 *
	 * @return  A number less than, equal to, or greater than 0 if this IP
	 *          address is less than, equal to, or greater than <TT>ipaddr</TT>.
	 */
	public int compareTo
		(IPAddress ipaddr)
		{
		long diff =
			(this.addr   & 0x00000000FFFFFFFFL) -
			(ipaddr.addr & 0x00000000FFFFFFFFL);
		return diff < 0 ? -1 : diff > 0 ? 1 : 0;
		}

	}
