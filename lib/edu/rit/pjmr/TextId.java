//******************************************************************************
//
// File:    TextId.java
// Package: edu.rit.pjmr
// Unit:    Class edu.rit.pjmr.TextId
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

package edu.rit.pjmr;

import java.io.File;

/**
 * Class TextId provides a data record ID returned by a {@linkplain
 * TextFileSource}. The data record ID consists of a file and a line number.
 *
 * @author  Alan Kaminsky
 * @version 06-Nov-2014
 */
public class TextId
	{

// Exported data members.

	/**
	 * File.
	 */
	public final File file;

	/**
	 * Line number within the file. Line numbers start at 1.
	 */
	public final long lineNumber;

// Exported constructors.

	/**
	 * Construct a new text ID object.
	 *
	 * @param  file        File.
	 * @param  lineNumber  Line number.
	 */
	public TextId
		(File file,
		 long lineNumber)
		{
		this.file = file;
		this.lineNumber = lineNumber;
		}

	/**
	 * Returns a string version of this text ID object.
	 *
	 * @return  String version.
	 */
	public String toString()
		{
		return file + " line " + lineNumber;
		}

	/**
	 * Determine if this text ID object is equal to the given object.
	 *
	 * @param  obj  Object to test.
	 *
	 * @return  True if this text ID object is equal to <TT>obj</TT>, false
	 *          otherwise.
	 */
	public boolean equals
		(Object obj)
		{
		return
			(obj instanceof TextId) &&
			(this.file.equals (((TextId)obj).file)) &&
			(this.lineNumber == ((TextId)obj).lineNumber);
		}

	/**
	 * Returns a hash code for this text ID object.
	 *
	 * @return  Hash code.
	 */
	public int hashCode()
		{
		return file.hashCode() + (int)(lineNumber >> 32) + (int)lineNumber;
		}

	}
