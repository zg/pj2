//******************************************************************************
//
// File:    ChunkOutput.java
// Package: edu.rit.image.png
// Unit:    Interface edu.rit.image.png.ChunkOutput
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

package edu.rit.image.png;

import java.io.IOException;

/**
 * Interface ChunkOutput specifies the interface for an object to which PNG
 * image file chunks can be written.
 *
 * @author  Alan Kaminsky
 * @version 22-May-2013
 */
public interface ChunkOutput
	{

// Exported operations.

	/**
	 * Write the given chunk to this chunk output object.
	 *
	 * @param  chunk  Chunk.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void write
		(Chunk chunk)
		throws IOException;

	}
